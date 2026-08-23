import mujoco
import mujoco.viewer
import casadi as ca
import numpy as np
import time
def quat_to_rot(quat):
    quat = np.asarray(quat, dtype=float)
    quat = quat / np.linalg.norm(quat)

    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    R = np.array([
        [
            1 - 2 * (y**2 + z**2),
            2 * (x*y - z*w),
            2 * (x*z + y*w)
        ],
        [
            2 * (x*y + z*w),
            1 - 2 * (x**2 + z**2),
            2 * (y*z - x*w)
        ],
        [
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x**2 + y**2)
        ]
    ])

    return R

def fixed_transform(pos, quat):
    R = quat_to_rot(quat)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos

    return ca.DM(T)

def rot_z(q):
    c = ca.cos(q)
    s = ca.sin(q)

    return ca.vertcat(
        ca.horzcat(c, -s, 0, 0),
        ca.horzcat(s,  c, 0, 0),
        ca.horzcat(0,  0, 1, 0),
        ca.horzcat(0,  0, 0, 1)
    )

def panda_fk(q):
    T = ca.DM.eye(4)

    T1 = fixed_transform(
        [0.0, 0.0, 0.333],
        [1.0, 0.0, 0.0, 0.0]
    )

    T2 = fixed_transform(
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0]
    )

    T3 = fixed_transform(
        [0.0, -0.316, 0.0],
        [1.0, 1.0, 0.0, 0.0]
    )

    T4 = fixed_transform(
        [0.0825, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0]
    )

    T5 = fixed_transform(
        [-0.0825, 0.384, 0.0],
        [1.0, -1.0, 0.0, 0.0]
    )

    T6 = fixed_transform(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0]
    )

    T7 = fixed_transform(
        [0.088, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0]
    )

    T_hand = fixed_transform(
        [0.0, 0.0, 0.107],
        [0.9238795, 0.0, 0.0, -0.3826834]
    )

    T = T @ T1 @ rot_z(q[0])
    T = T @ T2 @ rot_z(q[1])
    T = T @ T3 @ rot_z(q[2])
    T = T @ T4 @ rot_z(q[3])
    T = T @ T5 @ rot_z(q[4])
    T = T @ T6 @ rot_z(q[5])
    T = T @ T7 @ rot_z(q[6])

    T = T @ T_hand

    p = T[:3, 3]

    return p

model = mujoco.MjModel.from_xml_path(
    "../mujoco_menagerie/franka_emika_panda/scene_motor.xml"
)

data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)
# MPC参数
N = 50
mpc_dt = 0.02

substeps = max(
    1,
    int(round(mpc_dt / model.opt.timestep))
)


dt = substeps * model.opt.timestep

opti = ca.Opti()
# 未来状态 7个q和7个qdot
X = opti.variable(14, N + 1)
# 未来期望加速度  7个tau
U = opti.variable(7, N)

# 当前真实状态
x0 = opti.parameter(14)
# 目标状态
xd = opti.parameter(3)

opti.subject_to(
    X[:, 0] == x0
)
q_symbol = ca.MX.sym(
    "q",
    7
)

p_symbol = panda_fk(
    q_symbol
)

fk = ca.Function(
    "fk",
    [q_symbol],
    [p_symbol]
)

# 动力学约束
for k in range(N):
    q = X[:7 , k]
    qdot = X[7:14, k]
    qddot = U[:7, k] 

    q_next = q + dt * qdot
    qdot_next = qdot + dt * qddot
    
    x_next = ca.vertcat(
        q_next,
        qdot_next
    )

    opti.subject_to(
        X[:, k + 1] == x_next
    )

Q = np.diag([1,1,1,1,1,1,1,1,1,1,1,1,1,1])

R = np.diag([1,1,1,1,1,1,1])

O = np.diag([10,10,10])

J = 0
p_posgoal_new = np.zeros(3)

for k in range(N):
    qk = X[:7, k + 1]

    p_k = fk(qk)

    e_pos = p_k - xd      #这是在构建优化问题，opti.set_value(这个才是修改

    J += (ca.mtimes([e_pos.T, O, e_pos])
        
        +ca.mtimes([U[:7, k].T, R, U[:7, k]])
    )

acc_max = np.array([
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0
])

for i in range(7):
    opti.subject_to(
        opti.bounded(
            -acc_max[i],
            U[i, :],
            acc_max[i]
        )
    )

opti.minimize(J)

opti.solver(
    "ipopt",
    {
        "print_time": False
    },
    {
        "print_level": 0
    }
)

print("x≈0.25 - 0.70 m,y≈−0.40∼0.40 m,z≈0.15∼0.75 m")
x = float(input("请输入目标 x: "))
y = float(input("请输入目标 y: "))
z = float(input("请输入目标 z: "))
p_posgoal_new = np.array([x, y, z])

opti.set_value(
    xd,
    p_posgoal_new
)


M = np.zeros(
    (model.nv, model.nv)
)

first_solve = True

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        loop_start = time.perf_counter()

        mujoco.mj_forward(
            model,
            data
        )

        q_now = data.qpos[:7].copy()
        qdot_now = data.qvel[:7].copy()

        x_now = np.concatenate((
            q_now,
            qdot_now
        ))

        opti.set_value(
            x0,
            x_now
        )

        if first_solve:
            X_guess = np.tile(
                x_now.reshape(14, 1),
                (1, N + 1)
            )

            U_guess = np.zeros(
                (7, N)
            )

            opti.set_initial(
                X,
                X_guess
            )

            opti.set_initial(
                U,
                U_guess
            )

        sol = opti.solve()

        U_opt = np.asarray(
            sol.value(U),
            dtype=float
        ).reshape(7, N)

        X_opt = np.asarray(
            sol.value(X),
            dtype=float
        ).reshape(14, N + 1)

        # MPC只执行第一个期望加速度
        qddot_mpc =  U_opt[:, 0].copy()
        
        U_guess = np.hstack((
            U_opt[:, 1:],
            U_opt[:, -1:]
        ))

        X_guess = np.hstack((
            X_opt[:, 1:],
            X_opt[:, -1:]
        ))

        opti.set_initial(
            U,
            U_guess
        )

        opti.set_initial(
            X,
            X_guess
        )

        first_solve = False

        q = data.qpos[:7].copy()
        qdot = data.qvel[:7].copy()

        # MPC决定第1关节加速度
        qddot_cmd = np.zeros(model.nv)

        qddot_cmd[:7] = qddot_mpc


        mujoco.mj_fullM(
            model,
            data,
            M
        )

        # 逆动力学：tau = M*qddot + bias
        tau = (
            M @ qddot_cmd
            + data.qfrc_bias
        )

        data.ctrl[:7] = tau[:7]

        for _ in range(substeps):
            mujoco.mj_step(
                model,
                data
            )

        print(
                "p =",
                np.round(data.xpos[body_id], 4)
            )

        viewer.sync()

   

        elapsed = time.perf_counter() - loop_start

        if elapsed < dt:
            time.sleep(
                dt - elapsed
            )


