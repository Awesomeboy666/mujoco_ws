import mujoco
import mujoco.viewer
import numpy as np
import time


# 加载模型
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
#雅可比矩阵缓冲区
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

# 记录初始末端位姿
R_d = data.xmat[body_id].reshape(3, 3).copy()

# 末端任务参数
p_posgoal_new = np.zeros(3)  #根据输入的坐标作为目标
p_posgoal_old = data.xpos[body_id].copy()

kp_task = 2.0  # 末端位置反馈增益
kp_rot = 2.0   # 末端姿态反馈增益


# Panda 关节限位
q_min = np.array([
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973
])

q_max = np.array([
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973
])

q_mid = (q_min + q_max) / 2.0

k_null = 1.0  # 零空间关节限位回避增益
state = 2 #1 is go  2 is wait
T = 2
t = 0
# 关节速度 PI
kp = 15.0
ki = 10.0

int_e = np.zeros(7)


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        if state == 2 :
            print("x≈0.25 - 0.70 m,y≈−0.40∼0.40 m,z≈0.15∼0.75 m")
            x = float(input("请输入目标 x: "))
            y = float(input("请输入目标 y: "))
            z = float(input("请输入目标 z: "))
            p_posgoal_new = np.array([x, y, z])
            int_e[:] = 0.0
            state = 1
            t = 0

        t += model.opt.timestep

        # 计算当前 Jacobian
        mujoco.mj_jacBody(
            model,
            data,
            jacp,
            jacr,
            body_id
        )
        Jp = jacp[:, :7]
        Jr = jacr[:, :7]
        J = np.vstack((Jp, Jr))  # 完整 6×7 Jacobian


        # 末端位置控制
        p = data.xpos[body_id].copy()
        if t <= T:
            s = t/T
            p_path = p_posgoal_old + (p_posgoal_new - p_posgoal_old)* (
                                    10 * s**3 - 15 * s**4 + 6 * s**5)
            p_path_vel = (p_posgoal_new - p_posgoal_old)/ T * (
                                    30 * s**2 - 60 * s**3 + 30 * s**4)
        else:
            p_path = p_posgoal_new.copy()  # 规划结束后保持目标位置
            p_path_vel = np.zeros(3)       # 目标速度归零

        e_pos = p_path - p
        v_cmd = kp_task * e_pos + p_path_vel

        # 末端姿态控制
        R = data.xmat[body_id].reshape(3, 3)  #reshape 是啥意思

        R_err = R_d.T @ R - R.T @ R_d

        e_R = 0.5 * np.array([
            R_err[2, 1],
            R_err[0, 2],
            R_err[1, 0]
        ])
        omega_cmd = -kp_rot * e_R

        # 任务空间速度
        xvel_cmd = np.concatenate((v_cmd, omega_cmd))

        # 主任务逆运动学
        J_pinv = np.linalg.pinv(J)
        qvel_task = J_pinv @ xvel_cmd  # 主任务关节速度


        # 零空间关节限位回避
        q = data.qpos[:7]

        grad_H = 2.0 * (q - q_mid) / (q_max - q_min)**2  # 代价函数梯度

        qvel_null = -k_null * grad_H  # 沿负梯度方向运动

        N = np.eye(7) - J_pinv @ J  # 零空间投影矩阵

        qvel_d = qvel_task + N @ qvel_null  # 主任务 + 次任务


        # 关节速度 PI
        e = qvel_d - data.qvel[:7]

        int_e += e * model.opt.timestep

        bias = data.qfrc_bias[:7].copy()

        tau = kp * e + ki * int_e + bias

        data.ctrl[:7] = tau

        # 推进仿真
        mujoco.mj_step(model, data)

        check_goal = pow(((p_posgoal_new[0] - data.xpos[body_id][0])**2 
                            + (p_posgoal_new[1] - data.xpos[body_id][1])**2
                            + (p_posgoal_new[2] - data.xpos[body_id][2])**2),0.5)
        
        if check_goal < 0.01:
            state = 2
            p_posgoal_old = data.xpos[body_id].copy()

        # 打印观察
        print(
            "p =", np.round(data.xpos[body_id], 2),
        )
        #hhhhhhhhhhhhh
        viewer.sync()

        time.sleep(model.opt.timestep)