import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

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
q0 = data.qpos[:7].copy()

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
T = 3
t = 0

# 关节速度 PI
kp = 20.0
ki = 10.0
int_e = np.zeros(7)

# 关节位置 PD控制
kd = 15.0
e_old = 0.0
qpos_d = q0.copy()

print("x≈0.25 - 0.70 m,y≈−0.40∼0.40 m,z≈0.15∼0.75 m")
x = float(input("请输入目标 x: "))
y = float(input("请输入目标 y: "))
z = float(input("请输入目标 z: "))
p_posgoal_new = np.array([x, y, z])

new_goal_ready = False
input_goal = None
show =True

# 键盘输入线程
def input_thread():
    global new_goal_ready, input_goal,show
    while True:
        if show == False:
            try:
                print("x≈0.25 - 0.70 m,y≈−0.40∼0.40 m,z≈0.15∼0.75 m")
                x = float(input("请输入目标 x: "))
                y = float(input("请输入目标 y: "))
                z = float(input("请输入目标 z: "))
                input_goal = np.array([x, y, z])

                new_goal_ready = True
                show = True  # 已经收到新目标，先停止继续询问
            except ValueError:
                print("输入格式错误，请输入三个数字，例如：0.5 0.1 0.6")
        else:
            time.sleep(0.05)


threading.Thread(
    target=input_thread,
    daemon=True
).start()

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        if new_goal_ready:
            p_posgoal_old = data.xpos[body_id].copy()  # 从当前实际位置重新规划
            p_posgoal_new = input_goal.copy()

            t = 0.0
            int_e[:] = 0.0

            new_goal_ready = False
            
            

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

        '''# 末端姿态控制   不要末端姿态控制了
        R = data.xmat[body_id].reshape(3, 3)  #reshape 是啥意思

        R_err = R_d.T @ R - R.T @ R_d

        e_R = 0.5 * np.array([
            R_err[2, 1],
            R_err[0, 2],
            R_err[1, 0]
        ])
        omega_cmd = -kp_rot * e_R'''

        # 任务空间速度
        #xvel_cmd = np.concatenate((v_cmd, omega_cmd))

        # 主任务逆运动学
        J_pinv = np.linalg.pinv(Jp)
        #qvel_task = J_pinv @ xvel_cmd  # 主任务关节速度
        qvel_task = J_pinv @ v_cmd

        # 零空间关节限位回避
        q = data.qpos[:7]

        grad_H = 2.0 * (q - q_mid) / (q_max - q_min)**2  # 代价函数梯度

        qvel_null = -k_null * grad_H  # 沿负梯度方向运动

        N = np.eye(7) - J_pinv @ Jp  # 零空间投影矩阵

        qvel_d = qvel_task + N @ qvel_null  # 主任务 + 次任务


        '''# 关节速度 PI
        e = qvel_d - data.qvel[:7]
        int_e += e * model.opt.timestep
        bias = data.qfrc_bias[:7].copy()
        tau = kp * e + ki * int_e + bias
        data.ctrl[:7] = tau'''

        #关节位置控制PD
        qpos_d += qvel_d * model.opt.timestep
        e = qpos_d - data.qpos[:7]
        bias = data.qfrc_bias[:7].copy()
        de_e = (e - e_old)/model.opt.timestep
        e_old = e
        tau = kp * e + kd * de_e + bias
        data.ctrl[:7] = tau
        

        # 推进仿真
        mujoco.mj_step(model, data)

        check_goal = pow(((p_posgoal_new[0] - data.xpos[body_id][0])**2 
                            + (p_posgoal_new[1] - data.xpos[body_id][1])**2
                            + (p_posgoal_new[2] - data.xpos[body_id][2])**2),0.5)
        
        if check_goal < 0.01:
            show = False
            
            
        if show == True:
            # 打印观察
            print(
                "p =", np.round(data.xpos[body_id], 2),
            )

        viewer.sync()

        time.sleep(model.opt.timestep)