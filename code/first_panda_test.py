import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path(
    "../mujoco_menagerie/franka_emika_panda/scene_motor.xml"
)

data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)


# 找到 hand 的 body id
body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)


# 创建 Jacobian
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))


# 计算 hand body 原点的 Jacobian
mujoco.mj_jacBody(
    model,
    data,
    jacp,
    jacr,
    body_id
)


# Panda 机械臂只取前7个关节
Jp = jacp[:, :7]
Jr = jacr[:, :7]

v_const = 0.1

hand_position = np.zeros(3)
hand_position[0] = v_const
#算出qdot
qvel_d = np.linalg.pinv(Jp) @ hand_position

e = np.zeros(7)
int_e = np.zeros(7)
kp = 15.0
ki = 10.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        #得到前七个关节的雅可比矩阵
        mujoco.mj_jacBody(
            model,
            data,
            jacp,
            jacr,
            body_id
        )
        Jp = jacp[:, :7]
        #算出q
        qvel_d = np.linalg.pinv(Jp) @ hand_position
        #误差
        e = qvel_d - data.qvel[:7]
        int_e += e * model.opt.timestep
        #把重力考虑进去
        bias = data.qfrc_bias[:7].copy()
        #力矩 = PI控制 + g（q）
        u = e*kp + int_e*ki + bias

        data.ctrl[:7] = u
        mujoco.mj_step(model, data)
        print("hand position =", data.xpos[body_id])
        viewer.sync()
        
        time.sleep(model.opt.timestep)



