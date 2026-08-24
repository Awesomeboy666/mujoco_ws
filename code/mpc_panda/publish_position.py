import numpy as np
import rospy
from geometry_msgs.msg import Point

rospy.init_node("position_publisher")

pub = rospy.Publisher(
    "/target_position",
    Point,
    queue_size=10
)

rate = rospy.Rate(10)

msg = Point()

while not rospy.is_shutdown():
    
    print("x≈0.25 - 0.70 m,y≈−0.40∼0.40 m,z≈0.15∼0.75 m")

    msg.x = float(input("请输入目标 x: "))
    msg.y = float(input("请输入目标 y: "))
    msg.z = float(input("请输入目标 z: "))

    pub.publish(msg)

    rospy.loginfo(f"发布位置：{msg}")

    rate.sleep()
