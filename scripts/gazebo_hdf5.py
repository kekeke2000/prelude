#!/usr/bin/env python3
# coding=utf-8

import rospy
from geometry_msgs.msg import Twist
import h5py
import numpy as np
from sensor_msgs.msg import Image
import message_filters


cmd_vel_data_list = []
rgb_data_list = []



# CMD_VEL 回调函数
def cmd_vel_callback(msg):
    linear_x = msg.linear.x
    cmd_vel_data_list.append(linear_x)
    rospy.loginfo("线速度 x = %.2f", linear_x)

def imgmsg_to_cv2(img_msg):
        # if img_msg.encoding != "bgr8":
        #     rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        # print("cvtColor")
        # if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv

def imgmsg_to_cv2d(img_msg):
        # if img_msg.encoding != "bgr8":
        #     rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 1), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        # print("cvtColor")
        # if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv
    


# RGB图像数据回调函数
def rgb_callback(msg):
    # 处理RGB图像数据
    cv_image = imgmsg_to_cv2(msg)
    # 将图像数据转换为NumPy数组
    np_image = np.array(cv_image)
    # 将NumPy数组添加到列表中
    rgb_data_list.append(np_image)
    rospy.loginfo("收到RGB图像")
    

# 深度图像数据回调函数
def depth_callback(msg):
    # 处理深度图像数据
    cv_image = imgmsg_to_cv2d(msg)
    # 将图像数据转换为NumPy数组
    np_image = np.array(cv_image)
    print(np_image)

    

def sync_callback(cmd_vel, rgb):
    rospy.loginfo("收到CMD_VEL, RGB, 深度图像数据")
    linear_x = cmd_vel.linear.x
    cmd_vel_data_list.append(linear_x)
    rospy.loginfo("线速度 x = %.2f", linear_x)
    # 处理RGB图像数据
    cv_image = imgmsg_to_cv2(rgb)
    # 将图像数据转换为NumPy数组
    np_image = np.array(cv_image)
    # 将NumPy数组添加到列表中
    rgb_data_list.append(np_image)
    rospy.loginfo("收到RGB图像")


# 主函数
if __name__ == "__main__":
    rospy.init_node("cmd_vel_data")
    
    # depth_sub = rospy.Subscriber("/kinect2/sd/image_depth_rect", Image, depth_callback)
    # # 订阅 CMD_VEL 数据话题
    cmd_vel_sub = message_filters.Subscriber("/cmd_vel", Twist)
    # # 订阅RGB图像数据主题
    rgb_sub = message_filters.Subscriber("/kinect2/hd/image_color_rect", Image)
    
    # # 订阅深度图像数据主题
    sync = message_filters.ApproximateTimeSynchronizer(
                        [cmd_vel_sub, rgb_sub], queue_size=4, slop=0.5, allow_headerless=True
                    )
    sync.registerCallback(sync_callback)
    rospy.loginfo("开始记录CMD_VEL数据...")
    # rate = rospy.Rate(1) 
    try:
    #     # while not rospy.is_shutdown():
    #     #     rate.sleep()
        
    #     # 使用rospy.spin()来保持监听，直到节点被显式关闭
        rospy.spin()
    except KeyboardInterrupt:
        # 用户中断执行(通常是通过Ctrl+C)
        pass
    finally:
        # 节点关闭时，将数据写入HDF5文件
        rospy.loginfo("正在将CMD_VEL数据保存到HDF5文件中...")
        hf = h5py.File('cmd.hdf5', 'w')
        hf.create_dataset('rgb_data', data=np.array(rgb_data_list))
        hf.create_dataset('cmd_vel_data', data=np.array(cmd_vel_data_list))
        hf.close()
        rospy.loginfo("CMD_VEL数据保存完毕.")
