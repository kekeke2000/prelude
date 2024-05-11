# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
import os
import sys
from functools import partial
import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.utils.file_utils import policy_from_checkpoint
from robomimic.utils.log_utils import PrintLogger
from robomimic.envs.env_base import EnvType
from robomimic.utils.dataset import SequenceDataset

from wkwrepro.scrips.path import *
import time
import rospy
from geometry_msgs.msg import Twist
import h5py
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray


   
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





## Train Navigation Controller
def evaluate( eval_policy=None, random_arrayd=None, random_array=None,save_path=None, config_path=None, render=True,  seed=0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    """
    Train a model using the algorithm.
    """

  

    # 假设你的权重文件的路径是 weights_path
   

    


    eval_policy1 =eval_policy
        
    # 记录开始时间
    start_time = time.time()

    # random_array = np.random.rand(120, 212, 3)
    # random_arrayd = np.random.rand(120, 212, 1)

    obs_dict = {
            "agentview_rgb": np.transpose(random_array, (2, 0, 1)),
            "agentview_depth": np.transpose(random_arrayd, (2, 0, 1)),
            "yaw": np.array([0])
        }




    action = eval_policy1(obs_dict)
    print(action)
    vel_msg=Twist()
    vel_msg.linear.x = action[0]
    vel_msg.angular.z = action[1]
    cmd_vel_sub.publish(vel_msg)
    end_time = time.time()

            # 计算并打印运行时间，单位为毫秒
    execution_time = (end_time - start_time) * 1000
    print(f"Execution time: {execution_time} ms")

    return eval_policy1




def sync_callback(eval_policy,depth, rgb):
    arr = np.array(depth.data)

    # 打印原始数组
    print("Original array: ", len(arr))
    # 将arr reshape为(424, 512)
    arr = arr.reshape((424, 512))
    print("Reshaped array: ", arr.shape)

    # 将arr resize为(120, 212)
    arr = cv2.resize(arr, (212, 120))
    arr_reshaped = arr.reshape((120, 212, 1))/1000

    print("Resized array: ", np.amax(arr_reshaped))
    



    # 处理RGB图像数据
    cv_image = imgmsg_to_cv2(rgb)
    resizer = cv2.resize(cv_image, (212, 120))
    # 将图像数据转换为NumPy数组
    np_image = np.array(resizer)
    print(np_image.shape)
    # 将NumPy数组添加到列表中
    evaluate(eval_policy,arr_reshaped , np_image)
    rospy.loginfo("收到RGB图像")



if __name__ == '__main__':
#    bridge = CvBridge()
   
   nav_path  = "/home/tmp/wkw/PRELUDE/PRELUDE/pretrained_bcrnn/models/model_best_training.pth"
   eval_policy = policy_from_checkpoint(ckpt_path=nav_path)[0]
   print("\n============= Model Summary =============")
   print(eval_policy)
   rospy.init_node("cmd_vel_data")
   cmd_vel_sub = rospy.Publisher("/cmd_vel", Twist)
    # # 订阅RGB图像数据主题
   rgb_sub = message_filters.Subscriber("/kinect2/hd/image_color_rect", Image)
   depth_sub = message_filters.Subscriber("/numpy_image", Float32MultiArray)
    
    # # 订阅深度图像数据主题
   sync = message_filters.ApproximateTimeSynchronizer(
                        [depth_sub, rgb_sub], queue_size=4, slop=0.5, allow_headerless=True
                    )
#    sync.registerCallback(partial(sync_callback, eval_policy))
   print("sync_callback")
   sync.registerCallback(lambda depth, rgb: sync_callback(eval_policy,depth, rgb))

    # 加载权重
   while  not rospy.is_shutdown():
        rospy.spin()    




