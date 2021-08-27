#!/usr/bin/python3

import sys
import time
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

import cv2
import cv_bridge
import numpy as np
import time

import rospy
import message_filters

class projection_node:

    def __init__(self):
        self.source_topic = "/camera/rgb/image_color"
        self.depth_topic = "/camera/depth/image_raw"
        self.img_subscriber = message_filters.Subscriber(self.source_topic, Image, queue_size=5, buff_size=2**24)
        self.depth_subscriber = message_filters.Subscriber(self.depth_topic, Image, queue_size=5, buff_size=2**24)
        self.bridge = cv_bridge.CvBridge()
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_subscriber, self.depth_subscriber], queue_size=1, slop=0.1)
        self.ts.registerCallback(self.projection_callback)
        rospy.loginfo("Initialised the image subscriber successfully!")
    def projection_callback(self, rgb_image, depth_image):
        rgb_camera_matrix = np.array([[520.055928,   0.000000, 312.535255], \
                                      [  0.000000, 520.312173, 242.265554], \
                                      [  0.000000,   0.000000,   1.000000]])
        
        rgb_distortion = np.array([0.169160,  -0.311272, -0.014471, -0.000973, 0.000000])
        
        rgb_projection = np.array([[532.087158,   0.000000, 311.450990, 0.000000], \
                                   [  0.000000, 532.585205, 237.199006, 0.000000], \
                                   [  0.000000,   0.000000,   1.000000, 0.000000]])
        
        depth_camera_matrix = np.array([[519.9970609150881,   0.0            , 312.1825832030777], \
                                        [  0.0            , 519.9169979264075, 256.9132353905213], \
                                        [  0.0            ,   0.0            ,   1.0            ]])

        depth_distortion = np.array([0.1309893410538465, -0.2220648862292244, -0.0007491207145344614, -0.001087706204362299, 0.0])
        
        depth_projection = np.array([[529.3626708984375, 0.0,            311.3649876060372, 0.0], \
                                     [  0.0,           530.262939453125, 256.6228139598825, 0.0], \
                                     [  0.0,             0.0,              1.0,             0.0]])
        depth_to_rgb = np.array([[1, 0, 0, -0.0254 ],\
                                [0, 1, 0, -0.00013],\
                                [0, 0, 1, -0.00218],\
                                [0, 0, 0,  1      ]])


        rospy.loginfo("Got images")
        depth_image.encoding = "mono16"
        depth_img = self.bridge.imgmsg_to_cv2(depth_image, 'mono16')
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_image, 'bgr8')
        cv2.imshow('depth', depth_img)
        cv2.imshow('color', rgb_img)
        cv2.waitKey(0)
        
        import pdb; pdb.set_trace()
        
 
if __name__ == '__main__':
    rospy.init_node("projection_node")
    projection = projection_node()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit()

        
