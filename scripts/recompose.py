#!/usr/bin/env python

from attention_package.msg import FoveatedImageCombined, FoveatedImageMeta, Tuple
from yolov5_detector.msg import DetectionMsg, DetectionArray

import numpy as np
import rospy
import ros_numpy
import time
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
import PIL.Image
import cv_bridge
import cv2
from attention_utils.common import create_dir_and_return_path

'''
FOVEATEDIMAGEGROUPS
Header header

uint16 height
uint16 width

uint8 detected_objects
uint8 foveation_level
sensor_msgs/Image rgb_image

FoveatedImages[] foveated_images_groups


FOVEATEDIMAGES
attention_package/FoveatedImage[] foveated_images


FOVEATEDIMAGE
sensor_msgs/Image foveated_image
attention_package/Tuple bounding_box_origins
attention_package/Tuple bounding_box_sizes
'''

class recompose_node:

    def __init__(self):
        publish_topic = rospy.get_param("~publish_topic")
        foveation_topic = rospy.get_param("~foveation_topic")
        self.publisher = rospy.Publisher(publish_topic, Image, queue_size=5)
        self.subscriber = rospy.Subscriber(foveation_topic, FoveatedImageGroups, self.recompose_callback)
        self.bridge = cv_bridge.CvBridge()
        self.save_img = rospy.get_param("~save_img")
        self.show_img = rospy.get_param("~show_img")
        self.save_rgb = rospy.get_param("~save_rgb")
        self.save_path = rospy.get_param("~save_path") + "recompose_imgs"
        self.callback_counter = 0
        if(self.save_img):
            self.save_path = create_dir_and_return_path(self.save_path)
        init_string = "Initialised recompose_node with the following parameters:\n" + \
                    f"foveation_topic: {foveation_topic}\n" + \
                    f"publish_topic: {publish_topic}\n" + \
                    f"save_img: {self.save_img}\n" + \
                    f"save_rgb: {self.save_rgb}\n" + \
                    f"show_img: {self.show_img}\n" + \
                    f"save_path: {self.save_path}"

                        

        rospy.loginfo(init_string)

    def recompose_callback(self, data):
        print('Got a foveation message. Recomposing images...')
        start = time.time()
        detection_count = data.detected_objects
        fov_level = data.foveation_level
        print(detection_count)
        recomposed_img_arr = []
        # combined_img = np.zeros((data.height, data.width), dtype=np.uint16)
        for f in range(fov_level - 1, -1, -1): # we want to go in reverse, since the last index contains the lowest resolution image.
            if(self.save_rgb and self.save_img):
                recomposed_img = np.zeros((data.height, data.width, 3), dtype=np.uint8)
            else:
                recomposed_img = np.zeros((data.height, data.width), dtype=np.uint16)
            for i in range(0, detection_count): # and we want to loop through at the same foveation level, across detected objects.
                print(detection_count)
                curr_img = data.foveated_images_groups[i].foveated_images[f].foveated_image
                curr_img = cv2.imdecode(curr_img, cv2.IMREAD_ANYDEPTH)
                cv2.imshow("curr_img", curr_img)
                cv2.waitKey(0)
                curr_bb_origin = data.foveated_images_groups[i].foveated_images[f].bounding_box_origins.tpl
                curr_bb_end = data.foveated_images_groups[i].foveated_images[f].bounding_box_sizes.tpl
                resize_height = curr_bb_end[0] - curr_bb_origin[0]
                resize_width = curr_bb_end[1] - curr_bb_origin[1]

                if(self.save_rgb and self.save_img):
                    curr_img = self.bridge.imgmsg_to_cv2(curr_img, 'rgb8')
                else:
                    curr_img.encoding = 'mono16'
                    curr_img = self.bridge.imgmsg_to_cv2(curr_img, 'mono16')
                curr_img = cv2.resize(curr_img, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)

                # at the highest foveation level, there are lots of empty patches, so a bitwise OR is better than straight up replacement.
                try:
                    if(self.save_rgb and self.save_img):
                        recomposed_img[curr_bb_origin[0]:curr_bb_end[0], curr_bb_origin[1]:curr_bb_end[1], :] = \
                            cv2.bitwise_or(recomposed_img[curr_bb_origin[0]:curr_bb_end[0], curr_bb_origin[1]:curr_bb_end[1], :], curr_img)
                    else:    
                        recomposed_img[curr_bb_origin[0]:curr_bb_end[0], curr_bb_origin[1]:curr_bb_end[1]] = \
                            cv2.bitwise_or(recomposed_img[curr_bb_origin[0]:curr_bb_end[0], curr_bb_origin[1]:curr_bb_end[1]], curr_img)
                    #combined_img[curr_bb_origin[0]:curr_bb_end[0], curr_bb_origin[1]:curr_bb_end[1]] = \
                    #    cv2.bitwise_or(combined_img[curr_bb_origin[0]:curr_bb_end[0], curr_bb_origin[1]:curr_bb_end[1]], curr_img)
                except:
                    #breakpoint()
                    pass
                    

                # coloured_depth = ((recomposed_img / np.max(recomposed_img))*256).astype(np.uint8)
                # coloured_depth = cv2.applyColorMap(coloured_depth, cv2.COLORMAP_HSV)
                # cv2.imshow('recomposed, curr_img', coloured_depth)
                # cv2.waitKey(0)
            recomposed_img_arr.append(recomposed_img.copy())

        
        if(self.save_img and self.save_rgb): # rgb img
            recomposed_img = np.zeros((data.height, data.width, 3), dtype=np.uint8)
        else:
            recomposed_img = np.zeros((data.height, data.width), dtype=np.uint16)
        
        for i in range(0, fov_level):
            
            recomposed_img = cv2.bitwise_or(recomposed_img, recomposed_img_arr[i])

            if(self.save_img and not self.save_rgb): # depth img
                print('this should be called')
                coloured_depth = ((recomposed_img_arr[i] / np.max(recomposed_img_arr[i]))*256).astype(np.uint8)
                coloured_depth = cv2.applyColorMap(coloured_depth, cv2.COLORMAP_HSV)
                PIL.Image.fromarray(coloured_depth).save(f"{self.save_path}recomposed_img{self.callback_counter}__{i}_{f}.png")

                combined_coloured = ((recomposed_img / np.max(recomposed_img_arr[i]))*256).astype(np.uint8)
                combined_coloured = cv2.applyColorMap(combined_coloured, cv2.COLORMAP_HSV)
                PIL.Image.fromarray(combined_coloured).save(f"{self.save_path}combined_img{self.callback_counter}__{i}_{f}.png")
            
            if(self.save_img and self.save_rgb): # rgb img
                PIL.Image.fromarray(recomposed_img_arr[i], 'RGB').save(f"{self.save_path}recomposed_img_rgb{self.callback_counter}__{i}_{f}.png")

                PIL.Image.fromarray(recomposed_img, 'RGB').save(f"{self.save_path}combined_img_rgb{self.callback_counter}__{i}_{f}.png")
            

            if(self.show_img and not self.save_rgb): # depth img
                coloured_depth = ((recomposed_img_arr[i] / np.max(recomposed_img_arr[i]))*256).astype(np.uint8)
                coloured_depth = cv2.applyColorMap(coloured_depth, cv2.COLORMAP_HSV)
                cv2.imshow(f'img_fov_lvl{i}', coloured_depth)
                
                combined_coloured = ((recomposed_img / np.max(recomposed_img_arr[i]))*256).astype(np.uint8)
                combined_coloured = cv2.applyColorMap(combined_coloured, cv2.COLORMAP_HSV)
                cv2.imshow(f'combined', combined_coloured)

        #       combined_img_colour = ((combined_img / np.max(recomposed_img_arr[i]))*256).astype(np.uint8)
        #       combined_img_colour = cv2.applyColorMap(combined_img_colour, cv2.COLORMAP_HSV)
        #       cv2.imshow(f'other combined', combined_img_colour)
                cv2.waitKey(1)
        print(f"Done recomposing! that took {time.time() - start} sec")
        self.callback_counter += 1


if __name__ == '__main__':
    rospy.init_node("recompose_node")
    recompose = recompose_node()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit()

