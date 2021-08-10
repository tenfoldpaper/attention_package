#!/usr/bin/env python

from attention_package.msg import Tuple, FoveatedImageMeta, FoveatedImageCombined

from yolov5_detector.msg import DetectionMsg, DetectionArray

import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
import PIL.Image
import cv_bridge
import cv2
import time
from attention_utils.common import create_dir_and_return_path
'''
Topics in: 
- Bounding box of classified objects topic, of format 
    - bounding_box[]: whose len() will reveal the # of detected objects;
        - uintx[] bounding_box_origin
        - uintx[] bounding_box_size
- Color image from the RGBD camera, of format
    - image 
- Depth image from the RGBD camera, of format
    - image

Topic out: 
- FoveatedImageGroups

Logic:
Subscribe to the 3 topics above.

Callback should be tied to the image stream.

Bounding box data should be CONSUMED, and not stored in memory once it has been used for foveation.

If bounding box does not exist, then just set the foveation level to 1 by default, and compress the images accordingly.
Otherwise, use the provided foveation level, and 
    Compose the output message in a loop of [for object in detected_objects: for i in foveation_level]

'''

class foveation_node:
    def __init__(self):
        publish_topic = rospy.get_param("~publish_topic")
        rgb_publish_topic = rospy.get_param("~rgb_publish_topic")
        detection_topic = rospy.get_param("~detection_topic")
        self.publisher = rospy.Publisher(publish_topic, FoveatedImageMeta, queue_size=5)
        self.rgb_publisher = rospy.Publisher(rgb_publish_topic, FoveatedImageMeta, queue_size=5)
        self.subscriber = rospy.Subscriber(detection_topic, DetectionMsg, self.foveation_callback)
        self.bridge = cv_bridge.CvBridge()
        self.fov_level = int(rospy.get_param("~fov_level"))
        self.max_scale = int(rospy.get_param("~max_scale"))
        self.save_img = rospy.get_param("~save_img")
        self.save_rgb = rospy.get_param("~save_rgb")
        self.show_img = rospy.get_param("~show_img")
        self.save_path = rospy.get_param("~save_path") + "foveate_imgs"
        self.callback_counter = 0
        if(self.save_img):
            self.save_path = create_dir_and_return_path(self.save_path)

        init_string = "Initialised foveate_node with the following parameters:\n" + \
                    f"detection_topic: {detection_topic}\n" + \
                    f"publish_topic: {publish_topic}\n" + \
                    f"fov_level: {self.fov_level}\n" + \
                    f"max_scale: {self.max_scale}\n" + \
                    f"save_img: {self.save_img}\n" + \
                    f"show_img: {self.show_img}\n" + \
                    f"save_path: {self.save_path}"
                        
        rospy.loginfo(init_string)

    def xywh2ogsz(self, x):
        # Convert 1x4 array of xywh into top left yx - yx size (origin-size)
        y = np.copy(x)
        y[0] = x[1] - x[3] / 2 # top left y
        y[1] = x[0] - x[2] / 2 # top left x
        y[2] = (x[1] + x[3] / 2) - y[0] # y size
        y[3] = (x[0] + x[2] / 2) - y[1] # x size

        return y


    def identify_center_depth_range(self, img, bb_origin, bb_size):
        # Depth range should be configurable. 
        height = img.shape[0]
        width = img.shape[1]
        bb_range = (bb_origin[0] + bb_size[0] if bb_origin[0] + bb_size[0] < height else height - 1, \
            bb_origin[1] + bb_size[1] if bb_origin[1] + bb_size[1] < width else width - 1)

        crop_height = np.array(range(bb_origin[0], bb_range[0]))
        crop_width = np.array(range(bb_origin[1], bb_range[1]))
        crop_img = img[np.ix_(crop_height, crop_width)]
        hist_edges = np.histogram_bin_edges(img, bins=20) # we want to do the histogram partition based on the whole image, not just the cropped image.
        hist = np.histogram(crop_img, bins=hist_edges)

        # the range of depth for the center image lies between two range bins with highest number of pixels assigned to them.

        range_ind = np.argpartition(hist[0], -2)[-2:] # result is ascending-sorted, i.e. argmax is at the last index.
        range_ind_low = range_ind[0] if range_ind[0] < range_ind[1] else range_ind[1]
        range_ind_high = range_ind[1] if range_ind[1] > range_ind[0] else range_ind[0]
        return range_ind_low, range_ind_high, hist[1]

    def calculate_fovlevel_bb(self, bb_origin, bb_size, img_width, img_height, fovlevel):

        # we want a dynamically sized foveation boxes based on the location of the original bounding box and its size.

        # assuming it's (height, width)
        lower_bound = (bb_origin[0], bb_origin[1])
        upper_bound = (bb_origin[0] + bb_size[0] if bb_origin[0] + bb_size[0] < img_height else img_height-1, \
            bb_origin[1] + bb_size[1] if bb_origin[1] + bb_size[1] < img_width else img_width-1)

        # we could replace np.linspace to some nonlinear space if that improves performance, since this might result in
        # the outermost foveation being too small.
        lower_bounds = (np.linspace(lower_bound[0], 0, num=fovlevel).astype(int), np.linspace(lower_bound[1], 0, num=fovlevel).astype(int))
        upper_bounds = (np.linspace(upper_bound[0], img_height, num=fovlevel).astype(int), np.linspace(upper_bound[1], img_width, num=fovlevel).astype(int))
        lower_bounds = list(zip(lower_bounds[0], lower_bounds[1]))
        upper_bounds = list(zip(upper_bounds[0], upper_bounds[1]))
        

        fovlevel_bb = list(zip(lower_bounds, upper_bounds))
        return fovlevel_bb

    def calculate_fovlevel_depth(self, center_low, center_high, bins, fovlevel):

        fovlevel_depth = []
        
        lower_bounds = np.linspace(center_low, 0, num=fovlevel).astype(int)
        upper_bounds = np.linspace(center_high, len(bins) - 1, num=fovlevel).astype(int)
        
        fovlevel_depth = list(zip(lower_bounds, upper_bounds))
        return fovlevel_depth

    def foveation_callback(self, data):
        rospy.loginfo("Received data for foveation")
        start = time.time()
        # DetectionMsg format:
        # Header header
        # uint8 detection_count
        # sensor_msgs/Image rgb_image
        # sensor_msgs/Image depth_image
        # DetectionArray[] detection_array -> detection_info of float32[]
        # Convert the images into numpy format
        data.depth_image.encoding = ('mono16')
        #rgb_img = self.bridge.imgmsg_to_cv2(data.rgb_image, 'bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(data.depth_image, 'mono16')
        rgb_img = self.bridge.imgmsg_to_cv2(data.rgb_image, 'bgr8')
        img_height = depth_img.shape[0]
        img_width = depth_img.shape[1]
        
        # Initialize the message to be published

        fov_image_group = FoveatedImageMeta()
        fov_image_group.height = img_height
        fov_image_group.width = img_width
        fov_image_group.detected_objects = data.detection_count
        fov_image_group.foveation_level = self.fov_level
        fov_image_group.rgb_image = data.rgb_image
        
        if(self.save_rgb):
            fov_image_group_bgr = FoveatedImageMeta()
            fov_image_group_bgr.height = img_height
            fov_image_group_bgr.width = img_width
            fov_image_group_bgr.detected_objects = data.detection_count
            fov_image_group_bgr.foveation_level = self.fov_level
            fov_image_group_bgr.rgb_image = data.rgb_image
                

        _bb_origins = []
        _bb_sizes = []
        _det_classes = []
        _center_lows = []
        _center_highs = []
        _bins = []
        _fovlevel_bbs = []
        _fovlevel_depths = []
        _scale_values = []
        
        start = time.time()
        for i in range(0, data.detection_count):
            # Take out all the high-res images first.
            # The goal is to minimize the number of times we need to rescale the image.
            
            # Gather all the bounding box-related data of all the detected objects.
            
            _det_classes.append(data.detection_array[i].detection_info[0]) # first element 
            bb_raw = data.detection_array[i].detection_info[1:] # the next 4 elements of format xywh in ratio values, not absolute pixels
            
            # the bounding box data is of type xy center, xy height and width, so we gotta convert that first
            bb_ogsz = self.xywh2ogsz(np.array(bb_raw))
            
            # convert it to pixel positions
            _bb_origins.append(np.array([img_height * bb_ogsz[0], img_width * bb_ogsz[1]]).astype(int))
            _bb_sizes.append(np.array([img_height * bb_ogsz[2], img_width * bb_ogsz[3]]).astype(int))
            # Get the detection-specific parameters that will be used to crop the image
            center_low, center_high, bins = self.identify_center_depth_range(depth_img, _bb_origins[i], _bb_sizes[i])
            fovlevel_bb = self.calculate_fovlevel_bb( _bb_origins[i], _bb_sizes[i], depth_img.shape[1], depth_img.shape[0], self.fov_level)
            fovlevel_depth = self.calculate_fovlevel_depth(center_low, center_high, bins, self.fov_level)
            scale_values = np.linspace(1, self.max_scale, num=self.fov_level).astype(int)
            breakpoint()
            
            # Gather them all into arrays
            _center_highs.append(center_high)
            _center_lows.append(center_low)
            _bins.append(bins)
            _fovlevel_bbs.append(fovlevel_bb)
            _fovlevel_depths.append(fovlevel_depth)
            _scale_values.append(scale_values)
        
        img = depth_img.copy()
        for f in range(0, self.fov_level):
            img_mask = np.ones(img.shape)
            img_send = np.zeros(img.shape).astype(np.uint16)
            fov_msg = FoveatedImageCombined()
            for i in range(0, data.detection_count):
                bb_origin, bb_end = _fovlevel_bbs[i][f]
                depth_lower, depth_upper = _fovlevel_depths[i][f]
                scale_values = _scale_values[i]
                focus_height = np.array(range(int(bb_origin[0]), int(bb_end[0])))
                focus_width = np.array(range(int(bb_origin[1]), int(bb_end[1])))
                
                bbo_tpl = Tuple
                bbo_tpl.tpl = bb_origin
                bbe_tpl = Tuple
                bbe_tpl.tpl = bb_end

                fov_msg.bounding_box_origins.append(bbo_tpl)
                fov_msg.bounding_box_ends.append(bbe_tpl)

                try:
                    cropped_img = img[np.ix_(focus_height, focus_width)].astype(np.uint16)
                    
                except:
                    pass
                
                cropped_img[cropped_img < bins[int(depth_lower)]] = 0
                cropped_img[cropped_img > bins[int(depth_upper)]] = 0

                img_send[bb_origin[0]:bb_end[0], bb_origin[1]:bb_end[1]] = cropped_img

                cropped_mask = (cropped_img == 0)
                img_mask[bb_origin[0]:bb_end[0], bb_origin[1]:bb_end[1]] = cropped_mask
            
            img = (img * img_mask).astype(np.uint16)
            resized = cv2.resize(img_send, (int(img_send.shape[1]/scale_values[f]), int(img_send.shape[0]/scale_values[f])), interpolation=cv2.INTER_NEAREST)

            # Compress the image
            fov_msg.foveated_image.header.stamp = rospy.Time.now()
            fov_msg.foveated_image.format = "png"
            fov_msg.foveated_image.data = np.array(cv2.imencode('.png', resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1]).astype(np.uint16).tostring()

            fov_image_group.foveated_images_groups.append(fov_msg)
                
        print(f"Time taken: {time.time() - start}")
        # Publish it to the topic
        try:
            self.publisher.publish(fov_image_group)
            if(self.save_rgb):
                self.rgb_publisher.publish(fov_image_group_bgr)

        except:
            pass
        else:
            print('Published fov image group message with the following parameters:')
            print(f'Detected objects: {fov_image_group.detected_objects}')
            print(f'Foveation level: {fov_image_group.foveation_level}')
            print(f'Time taken: {time.time() - start}')
            self.callback_counter += 1



if __name__ == "__main__":

    rospy.init_node("foveation_node")
    # rospy.Subscriber('/camera/rgb/image_rect_color', Image, dataSaverCallback, [0])
    # rospy.Subscriber('/camera/depth_registered/image_raw', Image, dataSaverCallback, [1])
    # rospy.Subscriber('/affordance/detection/bounding_box', Tuple, dataSaverCallback, [2])
    fov = foveation_node()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
