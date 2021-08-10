#!/usr/bin/env python

from attention_package.msg import FoveatedImageGroups, FoveatedImage, FoveatedImages, Tuple
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
        self.publisher = rospy.Publisher(publish_topic, FoveatedImageGroups, queue_size=5)
        self.rgb_publisher = rospy.Publisher(rgb_publish_topic, FoveatedImageGroups, queue_size=5)
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
        breakpoint()
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
        
        breakpoint()
        fovlevel_bb = list(zip(lower_bounds, upper_bounds))

        return fovlevel_bb

    def calculate_fovlevel_depth(self, center_low, center_high, bins, fovlevel):

        fovlevel_depth = []
        
        lower_bounds = np.linspace(center_low, 0, num=fovlevel).astype(int)
        upper_bounds = np.linspace(center_high, len(bins) - 1, num=fovlevel).astype(int)
        

        fovlevel_depth = list(zip(lower_bounds, upper_bounds))
        breakpoint()
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

        fov_image_group = FoveatedImageGroups()
        fov_image_group.height = img_height
        fov_image_group.width = img_width
        fov_image_group.detected_objects = data.detection_count
        fov_image_group.foveation_level = self.fov_level
        fov_image_group.rgb_image = data.rgb_image
        
        if(self.save_rgb):
            fov_image_group_bgr = FoveatedImageGroups()
            fov_image_group_bgr.height = img_height
            fov_image_group_bgr.width = img_width
            fov_image_group_bgr.detected_objects = data.detection_count
            fov_image_group_bgr.foveation_level = self.fov_level
            fov_image_group_bgr.rgb_image = data.rgb_image
                

        # create a simple case for when detection_count is 0 --> just downsample the image by the scale factor
        for i in range(0, data.detection_count):
            
            fov_images = FoveatedImages()
            
            if(self.save_rgb):
                fov_images_bgr = FoveatedImages()
            # get the detection bounding boxes from the topic
            det_class = data.detection_array[i].detection_info[0] # first element 
            bb_raw = data.detection_array[i].detection_info[1:] # the next 4 elements of format xywh in ratio values, not absolute pixels
            # the bounding box data is of type xy center, xy height and width, so we gotta convert that first
            bb_ogsz = self.xywh2ogsz(np.array(bb_raw))
            # convert it to pixel positions
            bb_origin = np.array([img_height * bb_ogsz[0], img_width * bb_ogsz[1]]).astype(int)
            bb_size = np.array([img_height * bb_ogsz[2], img_width * bb_ogsz[3]]).astype(int)
            #bb_origin = np.array([img_height * bb_raw[0], img_width * bb_raw[1]]).astype(int)
            #bb_size = np.array([img_height * bb_raw[2], img_width * bb_raw[3]]).astype(int)

            # Get the foveation-specific parameters
            center_low, center_high, bins = self.identify_center_depth_range(depth_img, bb_origin, bb_size)
            fovlevel_bb = self.calculate_fovlevel_bb( bb_origin, bb_size, depth_img.shape[1], depth_img.shape[0], self.fov_level)
            fovlevel_depth = self.calculate_fovlevel_depth(center_low, center_high, bins, self.fov_level)
            scale_values = np.linspace(1, self.max_scale, num=self.fov_level).astype(int)
            
            # Get a copy of the image to be processed
            img = depth_img.copy()
            # Perform foveation 
            for f in range(0, self.fov_level):
                fov_image = FoveatedImage()
                bb_origin, bb_end = fovlevel_bb[f]
                depth_lower, depth_upper = fovlevel_depth[f]

                # crop it according to the bounding box calculated from fovlevel_bb
                focus_height = np.array(range(int(bb_origin[0]), int(bb_end[0])))
                focus_width = np.array(range(int(bb_origin[1]), int(bb_end[1])))
                

                try:
                    cropped_img = img[np.ix_(focus_height, focus_width)].astype(np.uint16)
                    cropped_bgr = rgb_img[np.ix_(focus_height, focus_width)]
                except:
                    #breakpoint()
                    pass

                # get rid of out-of-range depth values
                cropped_img[cropped_img < bins[int(depth_lower)]] = 0
                cropped_img[cropped_img > bins[int(depth_upper)]] = 0

                
                
                # get the cropped image and scale it according to the current foveation level
                #gm = PIL.Image.fromarray(cropped_img, mode="I;16")
                #resized = gm.resize((int(cropped_img.shape[1]/scale_values[f]), int(cropped_img.shape[0]/scale_values[f])), resample=PIL.Image.NEAREST)
                resized = cv2.resize(cropped_img, (int(cropped_img.shape[1]/scale_values[f]), int(cropped_img.shape[0]/scale_values[f])), interpolation=cv2.INTER_NEAREST)
                resized_ros = self.bridge.cv2_to_imgmsg(np.array(resized), 'mono16')
                fov_image.foveated_image = resized_ros
                fov_image.bounding_box_origins.tpl = bb_origin
                fov_image.bounding_box_sizes.tpl = bb_end

                if(self.save_rgb):
                    cropped_bgr[cropped_img < bins[int(depth_lower)]] = 0
                    cropped_bgr[cropped_img > bins[int(depth_upper)]] = 0
                    resized_bgr = cv2.resize(cropped_bgr, (int(cropped_bgr.shape[1]/scale_values[f]), int(cropped_bgr.shape[0]/scale_values[f])), interpolation=cv2.INTER_NEAREST)
                    resized_bgr_ros = self.bridge.cv2_to_imgmsg(np.array(resized_bgr), 'bgr8')
                    fov_image_bgr = FoveatedImage()
                    fov_image_bgr.foveated_image = resized_bgr_ros                    
                    fov_image_bgr.bounding_box_origins.tpl = bb_origin
                    fov_image_bgr.bounding_box_sizes.tpl = bb_end
                    fov_images_bgr.foveated_images.append(fov_image_bgr)
                    

                if(self.save_img):
                    coloured_depth = ((resized / np.max(resized))*256).astype(np.uint8)
                    coloured_depth = cv2.applyColorMap(coloured_depth, cv2.COLORMAP_HSV)
                    PIL.Image.fromarray(coloured_depth).save(f"{self.save_path}resized{self.callback_counter}_{i}_{f}.png")
                    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                    rgb_pil = PIL.Image.fromarray(cropped_rgb, "RGB")
                    rgb_pil.save(f"{self.save_path}rgb_ver{self.callback_counter}_{i}_{f}.png", optimize=True)
                
                if(self.show_img):

                    coloured_depth = ((resized / np.max(resized))*256).astype(np.uint8)
                    coloured_depth = cv2.applyColorMap(coloured_depth, cv2.COLORMAP_HSV)
                    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                    cv2.imshow("HSV-coloured depth", coloured_depth)
                    cv2.imshow("RGB image", cropped_rgb)
                    cv2.waitKey(1)
                    

                # obtain a mask based on values that have been retained in the current level
                cropped_mask = (cropped_img == 0)

                img_mask = np.ones(img.shape)
                img_mask[bb_origin[0]:bb_end[0], bb_origin[1]:bb_end[1]] = cropped_mask

                # Apply mask to the original image
                img = (img * img_mask).astype(np.uint16)
                fov_images.foveated_images.append(fov_image)

                # pack the data into FoveatedImage, then append to FoveatedImages to be published
            
            # pack the data into FoveatedImageGroups
            fov_image_group.foveated_images_groups.append(fov_images)
            if(self.save_rgb):
                fov_image_group_bgr.foveated_images_groups.append(fov_images_bgr)

        print(f'Time taken: {time.time() - start}')
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
            #print(f'Time taken: {time.time() - start}')
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
