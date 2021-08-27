#!/usr/bin/python3

from attention_package.msg import FoveatedImageMeta, FoveatedImageCombined
from yolov5_detector.msg import DetectionMsg, DetectionArray

import numpy as np
import rospy
import time
from sensor_msgs.msg import CameraInfo, Image, CompressedImage, PointCloud2, PointField
from sensor_msgs import point_cloud2

import std_msgs.msg
import struct
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
        # args from launch file
        rospy.loginfo('Pointcloud recomposing waiting for camera infos')
        self.save_img = rospy.get_param("~save_img")
        self.show_img = rospy.get_param("~show_img")
        self.save_rgb = rospy.get_param("~save_rgb")
        self.save_path = rospy.get_param("~save_path") + "recompose_imgs"
        publish_topic = rospy.get_param("~publish_topic")
        foveation_topic = rospy.get_param("~foveation_topic")
        rgb_camera = rospy.get_param("~rgb_camera")
        depth_camera = rospy.get_param("~depth_camera")
        #self.rgb_camera_info = rospy.wait_for_message(rgb_camera, CameraInfo)
        #self.depth_camera_info = rospy.wait_for_message(depth_camera, CameraInfo)
        
        # hardcode
        self.rgb_camera_info = {'K':[520.055928, 0.000000, 312.535255, 0.000000, 520.312173, 242.265554, 0.000000, 0.000000, 1.000000]}
        self.depth_camera_info = {'K':[576.092756, 0.000000, 316.286974, 0.000000, 575.853472, 239.895662, 0.000000, 0.000000, 1.000000]}
        
        # Camera parameters, where d = depth and r = rgb
        self.cx_d = self.depth_camera_info['K'][2]
        self.cy_d = self.depth_camera_info['K'][5]
        self.fx_d = self.depth_camera_info['K'][0]
        self.fy_d = self.depth_camera_info['K'][4]
        self.cx_r = self.rgb_camera_info['K'][2]
        self.cy_r = self.rgb_camera_info['K'][5]
        self.fx_r = self.rgb_camera_info['K'][0]
        self.fy_r = self.rgb_camera_info['K'][4]
        self.T = [-0.0254, -0.00013, -0.00218]
        
        # PointCloud2 args
        self.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1)
        ]
        self.header = std_msgs.msg.Header()
        self.header.frame_id = "camera_rgb_optical_frame"
        
        self.publisher = rospy.Publisher(publish_topic, PointCloud2, queue_size=5)
        self.subscriber = rospy.Subscriber(foveation_topic, FoveatedImageMeta, self.recompose_callback)
        self.bridge = cv_bridge.CvBridge()
        
        self.callback_counter = 0
        if(self.save_img):
            self.save_path = create_dir_and_return_path(self.save_path)
        init_string = "Initialised recompose_node with the following parameters:\n" + \
                    f"foveation_topic: {foveation_topic}\n" + \
                    f"publish_topic: {publish_topic}\n" + \
                    f"rgb_camera: {rgb_camera}\n" + \
                    f"depth_camera: {depth_camera}\n" + \
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
        recomposed_img_arr = []
        pc_counter = 0
        numpy_method = True
        xyz_points = np.empty((0,3), dtype=np.float32)
        rgba_points = np.empty((0), dtype=np.uint32)
        points = []
        rgb_img = self.bridge.imgmsg_to_cv2(data.rgb_image, 'bgr8')
        cv2.imwrite('rgb_img.png', rgb_img)
        
        # combined_img = np.zeros((data.height, data.width), dtype=np.uint16)
        for f in range(fov_level - 1, -1, -1): 
            # we want to go in reverse, since the last index contains the lowest resolution image.
            
            print(f"foveation level {f}")
            # basically, we want to minimize the amount of overlapping.
            # at higher foveation levels, we can just take the whole (scaled down) image.
            # Steps: 
            
            curr_img = np.frombuffer(data.foveated_images_groups[f].foveated_image.data, np.uint16)
            curr_img = cv2.imdecode(curr_img, cv2.IMREAD_ANYDEPTH)
            cv2.imwrite(f'curr_img{f}.png', curr_img.astype(np.uint16))
            P3D = np.zeros((curr_img.shape[0], curr_img.shape[1], 6), dtype=np.float32)
            
            y, x = (curr_img > 10).nonzero()
            #zipped = zip(y,x)
            #print(len(y), len(x))
            
            if(f > 0):
                # scale the RGB image to depth image size
                scaled_rgb_img = cv2.resize(rgb_img, (curr_img.shape[1], curr_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                scale_factor = curr_img.shape[1] / rgb_img.shape[1] # rgb image will always be the original size
            else:
                scaled_rgb_img = rgb_img
                scale_factor = 1
            # Scale the camera intrinsic parameters to fit the new size
            print(f'SCALE FACTOR: {scale_factor}')
            scaled_cx_d = self.cx_d * scale_factor
            scaled_cy_d = self.cy_d * scale_factor
            scaled_fx_d = self.fx_d * scale_factor
            scaled_fy_d = self.fy_d * scale_factor
            
            scaled_cx_r = self.cx_r * scale_factor
            scaled_cy_r = self.cy_r * scale_factor
            scaled_fx_r = self.fx_r * scale_factor
            scaled_fy_r = self.fy_r * scale_factor
            
            # Numpy vectorization method
            
            if(numpy_method):
                depth = curr_img[y, x]
                T = self.T * 1000 # we work in mm scale according to kinect, then divide it later to minimize out-of-bound image pixels.
                
                P3D_x = ((x - scaled_cx_d) * depth / scaled_fx_d) + T[0]
                P3D_y = ((y - scaled_cy_d) * depth / scaled_fy_d) + T[1]
                P3D_z = (depth) + T[2]

                P2D_x = (P3D_x * scaled_fx_r / P3D_z) + scaled_cx_r
                P2D_y = (P3D_y * scaled_fy_r / P3D_z) + scaled_cy_r
                P2D_rgba = (np.c_[scaled_rgb_img[P2D_y.astype(int), P2D_x.astype(int)], (np.ones(len(P2D_y)) * 255)]).astype(np.uint8)
                P3D_xyz = np.column_stack((P3D_x, P3D_y, P3D_z)) / 1000
                packed_rgba = np.squeeze(P2D_rgba.view(np.uint32))
                #P3D_xyzrgb = np.c_[P3D_x, P3D_y, P3D_z, packed_rgba]
                rgba_points = np.hstack((rgba_points, packed_rgba))
                xyz_points = np.vstack((xyz_points, P3D_xyz.astype(np.float32)))
                pc_counter += len(P3D_x)
            
            
            # zip method
            else:
                zipped = zip(y,x)
                depth_img = (curr_img.astype(np.float32))/1000 # scale it properly to meters
                for yx in zipped:
                    P3D = [0,0,0,0,0,0]
                    P2D = [0,0]
                    i = yx[0]
                    j = yx[1]
                    if(depth_img[i, j] <= 0 or np.isnan(depth_img[i, j])): # invalid pixel
                        continue
                    P3D[0] = ((j - scaled_cx_d) * depth_img[i, j] / scaled_fx_d) + self.T[0]
                    P3D[1] = ((i - scaled_cy_d) * depth_img[i, j] / scaled_fy_d) + self.T[1]
                    P3D[2] = (depth_img[i, j]) + self.T[2]

                    P2D[0] = (P3D[0] * scaled_fx_r / P3D[2]) + scaled_cx_r
                    P2D[1] = (P3D[1] * scaled_fy_r / P3D[2]) + scaled_cy_r
                    try:
                        P3D[3:] = scaled_rgb_img[int(P2D[1]), int(P2D[0])]
                        x = P3D[0]
                        y = P3D[1]
                        z = P3D[2]
                        r = int(P3D[3])
                        g = int(P3D[4])
                        b = int(P3D[5])
                        a = 255
                        rgb = struct.unpack('I', struct.pack('BBBB', b,g,r,a))[0]
                        
                        points.append([x,y,z,rgb])
                        pc_counter += 1
                    except:
                        pass
                
            '''
            if(f == 0):
                # lowest fov level, so we want as much detail here as possible
                
                for obj in range(0, detection_count):
                    current_origin = data.foveated_images_groups[f].bounding_box_origins[obj].tpl
                    current_end = data.foveated_images_groups[f].bounding_box_ends[obj].tpl
                    
                    # since the lowest level has the highest resolution and minimum overlap, we want to
                    # be selective in our processing.
                    for i in range(current_origin[0], current_end[0]):
                        for j in range(current_origin[1], current_end[1]):
                            if(depth_img[i, j] <= 0 or np.isnan(depth_img[i, j])): # invalid pixel
                                continue
                            P3D = [0,0,0,0,0,0]
                            P2D = [0,0]
                            P3D[0] = ((j - self.cx_d) * depth_img[i, j] / self.fx_d) + self.T[0] # x --> -y
                            P3D[1] = ((i - self.cy_d) * depth_img[i, j] / self.fy_d) + self.T[1] # y --> z
                            P3D[2] = (depth_img[i, j]) + self.T[2]                               # z --> x

                            P2D[0] = (P3D[0] * self.fx_r / P3D[2]) + self.cx_r
                            P2D[1] = (P3D[1] * self.fy_r / P3D[2]) + self.cy_r
                            try:
                                P3D[3:] = rgb_img[int(P2D[1]), int(P2D[0])]
                                x = P3D[0] # conversion to ros convention
                                y = P3D[1]
                                z = P3D[2]
                                r = int(P3D[3])
                                g = int(P3D[4])
                                b = int(P3D[5])
                                a = 255
                                rgb = struct.unpack('I', struct.pack('BBBB', b,g,r,a))[0]
                                points.append([x,y,z,rgb])
                                pc_counter += 1
                            except e:
                                print(e)
                                pass
            else:
                # scale the RGB image to depth image size
                scaled_rgb_img = cv2.resize(rgb_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                scale_factor = depth_img.shape[1] / rgb_img.shape[1] # rgb image will always be the original size
                print(f'SCALE FACTOR: {scale_factor}')
                # Scale the camera intrinsic parameters to fit the new size
                scaled_cx_d = self.cx_d * scale_factor
                scaled_cy_d = self.cy_d * scale_factor
                scaled_fx_d = self.fx_d * scale_factor
                scaled_fy_d = self.fy_d * scale_factor
                
                scaled_cx_r = self.cx_r * scale_factor
                scaled_cy_r = self.cy_r * scale_factor
                scaled_fx_r = self.fx_r * scale_factor
                scaled_fy_r = self.fy_r * scale_factor
                
                for i in range(0, scaled_rgb_img.shape[0]):
                    for j in range(0, scaled_rgb_img.shape[1]):
                        if(depth_img[i, j] <= 0 or np.isnan(depth_img[i, j])): # invalid pixel
                            continue
                        
                        P3D = [0,0,0,0,0,0]
                        P2D = [0,0]
                        P3D[0] = ((j - scaled_cx_d) * depth_img[i, j] / scaled_fx_d) + self.T[0]
                        P3D[1] = ((i - scaled_cy_d) * depth_img[i, j] / scaled_fy_d) + self.T[1]
                        P3D[2] = (depth_img[i, j]) + self.T[2]

                        P2D[0] = (P3D[0] * scaled_fx_r / P3D[2]) + scaled_cx_r
                        P2D[1] = (P3D[1] * scaled_fy_r / P3D[2]) + scaled_cy_r
                        try:
                            P3D[3:] = scaled_rgb_img[int(P2D[1]), int(P2D[0])]
                            x = P3D[0]
                            y = P3D[1]
                            z = P3D[2]
                            r = int(P3D[3])
                            g = int(P3D[4])
                            b = int(P3D[5])
                            a = 255
                            rgb = struct.unpack('I', struct.pack('BBBB', b,g,r,a))[0]
                            
                            points.append([x,y,z,rgb])
                            pc_counter += 1
                        except e:
                            print(e)
                            pass
                    #endfor j
                #endfor i
            #endif fovlevel == 0
        #endfor f
        '''        
        
        # Here, create PC2 message using create_cloud, then publish.
        
        # We want to know the bandwidth of this pointcloud vs. the bandwidth of the pure alt_recompose
        # to decide whether we perform this step on the nano or on the unity, disregarding processing power.
        print(f'Processed points: {pc_counter}')
        
        if(numpy_method):
            # This step takes up 90% of the processing time, just because of the datatype incompatibility. 
            # Need a way to improve it here, then I think we will get a pretty usable algorithm out of it.
            points = [(val[0][0], val[0][1], val[0][2], val[1]) for val in zip(xyz_points, rgba_points)]
        pc2 = point_cloud2.create_cloud(self.header, self.fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.publisher.publish(pc2)
        print(f"Done recomposing! that took {time.time() - start} sec")
        


if __name__ == '__main__':
    rospy.init_node("recompose_node")
    recompose = recompose_node()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit()

