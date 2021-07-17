#!/usr/bin/env python3

import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
import sys, select, os
from io import StringIO

foveated_publisher = None

def crop_foveation(data, bb_origin, bb_size, width=640, height=480):
    
    # Assuming it's (height, width),
    
    # variable initialisation
    empty_var = data[0][0].copy()
    empty_var[0] = np.nan
    empty_var[1] = np.nan
    empty_var[2] = np.nan
    empty_var[3] = 1.6410822e-38
    
    bb_range = (bb_origin[0] + bb_size[0] if bb_origin[0] + bb_size[0] < height else height, \
        bb_origin[1] + bb_size[1] if bb_origin[1] + bb_size[1] < width else width)
    focus_height = np.array(range(bb_origin[0], bb_range[0]))
    focus_width = np.array(range(bb_origin[1], bb_range[1]))
    focus_data = data[np.ix_(focus_height, focus_width)]
    data[np.ix_(focus_height, focus_width)] = np.tile(empty_var, focus_data.shape) # zero out for compression later; might have to use nan instead?
    peripheral_radius = 0.3
    # Do a second foveation around the edge of the bounding box
    #peripheral area
    pa_origin = ( int(bb_origin[0] - (peripheral_radius*bb_size[0]) if bb_origin[0] - (peripheral_radius*bb_size[0]) > 0 else 0), \
        int(bb_origin[1] - (peripheral_radius*bb_size[1]) if bb_origin[1] - (peripheral_radius*bb_size[1]) > 0 else 0))
    pa_range = ( int(bb_range[0] + (peripheral_radius*bb_size[0]) if bb_range[0] + (peripheral_radius*bb_size[0]) < height else height), \
        int(bb_range[1] + (peripheral_radius*bb_size[1]) if bb_range[1] + (peripheral_radius*bb_size[1]) < width else width))

    pa_height = np.array(range(pa_origin[0], pa_range[0]))
    pa_width = np.array(range(pa_origin[1], pa_range[1]))

    pa_data = data[np.ix_(pa_height,pa_width)]
    data[np.ix_(pa_height,pa_width)] = np.tile(empty_var, pa_data.shape)
    pa_mask = np.ones(pa_data.shape, dtype=bool)
    pa_mask[::3, ::3] = 0
    pa_data = pa_data[pa_mask]


    mask = np.ones(data.shape, dtype=bool)
    mask[np.ix_(pa_height,pa_width)] = 0 # we don't want points that are in the higher-density foveation messages
    mask[::2, ::2] = 0 
    mask[1::2, 1::2] = 0

    out_data = data[mask]
    return focus_data, pa_data, out_data

def dataSaverCallback(data, args):
    if(type(data) == CameraInfo):
        with open(args[0], 'w+') as f:
            
            f.write(str(data))
    
    else:
        numpy_data = ros_numpy.numpify(data)
        with open(args[0], 'wb+') as f:
            np.save(f, numpy_data)
    


def callback(data):
    #rospy.loginfo(len(data.data))
    #rospy.loginfo(f"Shape: {data.width} x {data.height}")
    #rospy.loginfo(f"other: {data.point_step}, {data.row_step}")
    numpy_data = ros_numpy.numpify(data)

    #breakpoint()
    # print(np.shape(numpy_data))
    # with open('test.npy', 'wb+') as f:
    #     np.save(f, numpy_data)
    in_data = numpy_data.copy()
    focus_data, pa_data, sdata = crop_foveation(in_data, (240, 320), (100, 100), width=data.width, height=data.height)

    msg1 = ros_numpy.msgify(PointCloud2, sdata)
    msg2 = ros_numpy.msgify(PointCloud2, focus_data)
    msg3 = ros_numpy.msgify(PointCloud2, pa_data)
    
    msg1.header.frame_id = 'camera_depth_link'
    msg2.header.frame_id = 'camera_depth_link'
    msg3.header.frame_id = 'camera_depth_link'

    outer_publisher.publish(msg1)
    focus_publisher.publish(msg2)
    periph_publisher.publish(msg3)
#    breakpoint()
    # Each point is 32 bytes.


if __name__ == "__main__":
    rospy.init_node("kinect_subscriber")
    rospy.Subscriber('/camera/depth_registered/camera_info', CameraInfo, dataSaverCallback, ["depthCameraInfo.txt"])
    rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, dataSaverCallback, ["colorCameraInfo.txt"])

    rospy.Subscriber('/camera/depth_registered/image_raw', Image, dataSaverCallback, ["depthRaw.npy"])
    rospy.Subscriber('/camera/rgb/image_rect_color/compressed', CompressedImage, dataSaverCallback, ["colorComp.npy"])
    rospy.Subscriber('/camera/rgb/image_rect_color', Image, dataSaverCallback, ["colorRaw.npy"])
    rospy.Subscriber('/camera/depth_registered/points', PointCloud2, dataSaverCallback, ["pointCloud2.npy"])

    #rospy.Subscriber('/camera/depth_registered/points', PointCloud2, callback)
    focus_publisher = rospy.Publisher('/camera/depth_registered/fov_focus', PointCloud2, queue_size = 10)
    periph_publisher = rospy.Publisher('/camera/depth_registered/fov_periph', PointCloud2, queue_size = 10)
    outer_publisher = rospy.Publisher('/camera/depth_registered/fov_outer', PointCloud2, queue_size = 10)
    
    breakpoint()
    
    rospy.spin()
