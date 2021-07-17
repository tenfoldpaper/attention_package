#!/usr/bin/env python

from attention_package.msg import FoveatedImageGroups

import numpy as np
import rospy

import roslaunch

def callback(data):
    print('got a callback!')


if __name__ == '__main__':
    rospy.init_node("simple_subber") # default node name when not overwritten
    sub = rospy.Subscriber('/attention/foveated', FoveatedImageGroups, callback)
    rospy.loginfo("FUCK")
    print(rospy.get_param('~fov_param1'))

    
    rospy.loginfo("started")


    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit()

