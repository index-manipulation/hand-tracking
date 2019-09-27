#!/usr/bin/env python
# license removed for brevity
import sys
try:
    sys.path.remove('/usr/local/lib/python3.5/dist-packages')
except:
    pass
import rospy
import tf
import geometry_msgs
import numpy as np
import rosbag
import subprocess
import threading
import os
import pickle as pkl
import time
last_one = None

thread = None
def get_average_transform():
    global last_one
    rospy.init_node('transform', anonymous=True)
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()
    transformerROS = tf.TransformerROS()
    alphaAccumulator = 0.2
    rate = rospy.Rate(10) # 10hz
    p1 = None
    while p1 == None:
        try:
            p1 = geometry_msgs.msg.PoseStamped()
            now = rospy.Time(0)
            listener.waitForTransform("camera_rgb_optical_frame", "ar_marker_8", now, rospy.Duration(2.0))
            # (trans, rot) = listener.lookupTransform("camera_rgb_optical_frame", "ar_marker_8", now)
            (trans, rot) = listener.lookupTransform("ar_marker_8", "camera_rgb_optical_frame", now)
            p1.pose.orientation = rot
            p1.pose.position = trans
        except:
            p1 = None

    while not rospy.is_shutdown():
        now = rospy.Time(0)
        
        listener.waitForTransform("camera_rgb_optical_frame", "ar_marker_8", now, rospy.Duration(2.0))
        (trans, rot) = listener.lookupTransform("ar_marker_8", "camera_rgb_optical_frame", now)

        # Exponential Moving Average to smoothen over values
        p1.pose.orientation = (alphaAccumulator * np.array(rot)) + (1.0 - alphaAccumulator) * np.array(p1.pose.orientation)
        p1.pose.position = (alphaAccumulator * np.array(trans)) + (1.0 - alphaAccumulator) * np.array(p1.pose.position)
        T = transformerROS.fromTranslationRotation(p1.pose.position,p1.pose.orientation)
        # origin = np.asmatrix([[0],[0],[0],[1]],dtype=np.float16)
        last_one = np.transpose(T)
        if not thread.is_alive():
            to_save = sys.argv[2] if len(sys.argv)>2 else '%s-tf.pkl' % (os.path.basename(sys.argv[1].split('.')[0]),)
            print('thread finished')
            pkl.dump(last_one, open(to_save, 'wb'))
            os._exit(0)
        
        
def test():
    os.system("rosbag play --duration=7 --clock '%s' " % (sys.argv[1],))

def ar_tracker():
    os.system('rosrun ar_track_alvar findMarkerBundlesNoKinect 6.7 0.1 0.1 /camera/rgb/image_raw /camera/rgb/camera_info camera_rgb_optical_frame ../table_8_9_10_2.xml')

if __name__ == '__main__':
    try:
        thread2 = threading.Thread(target=ar_tracker)
        thread2.start()
        thread = threading.Thread(target=test)
        thread.start()
        get_average_transform()
        
    except rospy.ROSInterruptException:
        pass        
