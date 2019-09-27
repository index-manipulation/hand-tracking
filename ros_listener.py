# bala paen 1 , chap rast 2 , jelo aghab 3
# pylint: disable=wrong-import-position

import sys
sys.path.append("lib")

from cv_bridge import CvBridge,CvBridgeError
from std_msgs.msg import String
import rospy
from sensor_msgs.msg import Image

import time
import os
import cv2


import numpy as np
import pickle as pkl

import PyCeresIK as IK
from common import image
from common.opencv_grabbers import OpenCVGrabber
from common.calibrate import OpenCVCalib2CameraMeta, LoadOpenCVCalib
from common import mva19
from common import factory
from common import pipeline
import PyMBVCore  as Core
import PyJointTools as jt

import yaml
from rosbag.bag import Bag

from utils import detector_utils
from tqdm import tqdm 


bridge = CvBridge()
fake_clb = OpenCVCalib2CameraMeta(LoadOpenCVCalib("res/calib_webcam_mshd_vga.json"))

info_dict = yaml.load(Bag(sys.argv[2], 'r')._get_yaml_info())
offset = float(sys.argv[3])

recording_delay = info_dict['start'] + offset

limbSeq = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17], # palm
           [1, 2], [2, 3], [3,4], # thump
           [5, 6], [6, 7], [7, 8], # index
           [9, 10], [10, 11], [11, 12], # middle
           [13, 14], [14, 15], [15, 16], # ring
           [17, 18], [18, 19], [19, 20], # pinky
        ]

final_result = []

def _center(a, b):
    return [(v[0]+v[1])/2000 for v in list(zip(a,b)) ]


def mono_hand_loop(acq, outSize, config, track=False, paused=False):
    
    print("Initialize WACV18 3D Pose estimator (IK)...")
    pose_estimator = factory.HandPoseEstimator(config)


    print("Initialize MVA19 CVRL Hand pose net...")
    estimator = mva19.Estimator(config["model_file"], config["input_layer"], config["output_layer"])


    detection_graph, sess = detector_utils.load_inference_graph()
    

    started = True
    delay = {True: 0, False: 1}
    p2d = bbox = None
    smoothing = config.get("smoothing", 0)
    boxsize = config["boxsize"]
    stride = config["stride"]
    peaks_thre = config["peaks_thre"]
    print("Entering main Loop.")

    for topic, img_msg, ros_time in tqdm(Bag(sys.argv[2]).read_messages()):
        if topic != "camera/rgb/image_raw":
            continue
        try:
            bgr = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            bgr = cv2.resize(bgr, (640,480), interpolation = cv2.INTER_AREA) 
        except Exception as e:
            print("Failed to grab", e)
            break

        clb = fake_clb

        # compute kp using model initial pose
        points2d = pose_estimator.ba.decodeAndProject(pose_estimator.model.init_pose, clb)
        oldKp = np.array(points2d).reshape(-1, 2)


        score = -1
        result_pose = None
        # STEP2 detect 2D joints for the detected hand.
        if started:
            if bbox is None:
                bbox = detector_utils.hand_bbox(bgr,detection_graph,sess)
                if bbox is None:
                    if sys.argv[4]=='1':
                        cv2.imshow("3D Hand Model reprojection",bgr)
                        cv2.waitKey(1)
                    to_AC_format(np.zeros((21,3)),ros_time, 1)
                    continue
            else:
                dbox = detector_utils.hand_bbox(bgr,detection_graph,sess)
                if dbox is not None:
                    x,y,w,h = bbox
                    x1,y1,w1,h1 = dbox
                    if (x1>x+w or x1+w1<x ) or y1+h1<y or y1>y+h:
                        bbox = dbox
                        print("updated")
                    else:
                        x,y,w,h = dbox
                        b = max((w,h,224))
                        b = int(b + b*0.3)
                        cx = x + w/2
                        cy = y + h/2
                        x = cx-b/2
                        y = cy-b/2

                        x = max(0,int(x))
                        y = max(0,int(y))

                        x = min(x, bgr.shape[1]-b)
                        y = min(y, bgr.shape[0]-b)
                        
                        bbox = [x,y,b,b]

            x,y,w,h = bbox
            crop = bgr[y:y+h,x:x+w]
            img, pad = mva19.preprocess(crop, boxsize, stride)
            t = time.time()
            hm = estimator.predict(img)
            est_ms = (time.time() - t)

            # use joint tools to recover keypoints
            scale = float(boxsize) / float(crop.shape[0])
            scale = stride/scale
            ocparts = np.zeros_like(hm[...,0])
            peaks = jt.FindPeaks(hm[...,:-1], ocparts, peaks_thre, scale, scale)

            # convert peaks to hand keypoints
            hand = mva19.peaks_to_hand(peaks, x, y)

            if hand is not None:
                keypoints = hand
            
                mask = keypoints[:, 2] < peaks_thre
                keypoints[mask] = [0, 0, 1.0]

                if track:
                    keypoints[mask, :2] = oldKp[mask]

                keypoints[:, 2] = keypoints[:, 2]**3
                
                rgbKp = IK.Observations(IK.ObservationType.COLOR, clb, keypoints)
                obsVec = IK.ObservationsVector([rgbKp, ])
                score, res = pose_estimator.estimate(obsVec)
                
                if track:
                    result_pose = list(smoothing * np.array(pose_estimator.model.init_pose) + (1.0 - smoothing) * np.array(res))
                else:
                    result_pose = list(res)
                # score is the residual, the lower the better, 0 is best
                # -1 is failed optimization.
                if track:
                    if -1 < score: #< 150:
                        pose_estimator.model.init_pose = Core.ParamVector(result_pose)
                    else:
                        print("\n===>Reseting init position for IK<===\n")
                        pose_estimator.model.reset_pose()
                        bbox = None

                if score > -1:  # compute result points
                    p2d = np.array(pose_estimator.ba.decodeAndProject(Core.ParamVector(result_pose), clb)).reshape(-1, 2)
                    # scale = w/config.boxsize
                    bbox = mva19.update_bbox(p2d,bgr.shape[1::-1])
            
            p3d = np.array(pose_estimator.ba.decode(Core.ParamVector(result_pose), clb)).reshape(-1,3)
            to_AC_format(p3d,ros_time,1)


        viz = np.copy(bgr)
        if sys.argv[4] == '1' and started and result_pose is not None:
            viz = mva19.visualize_3dhand_skeleton(viz, p2d)
            pipeline.draw_rect(viz, "Hand", bbox, box_color=(0, 255, 0), text_color=(200, 200, 0))
            cv2.putText(viz, 'Hand pose estimation', (20, 20), 0, 0.7, (50, 20, 20), 1, cv2.LINE_AA)
            cv2.imshow("3D Hand Model reprojection", viz)

        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused
        if key & 255 == ord('q'):
            break
        if key & 255 == ord('r'):
            print("\n===>Reseting init position for IK<===\n")
            pose_estimator.model.reset_pose()
            bbox = None
            print("RESETING BBOX",bbox)


        


def save():
    global final_result

    keys = "time  wristPosition0X	wristPosition0Y	wristPosition0Z boneCenter000X	boneCenter000Y	boneCenter000Z	boneDirection000X	boneDirection000Y	boneDirection000Z	boneCenter001X	boneCenter001Y	boneCenter001Z	boneDirection001X	boneDirection001Y	boneDirection001Z	boneCenter002X	boneCenter002Y	boneCenter002Z	boneDirection002X	boneDirection002Y	boneDirection002Z	boneCenter003X	boneCenter003Y	boneCenter003Z	boneDirection003X	boneDirection003Y	boneDirection003Z	boneCenter010X	boneCenter010Y	boneCenter010Z	boneDirection010X	boneDirection010Y	boneDirection010Z	boneCenter011X	boneCenter011Y	boneCenter011Z	boneDirection011X	boneDirection011Y	boneDirection011Z	boneCenter012X	boneCenter012Y	boneCenter012Z	boneDirection012X	boneDirection012Y	boneDirection012Z	boneCenter013X	boneCenter013Y	boneCenter013Z	boneDirection013X	boneDirection013Y	boneDirection013Z	boneCenter020X	boneCenter020Y	boneCenter020Z	boneDirection020X	boneDirection020Y	boneDirection020Z	boneCenter021X	boneCenter021Y	boneCenter021Z	boneDirection021X	boneDirection021Y	boneDirection021Z	boneCenter022X	boneCenter022Y	boneCenter022Z	boneDirection022X	boneDirection022Y	boneDirection022Z	boneCenter023X	boneCenter023Y	boneCenter023Z	boneDirection023X	boneDirection023Y	boneDirection023Z	boneCenter030X	boneCenter030Y	boneCenter030Z	boneDirection030X	boneDirection030Y	boneDirection030Z	boneCenter031X	boneCenter031Y	boneCenter031Z	boneDirection031X	boneDirection031Y	boneDirection031Z	boneCenter032X	boneCenter032Y	boneCenter032Z	boneDirection032X	boneDirection032Y	boneDirection032Z	boneCenter033X	boneCenter033Y	boneCenter033Z	boneDirection033X	boneDirection033Y	boneDirection033Z	boneCenter040X	boneCenter040Y	boneCenter040Z	boneDirection040X	boneDirection040Y	boneDirection040Z	boneCenter041X	boneCenter041Y	boneCenter041Z	boneDirection041X	boneDirection041Y	boneDirection041Z	boneCenter042X	boneCenter042Y	boneCenter042Z	boneDirection042X	boneDirection042Y	boneDirection042Z	boneCenter043X	boneCenter043Y	boneCenter043Z	boneDirection043X	boneDirection043Y	boneDirection043Z check"
    keys = keys.split()
    if not os.path.exists(sys.argv[5]):
        os.makedirs(sys.argv[5])
    save_path = os.path.join(sys.argv[5], os.path.basename(sys.argv[2]).split('.')[0]+'-result.csv')
    final_result = sorted(final_result, key=lambda item: item['time'])
    with open(save_path,'w+') as resfile:
        resfile.write('frame,'); resfile.write(",".join(keys)); resfile.write('\n')
        
        for i, item in enumerate(final_result):
            resfile.write(str(i)+',')
            for key in keys:
                a = item[key] if key in item.keys() else 0
                resfile.write(str(a)+',')
            resfile.write('\n')





def to_AC_format(p3d,t,check):
    t =  float(str(t.secs) + '.'+str(t.nsecs)) - recording_delay
    bonecenter = _center(p3d[0],p3d[0])
    root = np.dot(np.transpose(transform), np.append(bonecenter,1))
    d = {
        'time':t,
        'wristPosition0X': -root[1],
        'wristPosition0Y': root[2],
        'wristPosition0Z': -root[0],
        'check': check
    }
    for i, pair in enumerate(limbSeq):
        a = p3d[pair[0]]
        b = p3d[pair[1]]
        bone_center = _center(a,b)
        tag = "boneCenter0"
        if i in range(5):
            tag+=str(i)+'0'
        else:
            finger = int((i-5)/3)
            bone = (i-5) - finger*3 + 1
            tag= tag + str(finger)+str(bone)
        bone_center = np.dot(np.transpose(transform), np.append(bone_center ,1))

        d[tag+'X'] = -bone_center[1]
        d[tag+'Y'] = bone_center[2]
        d[tag+'Z'] = -bone_center[0]

    final_result.append(d)








if __name__ == "__main__":
    global config,outSize,track,paused, with_renderer,pub,transform
    
    transform = pkl.load(open(sys.argv[1],'rb'), encoding='latin1')
    print(transform)
    track=True
    paused=False
    outSize = (640,480)
    config = {
        "model": "models/hand_skinned.xml", "model_left": False,
        "model_init_pose": [-109.80840809323652, 95.70022984677065, 584.613931114289, 292.3322807284121, -1547.742897973965, -61.60146881490577, 435.33025195547793, 1.5707458637241434, 0.21444030289465843, 0.11033385117688158, 0.021952050059337137, 0.5716581133215294, 0.02969734913698679, 0.03414155945643072, 0.0, 1.1504613679382742, -0.5235922979328, 0.15626331136368257, 0.03656410417088128, 8.59579088582312e-07, 0.35789633949684985, 0.00012514308785717494, 0.005923001258945023, 0.24864102398139007, 0.2518954858979162, 0.0, 3.814694400000002e-13],
        "model_map": IK.ModelAwareBundleAdjuster.HAND_SKINNED_TO_OP_RIGHT_HAND,

        "ba_iter": 100,
        "padding": 0.3,
        "minDim": 170,

        "smoothing": 0.2,

        "model_file": "models/mobnet4f_cmu_adadelta_t1_model.pb",
        "input_layer": "input_1",
        "output_layer": "k2tfout_0",
        "stride": 4,
        "boxsize": 224,
        "peaks_thre": 0.1,
        
        # default bbox for the hand location 
        # "default_bbox": [322, 368, 110, 109],
    }
    
    mono_hand_loop(None, (640,480), config,  track=track)
    save()
    cv2.destroyAllWindows()
