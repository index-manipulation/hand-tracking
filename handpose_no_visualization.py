# bala paen 1 , chap rast 2 , jelo aghab 3
# pylint: disable=wrong-import-position
'''
Adapted from the MonoHand3D codebase for the MonocularRGB_3D_Handpose project (github release)

This script uses the 2D joint estimator of Gouidis et al.

@author: Paschalis Panteleris (padeler@ics.forth.gr)
'''

import sys
sys.path.append("lib")
import time
import cv2
import numpy as np

import PyCeresIK as IK
import PyMBVCore  as Core
import PyJointTools as jt

from common import image
from common.opencv_grabbers import OpenCVGrabber
from common.calibrate import OpenCVCalib2CameraMeta, LoadOpenCVCalib
from common import mva19
from common import factory
from common import pipeline

import pickle as pkl

from utils import detector_utils


masterCalib = OpenCVCalib2CameraMeta(LoadOpenCVCalib("res/calib_webcam_mshd_vga.json"))
transform = None


def mono_hand_loop(acq, config, output_file, track=False):
    print("Initialize WACV18 3D Pose estimator (IK)...")
    pose_estimator = factory.HandPoseEstimator(config)

    print("Initialize MVA19 CVRL Hand pose net...")
    estimator = mva19.Estimator(config["model_file"], config["input_layer"], config["output_layer"])

    detection_graph, sess = detector_utils.load_inference_graph()

    started = True
    ik_ms = est_ms = 0
    p2d = bbox = None
    count = 0
    smoothing = config.get("smoothing", 0)
    boxsize = config["boxsize"]
    stride = config["stride"]
    peaks_thre = config["peaks_thre"]
    joint_positions = []
    print("Entering main Loop.")
    while True:
        print('Estimating pose for frame {}'.format(count))
        try:
            imgs, clbs = acq.grab()

            if imgs is None or len(imgs) == 0:
                break
        except Exception as e:
            print("Failed to grab", e)
            break

        st = time.time()
        bgr = imgs[0]
        clb = clbs[0]

        # compute kp using model initial pose
        points2d = pose_estimator.ba.decodeAndProject(pose_estimator.model.init_pose, clb)
        oldKp = np.array(points2d).reshape(-1, 2)

        score = -1
        result_pose = None
        # STEP2 detect 2D joints for the detected hand.
        if started:
            if bbox is None:
                bbox = detector_utils.hand_bbox(bgr, detection_graph, sess)
                if bbox is None:
                    continue
            else:
                dbox = detector_utils.hand_bbox(bgr, detection_graph, sess)
                if dbox is not None:
                    x, y, w, h = bbox
                    x1, y1, w1, h1 = dbox
                    if (x1 > x + w or x1 + w1 < x) or y1 + h1 < y or y1 > y + h:
                        bbox = dbox
                        print("updated")

            x, y, w, h = bbox
            crop = bgr[y:y + h, x:x + w]
            img, pad = mva19.preprocess(crop, boxsize, stride)
            t = time.time()
            hm = estimator.predict(img)
            est_ms = (time.time() - t)

            # use joint tools to recover keypoints
            scale = float(boxsize) / float(crop.shape[0])
            scale = stride / scale
            ocparts = np.zeros_like(hm[..., 0])
            peaks = jt.FindPeaks(hm[..., :-1], ocparts, peaks_thre, scale, scale)

            # convert peaks to hand keypoints
            hand = mva19.peaks_to_hand(peaks, x, y)

            if hand is not None:
                keypoints = hand

                mask = keypoints[:, 2] < peaks_thre
                keypoints[mask] = [0, 0, 1.0]

                if track:
                    keypoints[mask, :2] = oldKp[mask]

                keypoints[:, 2] = keypoints[:, 2] ** 3

                rgbKp = IK.Observations(IK.ObservationType.COLOR, clb, keypoints)
                obsVec = IK.ObservationsVector([rgbKp, ])
                t = time.time()
                score, res = pose_estimator.estimate(obsVec)
                ik_ms = (time.time() - t)

                if track:
                    result_pose = list(
                        smoothing * np.array(pose_estimator.model.init_pose) + (1.0 - smoothing) * np.array(res))
                else:
                    result_pose = list(res)
                # score is the residual, the lower the better, 0 is best
                # -1 is failed optimization.
                if track:
                    if -1 < score:  # < 150:
                        pose_estimator.model.init_pose = Core.ParamVector(result_pose)
                    else:
                        print("\n===>Reseting init position for IK<===\n")
                        pose_estimator.model.reset_pose()
                        bbox = None

                if score > -1:  # compute result points
                    p2d = np.array(pose_estimator.ba.decodeAndProject(Core.ParamVector(result_pose), clb)).reshape(-1,
                                                                                                                   2)
                    bbox = mva19.update_bbox(p2d, bgr.shape[1::-1])

        # get the coordinates in 3D
        p3d = np.array(pose_estimator.ba.decode(Core.ParamVector(result_pose), clb)).reshape(-1, 3)

        # save the coordinates to file
        joint_positions.append(p3d)
        with open(output_file, 'wb') as f:
            pkl.dump(np.asarray(joint_positions), f)

        # increment the counter
        count += 1


limbSeq = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17],  # palm
           [1, 2], [2, 3], [3, 4],  # thump
           [5, 6], [6, 7], [7, 8],  # index
           [9, 10], [10, 11], [11, 12],  # middle
           [13, 14], [14, 15], [15, 16],  # ring
           [17, 18], [18, 19], [19, 20],  # pinky
           ]

final_result = []


def to_AC_format(p3d):
    global transform

    d = {}
    for i, pair in enumerate(limbSeq):
        a = p3d[pair[0]]
        b = p3d[pair[1]]
        bone_center = _center(a, b)
        tag = "boneCenter0"
        if i in range(5):
            tag += str(i) + '0'
        else:
            finger = int((i - 5) / 3)
            bone = (i - 5) - finger * 3 + 1
            tag = tag + str(finger) + str(bone)
        if transform is not None:
            bone_center = np.dot(transform, np.append(bone_center, 1))

        d[tag + 'X'] = -bone_center[2]
        d[tag + 'Y'] = -bone_center[0]
        d[tag + 'Z'] = bone_center[1]

    final_result.append(d)


def _center(a, b):
    return [(v[0] + v[1]) / 2000 for v in list(zip(a, b))]


if __name__ == '__main__':
    config = {
        "model": "models/hand_skinned.xml", "model_left": False,
        "model_init_pose": [-109.80840809323652, 95.70022984677065, 584.613931114289, 292.3322807284121,
                            -1547.742897973965, -61.60146881490577, 435.33025195547793, 1.5707458637241434,
                            0.21444030289465843, 0.11033385117688158, 0.021952050059337137, 0.5716581133215294,
                            0.02969734913698679, 0.03414155945643072, 0.0, 1.1504613679382742, -0.5235922979328,
                            0.15626331136368257, 0.03656410417088128, 8.59579088582312e-07, 0.35789633949684985,
                            0.00012514308785717494, 0.005923001258945023, 0.24864102398139007, 0.2518954858979162, 0.0,
                            3.814694400000002e-13],
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

    video_filename = '/dataset/hand_example/hand_example_video_only/alitrial.avi'
    save_filename = '/code/finger_joints.pkl'

    acq = OpenCVGrabber(video_filename, calib_file="res/calib_webcam_mshd_vga.json")
    acq.initialize()
    mono_hand_loop(acq, config, save_filename, track=True)