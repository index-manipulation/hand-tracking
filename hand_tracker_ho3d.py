# bala paen 1 , chap rast 2 , jelo aghab 3
# pylint: disable=wrong-import-position
'''
Adapted from the MonoHand3D codebase for the MonocularRGB_3D_Handpose project (github release)

This script uses the 2D joint estimator of Gouidis et al.

@author: Paschalis Panteleris (padeler@ics.forth.gr)
'''

import sys
sys.path.append("lib")
import os
import time
import cv2
import numpy as np
import PyCeresIK as IK
from common.calibrate import OpenCVCalib2CameraMeta, LoadOpenCVCalib
from common import mva19
from common import factory
from common import pipeline
import PyMBVCore  as Core
import PyJointTools as jt
from utils import detector_utils
import argparse
import pickle


def to_iccv_format(joints):
    # MONOHAND [Wrist (0),
    #           TMCP (1), TPIP (2), TDIP (3), TTIP (4),
    #           IMCP (5), IPIP (6), IDIP (7), ITIP (8),
    #           MMCP (9), MPIP (10), MDIP (11), MTIP (12),
    #           RMCP (13), RPIP (14), RDIP (15), RTIP (16),
    #           PMCP (17), PPIP (18), PDIP (19), PTIP (20)]
    # ICCV     [Wrist,
    #           TMCP, IMCP, MMCP, RMCP, PMCP,
    #           TPIP, TDIP, TTIP,
    #           IPIP, IDIP, ITIP,
    #           MPIP, MDIP, MTIP,
    #           RPIP, RDIP, RTIP,
    #           PPIP, PDIP, PTIP]
    joint_map = [0, 1, 5, 9, 13, 17, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
    iccv_joints = np.zeros(joints.shape)
    for i in range(len(joints)):
        iccv_joints[i, :] = joints[joint_map[i], :]

    return iccv_joints


class HandTracker:
    def __init__(self, args):
        self.clb = OpenCVCalib2CameraMeta(LoadOpenCVCalib("res/calib_hands_task3.json"))
        self.config = {
            "model": "models/hand_skinned.xml", "model_left": False,
            "model_init_pose": [-109.80840809323652, 95.70022984677065, 584.613931114289, 292.3322807284121,
                                -1547.742897973965, -61.60146881490577, 435.33025195547793, 1.5707458637241434,
                                0.21444030289465843, 0.11033385117688158, 0.021952050059337137, 0.5716581133215294,
                                0.02969734913698679, 0.03414155945643072, 0.0, 1.1504613679382742, -0.5235922979328,
                                0.15626331136368257, 0.03656410417088128, 8.59579088582312e-07, 0.35789633949684985,
                                0.00012514308785717494, 0.005923001258945023, 0.24864102398139007, 0.2518954858979162,
                                0.0,
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
            "peaks_thre": 0.1
        }

        self.ho3d_path = args.ho3d_path
        self.models_path = args.models_path
        self.with_renderer = args.with_renderer
        self.track = args.track
        self.paused = args.paused
        self.visualize = args.visualize
        self.save = args.save

        if args.save:
            self.save_path = os.path.join(args.ho3d_path, 'train', args.target, 'hand_tracker')
            if not os.path.exists(self.save_path):
                try:
                    os.makedirs(self.save_path)
                except OSError:
                    print('ERROR: Unable to create the save directory {}'.format(self.save_path))
                    return

        print("Initialize WACV18 3D Pose estimator (IK)...")
        self.pose_estimator = factory.HandPoseEstimator(self.config)
        self.hand_visualizer = None
        if self.with_renderer:
            print("Initialize Hand Visualizer...")
            self.hand_visualizer = pipeline.HandVisualizer(factory.mmanager, (640, 480))

        print("Initialize MVA19 CVRL Hand pose net...")
        self.estimator = mva19.Estimator(self.config["model_file"], self.config["input_layer"], self.config["output_layer"])
        self.detection_graph, self.sess = detector_utils.load_inference_graph()

        # Loop variables
        self.started = True
        self.delay = {True: 0, False: 1}
        self.ik_ms = self.est_ms = 0
        self.p2d = self.bbox = None
        self.smoothing = self.config.get("smoothing", 0)
        self.boxsize = self.config["boxsize"]
        self.stride = self.config["stride"]
        self.peaks_thre = self.config["peaks_thre"]

        self.rgb_path = os.path.join(args.ho3d_path, 'train', args.target, 'rgb')
        self.meta_path = os.path.join(args.ho3d_path, 'train', args.target, 'meta')

        self.mono_hand_loop()

    def mono_hand_loop(self):
        print("Entering main Loop")
        saved_hand_poses = []
        saved_framed_ids = []
        file_list = sorted(os.listdir(self.rgb_path))
        for im in file_list:
            # Get the image
            frame_path = os.path.join(self.rgb_path, im)
            bgr = cv2.imread(frame_path)

            # Get the hand pose
            st = time.time()
            result_pose, hand, score = self.estimate_hand_pose(bgr)
            print('Score for frame {}: {}'.format(im, score))

            # 2D visualization
            if self.visualize:
                if not self.opencv_vis(bgr, result_pose, hand, st):
                    break

            # Get 3D coordinates
            p3d = np.array(self.pose_estimator.ba.decode(Core.ParamVector(result_pose), self.clb)).reshape(-1, 3)

            # Transform to OpenGL coordinates and ICCV format
            coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
            p3d = p3d.dot(coord_change_mat.T)
            p3d = to_iccv_format(p3d)

            # Save
            if self.save:
                save_filename = os.path.join(self.save_path, im)
                save_filename = save_filename.replace(".png", ".pkl")
                with open(save_filename, 'wb') as f:
                    save_data = {}
                    save_data['handJoints3D'] = p3d
                    save_data['score'] = score
                    pickle.dump(save_data, f)

    def estimate_hand_pose(self, bgr):
        # Compute kp using model initial pose
        points2d = self.pose_estimator.ba.decodeAndProject(self.pose_estimator.model.init_pose, self.clb)
        oldKp = np.array(points2d).reshape(-1, 2)

        result_pose = None
        hand = None
        score = 1000.0

        # STEP2 detect 2D joints for the detected hand
        if self.started:
            if self.bbox is None:
                self.bbox = detector_utils.hand_bbox(bgr, self.detection_graph, self.sess)
                if self.bbox is None:
                    cv2.imshow("2D CNN estimation", bgr)
                    cv2.waitKey(1)
                    return result_pose, hand, score
            else:
                dbox = detector_utils.hand_bbox(bgr, self.detection_graph, self.sess)
                if dbox is not None:
                    x, y, w, h = self.bbox
                    x1, y1, w1, h1 = dbox
                    if (x1 > x + w or x1 + w1 < x) or y1 + h1 < y or y1 > y + h:
                        self.bbox = dbox
                        print("updated")

            x, y, w, h = self.bbox
            crop = bgr[y:y + h, x:x + w]
            img, pad = mva19.preprocess(crop, self.boxsize, self.stride)
            t = time.time()
            hm = self.estimator.predict(img)
            self.est_ms = (time.time() - t)

            # Use joint tools to recover keypoints
            scale = float(self.boxsize) / float(crop.shape[0])
            scale = self.stride / scale
            ocparts = np.zeros_like(hm[..., 0])
            peaks = jt.FindPeaks(hm[..., :-1], ocparts, self.peaks_thre, scale, scale)

            # Convert peaks to hand keypoints
            hand = mva19.peaks_to_hand(peaks, x, y)

            if hand is not None:
                keypoints = hand

                mask = keypoints[:, 2] < self.peaks_thre
                keypoints[mask] = [0, 0, 1.0]

                if self.track:
                    keypoints[mask, :2] = oldKp[mask]

                keypoints[:, 2] = keypoints[:, 2] ** 3

                rgbKp = IK.Observations(IK.ObservationType.COLOR, self.clb, keypoints)
                obsVec = IK.ObservationsVector([rgbKp, ])
                t = time.time()
                score, res = self.pose_estimator.estimate(obsVec)
                self.ik_ms = (time.time() - t)

                if self.track:
                    result_pose = list(
                        self.smoothing * np.array(self.pose_estimator.model.init_pose) + (1.0 - self.smoothing) * np.array(res))
                else:
                    result_pose = list(res)

                # Score is the residual, the lower the better, 0 is best
                # -1 is failed optimization.
                if self.track:
                    if -1 < score:  # < 150:
                        self.pose_estimator.model.init_pose = Core.ParamVector(result_pose)
                    else:
                        print("\n===> Resetting init position for IK <===\n")
                        self.pose_estimator.model.reset_pose()
                        self.bbox = None

                if score > -1:  # compute result points
                    self.p2d = np.array(self.pose_estimator.ba.decodeAndProject(
                        Core.ParamVector(result_pose), self.clb)).reshape(-1, 2)
                    self.bbox = mva19.update_bbox(self.p2d, bgr.shape[1::-1])

        return result_pose, hand, score

    def opencv_vis(self, bgr, result_pose, hand, st):
        viz = np.copy(bgr)
        viz2d = np.zeros_like(bgr)
        if self.started and result_pose is not None:
            viz2d = mva19.visualize_2dhand_skeleton(viz2d, hand, thre=self.peaks_thre)
            cv2.imshow("2D CNN estimation", viz2d)
            header = "FPS OPT+VIZ %03d, OPT %03d (CNN %03d, 3D %03d)" % (
                1 / (time.time() - st), 1 / (self.est_ms + self.ik_ms), 1.0 / self.est_ms, 1.0 / self.ik_ms)

            if self.with_renderer:
                self.hand_visualizer.render(self.pose_estimator.model, Core.ParamVector(result_pose), self.clb)
                mbv_viz = self.hand_visualizer.getDepth()
                cv2.imshow("MBV VIZ", mbv_viz)
                mask = mbv_viz != [0, 0, 0]
                viz[mask] = mbv_viz[mask]
            else:
                viz = mva19.visualize_3dhand_skeleton(viz, self.p2d)
                pipeline.draw_rect(viz, "Hand", self.bbox, box_color=(0, 255, 0), text_color=(200, 200, 0))
        else:
            header = "Press 's' to start, 'r' to reset pose, 'p' to pause frame."

        cv2.putText(viz, header, (20, 20), 0, 0.7, (50, 20, 20), 1, cv2.LINE_AA)
        cv2.imshow("3D Hand Model re-projection", viz)

        key = cv2.waitKey(self.delay[self.paused])
        if key & 255 == ord('p'):
            self.paused = not self.paused
        if key & 255 == ord('q'):
            return False
        if key & 255 == ord('r'):
            print("\n===> Resetting init position for IK <===\n")
            self.pose_estimator.model.reset_pose()
            bbox = None
            print("RESETTING BBOX", bbox)
        if key & 255 == ord('s'):
            self.started = not self.started

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand pose estimator for HO3D dataset')
    parser.add_argument("-target", type=str, help="Name of the target subset",
                        choices=['ABF10', 'BB10', 'GPMF10', 'GSF10', 'MDF10', 'ShSu10'], default='ABF10')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.models_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models'
    args.visualize = True
    args.save = False
    args.with_renderer = False
    args.track = True
    args.paused = True

    print(args)

    hand_tracker = HandTracker(args)

