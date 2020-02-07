# bala paen 1 , chap rast 2 , jelo aghab 3
# pylint: disable=wrong-import-position
'''
Adapted from the MonoHand3D codebase for the MonocularRGB_3D_Handpose project (github release)

This script uses the 2D joint estimator of Gouidis et al.

@author: Paschalis Panteleris (padeler@ics.forth.gr)
'''

import sys
sys.path.append("lib")
from os.path import join
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


def get_image_names(object_anno_path, object_name):
    # Open the object annotation file
    image_names = []
    with open(object_anno_path, 'r') as file:
        # read each line
        for anno in file:
            # Split the line into words
            anno = anno.split('\t')
            if anno[-1] == '\n':
                anno = anno[:-1]
            # Get the object name
            object_id = anno[1]
            if object_id == object_name:
                img_name = anno[0]
                image_names.append(img_name)

    # Return all the frame ids for the selected object
    return image_names


def save_annotation_file(filename, frame_ids, hand_poses):
    # Save to file
    f = open(filename, "w")
    for i in range(len(frame_ids)):
        f.write(frame_ids[i])
        f.write('\t')
        for p in hand_poses[i].flatten():
            f.write('{}\t'.format(p))
        f.write('\n')
    f.close()


def to_iccv_format(joints):
    # MONPHAND [Wrist (0),
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

        self.frame_root_path = args.frame_root_path
        self.object_name = args.object_model
        self.joint_anno_path = args.joint_anno_path
        self.object_anno_path = args.object_anno_path
        self.monohand_image_path = args.monohand_image_path

        self.mono_hand_loop((640, 480), track=True, with_renderer=True)

    def mono_hand_loop(self, out_size, track=False, paused=False, with_renderer=False):
        print("Initialize WACV18 3D Pose estimator (IK)...")
        pose_estimator = factory.HandPoseEstimator(self.config)
        hand_visualizer = None
        if with_renderer:
            print("Initialize Hand Visualizer...")
            hand_visualizer = pipeline.HandVisualizer(factory.mmanager, out_size)

        print("Initialize MVA19 CVRL Hand pose net...")
        estimator = mva19.Estimator(self.config["model_file"], self.config["input_layer"], self.config["output_layer"])

        detection_graph, sess = detector_utils.load_inference_graph()

        started = True
        delay = {True: 0, False: 1}
        ik_ms = est_ms = 0
        p2d = bbox = None
        smoothing = self.config.get("smoothing", 0)
        boxsize = self.config["boxsize"]
        stride = self.config["stride"]
        peaks_thre = self.config["peaks_thre"]
        print("Entering main Loop")
        saved_hand_poses = []
        saved_framed_ids = []
        image_names = get_image_names(self.object_anno_path, self.object_name)
        for i_name in image_names:
            frame_path = join(self.frame_root_path, i_name)
            bgr = cv2.imread(frame_path)
            st = time.time()

            # Compute kp using model initial pose
            points2d = pose_estimator.ba.decodeAndProject(pose_estimator.model.init_pose, self.clb)
            oldKp = np.array(points2d).reshape(-1, 2)

            result_pose = None
            hand = None
            # STEP2 detect 2D joints for the detected hand
            if started:
                if bbox is None:
                    bbox = detector_utils.hand_bbox(bgr, detection_graph, sess)
                    if bbox is None:
                        cv2.imshow("2D CNN estimation", bgr)
                        cv2.waitKey(1)
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

                # Use joint tools to recover keypoints
                scale = float(boxsize) / float(crop.shape[0])
                scale = stride / scale
                ocparts = np.zeros_like(hm[..., 0])
                peaks = jt.FindPeaks(hm[..., :-1], ocparts, peaks_thre, scale, scale)

                # Convert peaks to hand keypoints
                hand = mva19.peaks_to_hand(peaks, x, y)

                if hand is not None:
                    keypoints = hand

                    mask = keypoints[:, 2] < peaks_thre
                    keypoints[mask] = [0, 0, 1.0]

                    if track:
                        keypoints[mask, :2] = oldKp[mask]

                    keypoints[:, 2] = keypoints[:, 2] ** 3

                    rgbKp = IK.Observations(IK.ObservationType.COLOR, self.clb, keypoints)
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
                            print("\n===> Resetting init position for IK <===\n")
                            pose_estimator.model.reset_pose()
                            bbox = None

                    if score > -1:  # compute result points
                        p2d = np.array(pose_estimator.ba.decodeAndProject(
                            Core.ParamVector(result_pose), self.clb)).reshape(-1, 2)
                        bbox = mva19.update_bbox(p2d, bgr.shape[1::-1])

            viz = np.copy(bgr)
            viz2d = np.zeros_like(bgr)
            if started and result_pose is not None:
                viz2d = mva19.visualize_2dhand_skeleton(viz2d, hand, thre=peaks_thre)
                cv2.imshow("2D CNN estimation", viz2d)
                header = "FPS OPT+VIZ %03d, OPT %03d (CNN %03d, 3D %03d)" % (
                    1 / (time.time() - st), 1 / (est_ms + ik_ms), 1.0 / est_ms, 1.0 / ik_ms)

                if with_renderer:
                    hand_visualizer.render(pose_estimator.model, Core.ParamVector(result_pose), self.clb)
                    mbv_viz = hand_visualizer.getDepth()
                    cv2.imshow("MBV VIZ", mbv_viz)
                    mask = mbv_viz != [0, 0, 0]
                    viz[mask] = mbv_viz[mask]
                else:
                    viz = mva19.visualize_3dhand_skeleton(viz, p2d)
                    pipeline.draw_rect(viz, "Hand", bbox, box_color=(0, 255, 0), text_color=(200, 200, 0))
            else:
                header = "Press 's' to start, 'r' to reset pose, 'p' to pause frame."

            cv2.putText(viz, header, (20, 20), 0, 0.7, (50, 20, 20), 1, cv2.LINE_AA)
            cv2.imshow("3D Hand Model re-projection", viz)

            key = cv2.waitKey(delay[paused])
            if key & 255 == ord('p'):
                paused = not paused
            if key & 255 == ord('q'):
                break
            if key & 255 == ord('r'):
                print("\n===> Resetting init position for IK <===\n")
                pose_estimator.model.reset_pose()
                bbox = None
                print("RESETTING BBOX", bbox)
            if key & 255 == ord('s'):
                started = not started

            p3d = np.array(pose_estimator.ba.decode(Core.ParamVector(result_pose), self.clb)).reshape(-1, 3)

            # Transform to OpenGL coordinates and ICCV format
            coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
            p3d = p3d.dot(coord_change_mat.T)
            p3d = to_iccv_format(p3d)

            # Save
            # saved_hand_poses.append(p3d)
            # saved_framed_ids.append(i_name)

            # Save any frames where a good score is achieved
            # score is the residual, the lower the better, 0 is best
            if score < 10:
                print('Saved good pose:', i_name)
                img_save_filename = join(self.monohand_image_path, i_name)
                cv2.imwrite(img_save_filename, viz)
                saved_hand_poses.append(p3d)
                saved_framed_ids.append(i_name)

        # Save data
        save_annotation_file(self.joint_anno_path, saved_framed_ids, saved_hand_poses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Hand pose estimator')
    parser.add_argument('--object-model', type=str, default='003_cracker_box', required=False,
                        help='Name of the object model')
    args = parser.parse_args()
    args.frame_root_path = '/dataset/Hands/HANDS_Challenge_ICCV_2019/Task3/training_images_small'
    args.object_anno_path = '/dataset/Hands/HANDS_Challenge_ICCV_2019/Task3/training_object_annotation_small.txt'
    args.joint_anno_path = '/dataset/Hands/HANDS_Challenge_ICCV_2019/Task3/monohand_joint_annotation.txt'
    args.monohand_image_path = '/dataset/Hands/HANDS_Challenge_ICCV_2019/Task3/monohand_images'

    hand_tracker = HandTracker(args)
