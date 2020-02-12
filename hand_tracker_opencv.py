from __future__ import division
import os
import cv2
import time
import numpy as np
import argparse
import copy


POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4],
              [0, 5], [5, 6], [6, 7], [7, 8],
              [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16],
              [0, 17], [17, 18], [18, 19], [19, 20]]
NUM_POINTS = 21
THRESHOLD = 0.5


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
        self.proto_file = args.proto_file
        self.weights_file = args.weights_file
        self.visualize = args.visualize
        self.width = 640
        self.height = 480
        self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)
        self.save_filename = args.joint_anno_path

        object_name = args.object_model
        if not object_name:
            self.rgb_image, self.dep_image = self.get_images(args.rgb, args.depth)
            points = self.process()
            save_annotation_file(self.save_filename, [os.path.basename(args.rgb)], [points])
        else:
            self.visualize = False
            image_names = get_image_names(args.object_anno_path, object_name)
            processed_images = []
            processed_points = []
            for im in image_names:
                rgb_filename = os.path.join(args.frame_root_path, im)
                dep_filename = list(im)
                dep_filename[6] = 'D'
                dep_filename = ''.join(dep_filename)
                dep_filename = os.path.join(args.depth_frame_root_path, dep_filename)
                self.rgb_image, self.dep_image = self.get_images(rgb_filename, dep_filename)
                points = self.process()
                processed_images.append(im)
                processed_points.append(points)
                save_annotation_file(self.save_filename, processed_images, processed_points)

    def process(self):
        self.width = self.rgb_image.shape[1]
        self.height = self.rgb_image.shape[0]
        aspect_ratio = self.width / self.height

        t = time.time()

        # input image dimensions for the network
        in_height = 368
        in_width = int(((aspect_ratio*in_height)*8)//8)
        net_input = cv2.dnn.blobFromImage(self.rgb_image, 1.0 / 255, (in_width, in_height), (0, 0, 0),
                                          swapRB=False, crop=False)

        self.net.setInput(net_input)

        pred = self.net.forward()
        print("Time taken by network : {:.3f}".format(time.time() - t))

        points = self.get_keypoints(pred)

        points3D = self.get_world_coordinates(points)
        coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        points3D = points3D.dot(coord_change_mat.T)
        points3D = to_iccv_format(points3D)

        if self.visualize:
            self.draw(points)

        return points3D

    @staticmethod
    def get_images(rgb_filename, dep_filename):
        rgb_image = cv2.imread(rgb_filename)
        dep_image = cv2.imread(dep_filename)
        depth_scale = 0.00012498664727900177
        dep_image = dep_image[:, :, 2] + dep_image[:, :, 1] * 256
        dep_image = dep_image * depth_scale

        return rgb_image, dep_image

    def get_keypoints(self, pred):
        # Empty list to store the detected keypoints
        points = []

        for i in range(NUM_POINTS):
            # confidence map of corresponding body's part.
            prob_map = pred[0, i, :, :]
            prob_map = cv2.resize(prob_map, (self.width, self.height))

            # Find global maxima of the probability map
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            if prob > THRESHOLD:
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)
        
        return points

    def get_world_coordinates(self, points):
        points_3D = np.zeros((len(points), 3))

        ux = 312.42
        uy = 241.42
        fx = 617.343
        fy = 617.343
        i_fx = 1 / fx
        i_fy = 1 / fy

        for i in range(len(points)):
            if points[i] is not None:
                points_3D[i, 2] = self.dep_image[points[i][1], points[i][0]] * 1000
                points_3D[i, 0] = (points[i][0] - ux) * points_3D[i, 2] * i_fx
                points_3D[i, 1] = (points[i][1] - uy) * points_3D[i, 2] * i_fy
            else:
                points_3D[i, :] = 0.

        return points_3D

    def draw(self, points):
        # Draw the keypoints
        rgb_keypoints = np.copy(self.rgb_image)
        for i in range(len(points)):
            if points[i] is not None:
                cv2.circle(rgb_keypoints, (int(points[i][0]), int(points[i][1])), 8, (0, 255, 255),
                           thickness=-1, lineType=cv2.FILLED)
                cv2.putText(rgb_keypoints, "{}".format(i), (int(points[i][0]), int(points[i][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Draw Skeleton
        rgb_skeleton = np.copy(self.rgb_image)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(rgb_skeleton, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(rgb_skeleton, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(rgb_skeleton, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        cv2.imshow('Output-Keypoints', rgb_keypoints)
        cv2.imshow('Output-Skeleton', rgb_skeleton)

        # cv2.imwrite('Output-Keypoints.jpg', rgb_keypoints)
        # cv2.imwrite('Output-Skeleton.jpg', rgb_skeleton)

        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand tracker using OpenCV 2')
    parser.add_argument('--rgb', type=str, required=False,
                        default='/home/tpatten/Data/Hands/HANDS_Challenge_ICCV_2019/Task3/training_images/image_C00000000.png',
                        help='Name of the rgb image')
    parser.add_argument('--depth', type=str, required=False,
                        default='/home/tpatten/Data/Hands/HANDS_Challenge_ICCV_2019/Task3/training_images_depth/image_D00000000.png',
                        help='Name of the depth image')
    parser.add_argument('--object-model', type=str, default='', required=False,
                        help='Name of the object model')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    args.proto_file = 'caffe_models/pose_deploy.prototxt'
    args.weights_file = 'caffe_models/pose_iter_102000.caffemodel'
    args.joint_anno_path = '/home/tpatten/Data/Hands/HANDS_Challenge_ICCV_2019/Task3/keypoint_joint_annotation.txt'
    args.object_anno_path = '/home/tpatten/Data/Hands/HANDS_Challenge_ICCV_2019/Task3/training_object_annotation_small.txt'
    args.frame_root_path = '/home/tpatten/Data/Hands/HANDS_Challenge_ICCV_2019/Task3/training_images_small/'
    args.depth_frame_root_path = '/home/tpatten/Data/Hands/HANDS_Challenge_ICCV_2019/Task3/training_images_depth/'

    hand_tracker = HandTracker(args)
