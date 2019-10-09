#!/bin/bash

OUT_DIR=$1
echo "Saving output to $OUT_DIR"

## Create the directory to save data to
if [ -d "$OUT_DIR" ]; then
  # Take action if directory exists
  echo "$OUT_DIR already exists, please save to a new directory"
  exit 1
else
  # Control will jump here if directory does NOT exist
  mkdir -p $OUT_DIR
fi

## Record a rosbag for 5 seconds
rosbag record /camera/depth_registered/camera_info /camera/depth_registered/image /camera/depth_registered/points /camera/rgb/camera_info /camera/rgb/image_raw /camera/rgb/image_rect_color --output-name=$OUT_DIR/rawbag.bag #--duration=5

#rosrun image_view image_saver image:=/camera/rgb/image_rect_color
#rosrun image_view video_recorder image:=/camera/rgb/image_rect_color filename:=hand_example
