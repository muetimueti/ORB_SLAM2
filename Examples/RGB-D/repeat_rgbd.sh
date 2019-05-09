#!/bin/bash

for i in {1..5}
do
    /home/ralph/CLionProjects/ORB_SLAM2/Examples/RGB-D/rgbd_tum /home/ralph/CLionProjects/ORB_SLAM2/Vocabulary/ORBvoc.txt /home/ralph/CLionProjects/ORB_SLAM2/Examples/RGB-D/TUM1.yaml /home/ralph/SLAM/rgbd_dataset_freiburg1_xyz/ /home/ralph/SLAM/rgbd_dataset_freiburg1_xyz/associate.txt
done
