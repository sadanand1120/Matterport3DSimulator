#!/bin/bash
# first unmount if already mounted
fusermount -u /home/dynamo/Music/Matterport3DSimulator/mp3d_unzipped/v1/scans
#fusermount -u /home/dynamo/Music/Matterport3DSimulator/thirdparty/DepthAnythingV2/metric_depth/checkpoints
#fusermount -u /home/dynamo/Music/Matterport3DSimulator/thirdparty/Grounded-Segment-Anything/weights
fusermount -u /home/dynamo/Music/Matterport3DSimulator/thirdparty/NaviLLM/data

sleep 5

sshfs -o allow_root robovision:/robodata/smodak/r2r/mp3d_unzipped/v1/scans /home/dynamo/Music/Matterport3DSimulator/mp3d_unzipped/v1/scans/
#sshfs -o allow_root robovision:/robodata/smodak/sotavln/repos/Matterport3DSimulator/thirdparty/DepthAnythingV2/metric_depth/checkpoints /home/dynamo/Music/Matterport3DSimulator/thirdparty/DepthAnythingV2/metric_depth/checkpoints/
#sshfs -o allow_root robovision:/robodata/smodak/sotavln/repos/Matterport3DSimulator/thirdparty/Grounded-Segment-Anything/weights /home/dynamo/Music/Matterport3DSimulator/thirdparty/Grounded-Segment-Anything/weights/
sshfs -o allow_root robovision:/robodata/smodak/sotavln/repos/Matterport3DSimulator/thirdparty/NaviLLM/data /home/dynamo/Music/Matterport3DSimulator/thirdparty/NaviLLM/data/
