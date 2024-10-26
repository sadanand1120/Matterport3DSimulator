# Usage: ./container.sh [--name CONTAINER_NAME] [--flag1=bla1 --flag2=bla2 ...] [IMAGE_NAME]
#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

DEFAULT_IMAGE_NAME="vln"
DEFAULT_CONTAINER_NAME="vln"

# Initialize variables for options and image name
FLAGS=""
IMAGE_NAME=""
CONTAINER_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --name) # Check for --name flag
      CONTAINER_NAME="$2"
      shift 2 # Shift past the flag and its value
      ;;
    --*) # Any other argument starting with "--" is treated as a flag
      FLAGS="$FLAGS $1"
      shift # Move to next argument
      ;;
    *)  # Anything else is treated as the image name
      IMAGE_NAME="$1"
      shift
      ;;
  esac
done

# If no image name is provided, use the default image name
IMAGE_NAME=${IMAGE_NAME:-$DEFAULT_IMAGE_NAME}
echo "Using image: $IMAGE_NAME"

# If no --name flag is provided, use a default container name
CONTAINER_NAME=${CONTAINER_NAME:-$DEFAULT_CONTAINER_NAME}
echo "Using container name: $CONTAINER_NAME"

echo "Additional flags: $FLAGS"

echo "HOST_UID=$HOST_UID"
echo "DISPLAY=$DISPLAY"

echo "[INFO] DISPLAY might NOT properly work the FIRST time you enter the container after creation. Just exit and re-enter the container."
# DO NOT use --hostname flag, it will cause issues with podman
docker run -it \
    --name $CONTAINER_NAME \
    --network host \
    --ipc host \
    --workdir /root \
    --group-add dialout \
    --privileged \
    -e XAUTHORITY=/root/.Xauthority \
    -e HOST_UID=${HOST_UID} \
    -v /tmp:/tmp \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v ${HOME}/.Xauthority:/root/.Xauthority:rw \
    -v /dev/dri:/dev/dri:ro \
    -v ${HOME}/.gitconfig:/root/.gitconfig:rw \
    -v `pwd`:/root/mount/Matterport3DSimulator:rw \
    --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans \
    -u 0:0 \
    $FLAGS \
    $IMAGE_NAME


# cd /root/mount/Matterport3DSimulator && feh teaser.jpg

# docker run -it --name trial --hostname rlidar --network host --ipc host --workdir /root --group-add dialout --privileged -u 0:0 -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v ${HOME}/.Xauthority:/root/.Xauthority:rw -v /dev/dri:/dev/dri:ro -v ${HOME}/.gitconfig:/root/.gitconfig:rw -v /tmp:/tmp -e DISPLAY=${DISPLAY} -e XAUTHORITY=/root/.Xauthority --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans -v `pwd`:/root/mount/Matterport3DSimulator:rw monitor

# docker run -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans --volume `pwd`:/root/mount/Matterport3DSimulator:rw --volume "$HOME/.Xauthority:/root/.Xauthority:rw" --net=host --ipc=host --name temp monitor