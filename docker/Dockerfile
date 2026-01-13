FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# ---- basic ----
RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release ca-certificates \
    build-essential git \
    python3 python3-pip python3-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# ---- ROS Noetic ----
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
 && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros1.list \
 && apt-get update && apt-get install -y \
    ros-noetic-ros-base \
    ros-noetic-cv-bridge ros-noetic-image-transport \
    ros-noetic-vision-msgs \
    ros-noetic-tf2-ros ros-noetic-tf2-geometry-msgs \
    ros-noetic-usb-cam \
    python3-catkin-tools \
 && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# ---- Python libs (YOLO11) ----
RUN python3 -m pip install --no-cache-dir -U pip \
 && python3 -m pip install --no-cache-dir ultralytics

# ---- workspace ----
ENV CATKIN_WS=/catkin_ws
RUN mkdir -p ${CATKIN_WS}/src
WORKDIR ${CATKIN_WS}

# entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
