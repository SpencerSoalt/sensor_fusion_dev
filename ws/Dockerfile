FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

#  Locale  
RUN apt-get update && apt-get install -y --no-install-recommends locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

#  System / ROS deps 
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-vision-msgs \
    ros-humble-visualization-msgs \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-tf2-ros \
    ros-humble-tf2-sensor-msgs \
    ros-humble-message-filters \
    ros-humble-sensor-msgs-py \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-foxglove-bridge \
    ros-humble-rmw-cyclonedds-cpp \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    tmux \
    vim \
    && rm -rf /var/lib/apt/lists/*

#  DDS defaults 
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

#  Python deps 
RUN python3 -m pip install --no-cache-dir \
    "numpy==1.26.4" \
    scikit-learn

#  ROS workspace 

ENV WS=/ws
WORKDIR ${WS}

# Copy your package into the workspace
COPY src ${WS}/src/

#  rosdep 
RUN rosdep init 2>/dev/null || true && \
    rosdep update && \
    source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y

#  Build 
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

#  Entrypoint 
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
