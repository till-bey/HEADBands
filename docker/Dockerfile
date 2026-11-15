FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-control-msgs \
        ros-humble-teleop-twist-keyboard \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers \
        ros-humble-rqt-joint-trajectory-controller \
        ros-humble-canopen-interfaces \
        python3-pykdl \
        python3-pip \
        python3-pil \
    && rm -rf /var/lib/apt/lists/*

# Cloud interface bridge (interface/robot_bridge, started with docker/bridge.sh): outbound WebSocket to GCP.
RUN python3 -m pip install --no-cache-dir "websockets>=14,<16" "pydantic>=2.6,<3"

ENV ROS_DOMAIN_ID=15
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

COPY src/unix_msgs /tmp/unix_ws/src/unix_msgs
RUN . /opt/ros/humble/setup.sh \
    && cd /tmp/unix_ws \
    && colcon build \
        --merge-install \
        --install-base /opt/unix_ws \
        --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
    && rm -rf /tmp/unix_ws

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && printf '\n[ -f /workspace/docker/shell.sh ] && . /workspace/docker/shell.sh\n' >> /root/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
