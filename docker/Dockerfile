FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-control-msgs \
        ros-humble-teleop-twist-keyboard \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers \
        ros-humble-rqt-joint-trajectory-controller \
        python3-pykdl \
    && rm -rf /var/lib/apt/lists/*

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
