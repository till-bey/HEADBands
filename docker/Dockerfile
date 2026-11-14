FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-control-msgs \
        ros-jazzy-teleop-twist-keyboard \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DOMAIN_ID=15
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

COPY src/unix_msgs /tmp/unix_ws/src/unix_msgs
RUN . /opt/ros/jazzy/setup.sh \
    && cd /tmp/unix_ws \
    && colcon build \
        --merge-install \
        --install-base /opt/unix_ws \
        --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
    && rm -rf /tmp/unix_ws

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
