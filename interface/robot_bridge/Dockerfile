FROM ros:jazzy-ros-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ROS_DOMAIN_ID=15 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-control-msgs \
    python3-pip \
    python3-venv \
    python3-numpy \
    python3-pil \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
COPY robot_bridge/requirements.txt /ros2_ws/requirements.txt
RUN pip install --break-system-packages -r /ros2_ws/requirements.txt

COPY robot_bridge/ros2_ws/src /ros2_ws/src

RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build --merge-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3"

COPY robot_bridge/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "swiss_robot_bridge.bridge_node"]
