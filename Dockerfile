FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-nav-msgs \
    ros-humble-std-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros_ws

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ src/

RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
