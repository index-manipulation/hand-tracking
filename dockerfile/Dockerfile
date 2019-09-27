FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get update && apt install -y python-rosinstall python-rosinstall-generator python-wstool build-essential wget

ENV PATH="/usr/lib/nvidia-418/bin:${PATH}"
ENV LD_LIBRARY_PATH="/code/lib:/usr/lib/nvidia-418:/usr/lib32/nvidia-418:${LD_LIBRARY_PATH}"

WORKDIR /root/tmp_install_files
COPY ./install-opencv.sh .
RUN chmod +x install-opencv.sh && ./install-opencv.sh

RUN apt install -y libgoogle-glog-dev libtbb-dev libcholmod3.0.6 libatlas-base-dev libopenni0 libbulletdynamics2.83.6 libboost-all-dev libglew-dev

# INSTALLING ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && apt-get update


RUN apt-get install -y ros-kinetic-ros-base && rosdep init && rosdep update

RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc && /bin/bash -c "source /opt/ros/kinetic/setup.bash"
WORKDIR /root/catkin_ws

COPY ./catkin_build_ws /root/catkin_build_ws
RUN apt-get install -y ros-kinetic-ar-track-alvar
RUN apt-get install -y python3-pip python3-yaml && pip3 install rospkg catkin_pkg tqdm
RUN /bin/bash -c  ". /opt/ros/kinetic/setup.bash; cd /root/catkin_ws; mkdir src; catkin_make"
COPY ./create_avc_transformations /root/catkin_ws/src/create_avc_transformations
RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "source /root/catkin_build_ws/devel/setup.bash --extend" >> ~/.bashrc
# WORKDIR /usr/local/lib/python3.5/dist-packages/
RUN ln -sf /usr/local/lib/python3.5/dist-packages/cv2.so cv2.so
RUN echo 'export PYTHONPATH="/usr/local/lib/python3.5/dist-packages:${PYTHONPATH}"' >> ~/.bashrc
