Xavier -NX (setup preparation)
sudo apt-cache show nvidia-jetpack

    • Flash TX2 NX wth Jetpack(5.1.1 rev) by ./install.sh
      (link for setup file: https://onedrive.live.com/?authkey=%21AOpARfjdlrAVHqI&id=10A4D1CF2037CA25%218673&cid=10A4D1CF2037CA25)
      
           how to flash ?
1) Enter to Recovery mode 
power off -> press recovery button -> power on -> wait 2 seconds -> release recovery button
2) Start to flash BSP ( go to inside ‘Linux_for_Tegra’ dir and run ‘install.sh )

    • Shift root file system from internal to SD card.
      (link: https://www.forecr.io/blogs/bsp-development/change-root-file-system-to-sd-card-directly)

         optional:ssh connection enable for headless
            (ssh-keygen -f "/home/gaurav/.ssh/known_hosts" -R "10.250.201.182")
            optional:Transfer file from Host to headless machine
         (scp /path/to/file.txt user@ssh_pc:/path/on/ssh_pc/)

    • Install SDK  components like CUDA,TensorRT,etc by using Nvidia SDK manager ( host normal ubuntu18 machine).
      
    1. CUDA
    2. CUDA-X-AI
    3. Computer Vision
    4. NVIDIA container runtime
    5. Multimedia
    6. Developer Tools
    7. Deepstream
      
    • install pip in jetson :
      
      “sudo apt-get update && sudo apt-get install python-pip python3-pip”
       sudo apt-get install python3-setuptools
      
      
    • install jtop with pip3 and reboot
      
      “sudo pip3 install -U jetson-stats”




Install opencv (with CUDA for jetson)
    • sudo apt update
    • sudo apt install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openjdk-8-jdk
    • cd ~
    • git clone --branch 4.5.4 https://github.com/opencv/opencv.git
    • git clone --branch 4.5.4 https://github.com/opencv/opencv_contrib.git
    • cd opencv
    • mkdir build
    • cd build
    • cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D WITH_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_LIBV4L=ON -D BUILD_opencv_python3=ON -D OPENCV_GENERATE_PKGCONFIG=ON ..
    • make -j4   (take time around 4-5hr on xavier NX 8gb).
      for case of enough hardware use this >>> make -j$(nproc) 
    • sudo make install
    • sudo ldconfig

Install pytorch for Jetson (skip, it was old try)
      ( link: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 )
      
                   JetPack 5.1 (L4T R35.2.1) / JetPack 5.1.1 (L4T R35.3.1) 
          Python 3.8 – torch v2.0.0, torchvision v0.15.1
        ◦ wget https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl -O torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
        ◦ sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
        ◦ pip3 install Cython 
        ◦ sudo pip3 install numpy torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
        ◦ sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
        ◦ git clone --branch v0.15.1 https://github.com/pytorch/vision torchvision
        ◦ cd torchvision
        ◦ export BUILD_VERSION=0.15.1
        ◦ python3 setup.py install –user
        ◦ cd ../
        ◦ pip install ‘pillow<7’

note:torchvision(0.15.2) requires torch==2.0.1, but you have torch 2.0.0+nv23.5 which is incompatible.

verification:
      >>> import torch
                >>> print(torch.__version__)
                >>> import torchvision
                >>> print(torchvision.__version__)

     Install VS Code (arm64) on Jetson
    • go to VS code website and click on download for other platform
    • downlaod .deb Arm64 file
    • go to download path and run below comd to install vs code
    • sudo apt install ./codeVS.deb
Install and build trtexec ( for Onnc to engine file)
    • convert yolov5 >>> onnx >>> trt/engine file
    • python3 export.py --weights yolov5s.pt --include torchscript onnx
    • install tensorRT on machine
    • export the path into bashrc file : export PATH=/usr/src/tensorrt/bin:$PATH
    • then; source ~/.bashrc
    • trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.trt

$ export PATH=/usr/local/cuda-10.2/bin:$PATH
$ export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
$ pip3 install pycuda --user
Install Pytorch with CUDA enabled:

    • sudo apt-get -y update
    • sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev;
    • export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
    • python3 -m pip install --upgrade pip; python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL

check and verify torch with cuda

    • import torch
    • print(torch.__version__)
    • print(torch.cuda.is_available())
    • print(torch.empty((1, 2), device=torch.device("cuda")))













Install torch2trt( skip it old)

    • git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    • cd torch2trt
    • python3 setup.py install --user
    • cmake -B build . && cmake --build build --target install && ldconfig



Code Convertion (.pt to .engine file )

Clone the YOLOv5 repository:
git clone https://github.com/ultralytics/yolov5.git
##############################################################################################################################

Convert the YOLOv5 model to ONNX format:
Go to the yolov5 directory:
cd yolov5
###############################################################################################################################

Convert the YOLOv5 PyTorch model to ONNX format:
#python models/export.py --weights yolov5s.pt --img-size 640 --batch 1
python export.py --weights yolov5s.pt --include torchscript onnx
###############################################################################################################################

Convert ONNX model to TensorRT engine:

Install the TensorRT package for your Jetson device. Refer to NVIDIA's documentation for the installation steps.


Convert the ONNX model to a TensorRT engine:
*****
nvidia@tegra-ubuntu:~/Desktop/Y10/yolov5$ 
source bashrc
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.trt       //trt file
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.engine    //engine file


Install tensorflow on jetson

(https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)

    • sudo apt-get update
    • sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
    • sudo apt-get install python3-pip
    • sudo python3 -m pip install --upgrade pip
    • sudo pip3 install -U testresources setuptools==65.5.0
    • sudo pip3 install -U numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.6.0
    • sudo pip3 install --upgrade –extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 tensorflow==2.11.0+nv23.01




Install Mediapipe and Bazel

(link: https://github.com/jiuqiant/mediapipe_python_aarch64/blob/main/README.md )
      
      build MediaPipe
      
    • sudo apt install -y python3-dev
    • sudo apt install -y cmake
    • sudo apt install -y protobuf-compiler
    • git clone https://github.com/google/mediapipe.git
    • git clone https://github.com/jiuqiant/mediapipe_python_aarch64.git
    • cd mediapipe
    • sed -i -e "/\"imgcodecs\"/d;/\"calib3d\"/d;/\"features2d\"/d;/\"highgui\"/d;/\"video\"/d;/\"videoio\"/d" third_party/BUILD
    • sed -i -e "/-ljpeg/d;/-lpng/d;/-ltiff/d;/-lImath/d;/-lIlmImf/d;/-lHalf/d;/-lIex/d;/-lIlmThread/d;/-lrt/d;/-ldc1394/d;/-lavcodec/d;/-lavformat/d;/-lavutil/d;/-lswscale/d;/-lavresample/d" third_party/BUILD
    • cd third_party/ nano BUILD ( add these two lines )
              
              >> "ENABLE_NEON": "OFF",
              >> "WITH_TENGINE": "OFF",
    • change ver in setup.py file with mediapipe version ( 0.8.11)
    • python3 setup.py gen_protos && python3 setup.py bdist_wheel
      
      install Mediapipe
      
    • python3 -m pip install cython
    • python3 -m pip install numpy
    • python3 -m pip install pillow
    • copy .whl to dist(create it) folder in mediapipe
    • python3 -m pip install mediapipe/dist/mediapipe-0.8-cp38-cp38-linux_aarch64.whl
      or 
    • python3 -m pip install mediapipe-python-aarch64/mediapipe-0.8.4-cp38-cp38-linux_aarch64.whl
    • pip install mediapipe==0.8.9.1
      
      check mediapipe
      
    • >> python3
    • >> import mediapipe as mp




install vnc on jetson: https://raspberry-valley.azurewebsites.net/NVIDIA-Jetson-Nano/


something in the following made it work: pip3 install --upgrade setuptools sudo pip3 install -U setuptools sudo apt-get install libpcap-dev libpq-dev sudo pip3 install cython finally!: sudo pip3 install git+https://github.com/scikit-learn/scikit-learn.git 
Disable Desktop GUI
    • sudo init 3 #temp disable Desktop GUI
    • sudo init 5 #enable Desktop GUI
      #disable GUI on every boot
    • sudo systemctl set-default multi-user.target
      #enable GUI on every boot
    • sudo systemctl set-default graphical.target
Install sklearn on Jetson

    • pip3 install --upgrade setuptools
    • sudo pip3 install -U setuptools
    • sudo apt-get install libpcap-dev libpq-dev
    • sudo pip3 install cython
    • sudo pip3 install git+https://github.com/scikit-learn/scikit-learn.git

Add more swap memory:

    • # Disable ZRAM:
    • sudo systemctl disable nvzramconfig
    • 
    • # Create 4GB swap file
    • sudo fallocate -l 4G /mnt/4GB.swap
    • sudo chmod 600 /mnt/4GB.swap
    • sudo mkswap /mnt/4GB.swap
    • 
    • # Append the following line to /etc/fstab
    • sudo su
    • echo "/mnt/4GB.swap swap swap defaults 0 0" >> /etc/fstab
    • exit
    • 
    • # Restart the Jetson!

Add uso user in virtaulbox:

    • su root
    • apt-get install sudo -y
    • adduser vboxuser sudo
    • chmod 0440 /etc/sudoers
    • reboot





install python2 pip on U20




curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py # Fetch get-pip.py for python 2.7 
python2 get-pip.py
pip --version








Integrating Companion computer and FC

      Note:-Its very important to test our code on simulator fisrt as per safety concerns.
      
    • Process to setup simulation process:
       
       install Ardupilot firmware for copter
       (https://github.com/punkypankaj/Installing-ArduPilot-directory/blob/main/docs.md)
       
    1. git clone https://github.com/ArduPilot/ardupilot.git
    2. cd ardupilot
    3. git checkout Copter-3.6
    4. git submodule update --init --recursive
       
       install python and its dependencies
       
    5. sudo apt install python-matplotlib python-serial python-wxgtk3.0 python-wxtools python-lxml python-scipy python-opencv ccache gawk python-pip python-pexpect
    6. sudo pip install future pymavlink MAVProxy
    7. gedit ~/.bashrc
    8. export PATH=$PATH:$HOME/ardupilot/Tools/autotest
    9. export PATH=/usr/lib/ccache:$PATH
    10. . ~/.bashrc
       
       run simulator:
       
    11. cd ~/ardupilot/ArduCopter
    12. sim_vehicle.py -w
       
       install Gazebo and Ardupilot plugins
       
    13.  sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
    14.  wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
    15.  sudo apt update
    16.  sudo apt install gazebo9 libgazebo9-dev
    17.  gazebo --verbose
    18.  git clone https://github.com/khancyr/ardupilot_gazebo
    19.  cd ardupilot_gazebo
    20.  mkdir build
    21.  cd build
    22.  cmake ..
    23.  make -j4
    24.  sudo make install
    25.  echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
    26.  echo 'export GAZEBO_MODEL_PATH=~/ardupilot_gazebo/models' >> ~/.bashrc
    27.  . ~/.bashrc
       
       
       
       Run Arducopter with runway in Gazebo
       
    28.  gazebo --verbose ~/ardupilot_gazebo/worlds/iris_arducopter_runway.world 
       
       Launch SITL
       
    29.  cd ~/ardupilot/ArduCopter/
    30.  sim_vehicle.py -v ArduCopter -f gazebo-iris –console
       
       Install mavproxy

    31. sudo apt-get install python3-dev python3-opencv python3-wxgtk4.0 python3-pip python3-matplotlib python3-lxml python3-pygame
    32. pip3 install PyYAML mavproxy --user
    33. echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
    34. sudo pip3 install wxPython
    35. sudo pip3 install gnureadline
    36. sudo pip3 install billiard
    37. sudo pip3 install numpy pyparsing
    38. sudo pip3 install MAVProxy
       
       Install dronekit (python3)
       
    39. sudo pip3 install dronekit



step 1.) Run simulator with arducopter (ref. point 28)

step 2.) Run SITL (ref.point 29,30)

step 3.) Run your python script with python3














