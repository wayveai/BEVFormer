FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key del 7fa2af80 \
  && rm /etc/apt/sources.list.d/cuda.list \
  && apt-get -y update \
  && apt-get install -y \
        curl \
        wget \
        git \
        zip \
        build-essential \
        software-properties-common \
        libomp5 \
        # required for torchvision
        libpng-dev \
        zlib1g-dev \
        libjpeg-dev  \
        ffmpeg \
        libsm6 \
        libxext6 \
   # Clean up
   && apt-get autoremove -y \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*


# gcc 9
RUN apt-get update \
  && apt-get -y install \
    software-properties-common \
  && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
  && apt-get -y update \
  && apt-get -y install \
    gcc-9 g++-9 libstdc++-9-dev \
  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 50 \
  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100 \
  && update-alternatives --set gcc /usr/bin/gcc-9 \
  && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 50 \
  && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100 \
  && update-alternatives --set g++ /usr/bin/g++-9 \
  # Clean up
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa \
  && apt-get update \
  && apt install -y \
        python3.8 \
        python3.8-dev \
        python3-pip \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
  && python3 -m pip install --upgrade pip \
  && ln -s /usr/bin/python3 /usr/bin/python


RUN python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install mmdet==2.14.0 mmsegmentation==0.14.1 timm
RUN python -m pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html

RUN git clone https://github.com/open-mmlab/mmdetection3d.git \
    && cd mmdetection3d \
    && git checkout v0.17.1 \
    # && pip install -v -e . \
    && python3 -m pip install -r requirements/runtime.txt \
    && python3 -m pip install numpy cython pythran \
    && python3 setup.py develop \
    && cd ..

RUN python -m pip install --upgrade transforms3d numpy
RUN apt-get install -y python3.8-tk

# RUN cp -r /mmdetection3d/mmdet3d /usr/local/lib/python3.8/dist-packages/
ENV PYTHONPATH "${PYTHONPATH}:/mmdetection3d"
CMD CUDA_VISIBLE_DEVICES=6,7 /BEVFormer/tools/dist_train.sh /BEVFormer/projects/configs/bevformer/bevformer_small.py 2
