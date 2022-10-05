#FROM python:3.8
FROM nvidia/cuda:11.6.0-base-ubuntu20.04
ARG CUDA=cu114
RUN ["apt-get", "update"]
RUN ["apt", "install", "python3", "-y"]
RUN ["apt", "install", "python3-pip", "-y"]
RUN ["pip3", "install", "--upgrade", "pip"]

COPY requirements.txt requirements.txt
COPY requirements_cuda.txt requirements_cuda.txt
RUN ["pip3", "install", "-r", "requirements.txt"]
RUN ["pip3", "install", "-r", "requirements_cuda.txt", "--extra-index-url", "https://download.pytorch.org/whl/cu116"]

RUN ["mkdir", "/workspace" ]
#COPY . workspace
#COPY train.py train/train.py
#COPY classcount.py train/classcount.py
#COPY diceloss.py train/diceloss.py
#COPY distribution.py train/distribution.py
#COPY eval.py train/eval.py
#COPY utils train/utils
#COPY predict.py train/predict.py
#COPY unet train/unet

#COPY data data

WORKDIR "/workspace"


CMD ["bash"]