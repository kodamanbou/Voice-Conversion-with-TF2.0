FROM nvidia/cuda:10.0-base-ubuntu18.04

RUN apt-get update && apt-get upgrade -y
RUN apt-get install git \
                    zip \
                    unzip \
                    ffmpeg \
                    nano

RUN pip3 install librosa \
                 pyworld \
                 matplotlib \
                 tqdm \
                 tensorflow-gpu==2.0.0 \
                 tensorflow-addons==0.6.0

RUN git clone https://github.com/kodamanbou/Voice-Conversion-with-TF2.0.git
WORKDIR ./Voice-Conversion-with-TF2.0

COPY datasets.zip .
RUN unzip -d datasets datasets.zip
RUN mkdir outputs
RUN mkdir logdir
EXPOSE 6006:6006