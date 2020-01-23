FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update
RUN apt-get install git \
                    zip \
                    unzip \
                    ffmpeg \
                    nano

RUN pip3 install librosa \
                 pyworld \
                 matplotlib \
                 tqdm

RUN pip3 install --no-deps tensorflow-addons==0.6.0

RUN git clone https://github.com/kodamanbou/Voice-Conversion-with-TF2.0.git
WORKDIR ./Voice-Conversion-with-TF2.0

RUN mkdir outputs
RUN mkdir logdir
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true
EXPOSE 6006:6006