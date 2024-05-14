# set base image (host OS)
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update -y \
    && apt-get install -y python3-pip \
    && apt-get install -y git \
    && apt-get install -y wget

# set the working directory in the container
WORKDIR /vqaglob

RUN cd lmms-eval
RUN pip install -e .
RUN pip install -r llava_repr_requirements.txt

RUN cd ..

# RUN mkdir -p ~/miniconda3 \
#     && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
#     && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
#     && rm -rf ~/miniconda3/miniconda.sh

# RUN conda init bash
