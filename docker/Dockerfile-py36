FROM nvcr.io/nvidia/pytorch:19.01-py3

RUN apt-get update

# For OpenCV
RUN apt-get install -y libgl1-mesa-glx

RUN conda update -n base -c defaults conda
RUN conda install -y -c conda-forge \
    attrdict \
    opencv \
    tensorboardX
# tensorboard server (if desired to run in container)
RUN conda install -y -c conda-forge tensorflow tensorboard

RUN pip install git+https://github.com/eskjorg/maskrcnn-benchmark.git@master
