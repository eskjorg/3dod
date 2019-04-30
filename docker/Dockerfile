FROM nvcr.io/nvidia/pytorch:19.04-py3

RUN conda update -n base -c conda-forge conda
RUN conda install -y -c conda-forge \
    attrdict \
    opencv \
    tensorboardX
RUN pip install git+https://github.com/facebookresearch/maskrcnn-benchmark.git@master
RUN pip install nuscenes-devkit