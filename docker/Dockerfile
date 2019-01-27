FROM nvcr.io/nvidia/pytorch:19.01-py3

RUN conda update -n base -c defaults conda
RUN conda install -y -c conda-forge \
    attrdict \
    opencv \
    tensorboardX

RUN pip install git+https://github.com/facebookresearch/maskrcnn-benchmark.git@master
