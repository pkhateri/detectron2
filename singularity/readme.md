This folder contains the singularity def files and their container sif file.
1- detectron2_built_from_library_on_gcloud.sif
        - is built on the gcloud based on a docker image from my dockerfile
2- detectron2.sif
        - is built from the def file here based on ubuntu20.04 docker image
        - tested. works fine with gpu utilization.
3- detectron2_ngc_cuda.sif
        - is built from the def file here based on ngc cuda docker image
        - tested. works fine with gpu utilization.
4- detectron2_ngc_torch.sif
        - is built from the def file here based on ngc torch docker image
        - tested. works fine with gpu utilization.


Use detectron.sif    ->   read below why:

- three different singularity containers: 
	1) based on plain Ubuntu
	2) based on ngc torch
	3) based on ngc cuda.
- All three containers utilize the gpu in the same way. Because Emanuel has already installed GPU on the WS. The reason that the nvcc command is not found is because it is a sdk command. Emanuel has installed only a run version of cuda.

