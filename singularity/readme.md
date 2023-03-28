Use detectron.sif
read below why.

- three different singularity containers: 
	1) based on plain Ubuntu
	2) based on ngc torch
	3) based on ngc cuda.
- All three containers utilize the gpu in the same way. Because Emanuel has already installed GPU on the WS. The reason that the nvcc command is not found is because it is a sdk command. Emanuel has installed only a run version of cuda.

