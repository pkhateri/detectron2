# install on local computer
## install python-dev 3.8 before virtual environment
`sudo apt-get install python3.8-dev`

## make a virtual environment and activate it
```
virtualenv --python=3.8 venv
source venv/bin/activate
```

## install prerequisites
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install opencv-python
pip install 'git+https://github.com/facebookresearch/fvcore'
```

## install detectron
`pip install .`

## test the software
```
wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
mkdir output
python3 demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input.jpg --output outputs/ --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
AssertionError: Torch not compiled with CUDA enabled

# install on google VM
## install python-dev 3.8 before virtual environment
`sudo apt-get install python3.8-dev`

## make a virtual environment and activate it
```
virtualenv --python=3.8 venv
source venv/bin/activate
```

## install prerequisites
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install opencv-python
pip install 'git+https://github.com/facebookresearch/fvcore'
```
## install detectron
pip install .

## test the software
wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
mkdir output
python3 demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input.jpg --output outputs/ --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
### you get an image in the outputs folder with the boxes detecting the objects in the image

# build docker image on my local computer
not enough space :(

# build and run docker container on VM google cloud:
## installl docker
## install nvidia-container-toolkit -> Setting up NVIDIA Container Toolkit(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
## build and run docker container on VM:
```
sudo docker build --build-arg USER_ID=$UID -t detectron2:v0 .
sudo docker run --gpus all -it   --shm-size=8gb --env="DISPLAY" \
                                -v "/home/parisa_khateri_iob_ch/git_software/detectron2:/home/appuser/detectron2_repo" \
                                --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  detectron2:v0
```
### Note 1:
`sudo docker build -t detectron2:v0 .`
- ERROR: Package 'detectron2' requires a different Python: 3.6.9 not in '>=3.7'
- SOlUTION: change the dockerfile: Upgrade ubuntu version from 18.04 to 20.4:
		`FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 # 18.04 to 20.04`
		...
		`RUN wget https://bootstrap.pypa.io/get-pip.py && \ # do not specify pip version`
		...
### Note 2:
`sudo docker run --gpus all -it   --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"   --name=detectron2 detectron2:v0`
- PROBLEM: docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
- SOLUTION: you need to install nvidia-container-toolkit -> Setting up NVIDIA Container Toolkit(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Note 3:
`sudo docker run --gpus all -it   --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"   --name=detectron2 detectron2:v0`
- PROBLEM: when inside docker and run the python program -> ModuleNotFoundError: No module named 'tqdm'
- SOLUTION: you need to specify the userid as in the docker file a userid is used to run as not-root user
		`sudo docker build --build-arg USER_ID=$UID -t detectron2:v0 .`
### Note 4:
- PROBLEM: run python code inside the container and get output, but the output is gone when out of container.
- SOLUTION: you need to mount the local path to a path inside container using -v flag:
		```
		sudo docker run --gpus all -it   --shm-size=8gb --env="DISPLAY" \
				-v "/home/parisa_khateri_iob_ch/git_software/detectron2:/home/appuser/detectron2_repo" \
				--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  detectron2:v0
		```
-------------------------------------------------
# build and run sinularity container from singularity def file from scratch in VM:
`sudo singularity build detectron2.sif singularity/detectron2.def`

## Note:
- building the sinularity container based on the docker image did not work because of the user definition in the dockerfile which is non-root. I couldnot find a way to mimic this non-root user for singu container.

## run the singularity container
`singularity run --nv docker/detectron2.sif`

## run detectron2 inside the container:
`python3 demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input.jpg --output outputs/ --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl`


# build and run singularity container from singularity def file from ngc pytorch container:
check [nvcr webpage](nvcr.io/nvidia/pytorch:xx.xx-py3)

# run a simple training on existing dataset
## download the baloon dataset
`wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip `
## extract it to data folder
`unzip balloon_dataset`
## run data preparation, training and evaluation
`python3 datasets/prepare_ballon_dataset_train_simple_model_inference.py`
-------------------------------------------------

# use for box annotation
1. you need to have json files for train and val data. create them from box_annotation excel files using this code. it chooses randomly from the files for train (k%) and val (n-k%)/2 and test (n-k%)/2.
```
python convert_box_annot_csv_to_json.py -c <csv_filename> -d <json_dir> -j <json_prefix> -n <number_of_rows> -k <train_percentage>

example for multiple boxes:
csv_filename = '/projects/parisa/data/progstar/box_annot/box_annotation_final.csv'
json_dir = '/projects/parisa/data/progstar/box_annot/'
json_prefix = 'box_annotation'
example for single box:
csv_filename = '/usr/local/scratch/parisa/data/test_detectron/single_box_annotation_final.csv'
json_dir = '/usr/local/scratch/parisa/data/test_detectron/'
json_prefix = 'single_box_annotation'

```
2. run the singularity container
```
singularity run --nv -B /projects docker/detectron2.sif
```
3. run the training
```
python3 datasets/prepare_oct_dataset_box_train_simple_model_inference.py
```
4. see loss plots using tensorboard:
```
tensorboard --logdir output/
```
	- specify the port if remote on miac:
```
tensorboard --logdir output/ --port 6006
```
