'''
1- prepare oct dataset 
2- fine-tune a COCO-pretrained faster_rcnn_R_50_FPN_3x on the oct dataset
3- infere and evaluate
'''

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



# data preparation
from detectron2.structures import BoxMode

#{"file_name": "11006_OD_20140307_21h30m28s_y49_z496_x1024_oct-026.png", "boxes": {"0": {"x_box": [234, 740], "y_box": [99, 170]}, "1": {"x_box": [777, 806], "y_box": [90, 168]}}},
def get_oct_dicts(img_dir):
    json_file = os.path.join(img_dir, "x_boxes.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for idx, value in enumerate(imgs_anns):        
        filename = os.path.join(img_dir, value["file_name"])
        height = int(value["file_name"].split('_')[5][1:])
        width = int(value["file_name"].split('_')[6][1:])
        
        record = {}
        objs = []
        for _, box in value["boxes"].items():    
            x_box = box["x_box"]
            y_box = box["y_box"]
            obj = {
                "bbox": [x_box[0], y_box[0], x_box[1], y_box[1]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objs.append(obj)
        
        record["file_name"] = filename
        record["image_id"] = idx       
        record["height"] = height
        record["width"] = width
        record["annotations"] = [obj]
        if x_box[0]==x_box[1]==0:
            record["annotations"] = []
        dataset_dicts.append(record)
    return dataset_dicts

dir = "/projects/parisa/data/test_detectron_oct/"
for d in ["train", "val"]:
    DatasetCatalog.register("oct_" + d, lambda d=d: get_oct_dicts(dir + d))
    MetadataCatalog.get("oct_" + d).set(thing_classes=["damaged_retina"])
oct_metadata = MetadataCatalog.get("oct_train")


# training
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("oct_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (damaged_retina). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # to use those data with empty annotation

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
# inference and evaluation
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_oct_dicts(dir+"val")
for d in random.sample(dataset_dicts, 20):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    from detectron2.data import detection_utils as utils
    groundtruth_instances = utils.annotations_to_instances(d['annotations'], (d['height'],d['height']))
    v_pred = Visualizer(im[:, :, ::-1],
                   metadata=oct_metadata, 
                   scale=2#, 
                   #instance_mode=ColorMode.IMAGE_BW   # This option is only available for segmentation models
    )
    v_groundtruth = Visualizer(im[:, :, ::-1],
                   metadata=oct_metadata,
                   scale=2#, 
                   #instance_mode=ColorMode.IMAGE_BW   # This option is only available for segmentation models
    )
    out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_groundtruth = v_groundtruth.draw_dataset_dict(d)
    import matplotlib.pyplot as plt
    import os
    figure, axis = plt.subplots(1, 2, figsize=(20, 10))
    axis[0].imshow(out_pred.get_image()[:, :, ::-1])
    axis[1].imshow(out_groundtruth.get_image()[:, :, ::-1])
    plt.savefig("./output/"+os.path.basename(d["file_name"]))     
    
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("oct_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "oct_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
