'''
1- prepare oct dataset 
2- fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the oct dataset
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

def get_oct_dicts(img_dir):
    json_file = os.path.join(img_dir, "x_boxes.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for row in imgs_anns:
        record = {}
        
        filename = os.path.join(img_dir, row["file_name"])
        x_box = row["x_box"]
        obj = {
            "bbox": [x_box[0], 0, x_box[1], 0],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
        }
                
        record["file_name"] = filename
        record["annotations"] = [obj]
        dataset_dicts.append(record)
    return dataset_dicts

dir = "/projects/parisa/data/test_detectron_oct/"
for d in ["train", "val"]:
    DatasetCatalog.register("oct_" + d, lambda d=d: get_oct_dicts(dir + d))
    MetadataCatalog.get("oct_" + d).set(thing_classes=["damaged_retina"])
balloon_metadata = MetadataCatalog.get("oct_train")


# training
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) #https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
cfg.DATASETS.TRAIN = ("oct_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (damaged_retina). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# inference and evaluation
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_balloon_dicts(dir+"val")
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    import matplotlib.pyplot as plt
    import os
    #cv2.imshow(out.get_image()[:, :, ::-1])
    plt.figure()
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.savefig("./output/"+os.path.basename(d["file_name"])) 
    
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("balloon_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "balloon_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
