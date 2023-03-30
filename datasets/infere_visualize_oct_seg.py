'''
1- read oct data 
2- load a pretrained model
3- infere and evaluate

it loads the saved model in file model_final.pth in the default output_dir which is "output"
and will save the new prediction in the same output dir
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

'''
image data files are not copied here.
only xml files are copied and divided to train/val catgories
'''
#TODO call this from a class
def get_oct_dicts(img_dir, xml_dir):
    import sys, glob, lxml
    import xml.etree.ElementTree as ET

    dataset_dicts = []
    for idx, xml_file in enumerate(glob.iglob(os.path.join(xml_dir, '*.*'))): #TODO later iterate over png files rather than xml files, because each xml corresponds to a volume not a bscan.
        record = {}

        if not os.path.exists(xml_file):
            sys.exit("Error: xml_file does not exist!")
        filename =os.path.join(img_dir,os.path.basename(xml_file)[:-18], os.path.basename(xml_file)[:-18]+"_oct-025.png") # remove the suffix "_Surfaces_Iowa.xml" with 18 char and add image path and the suffix ".png"
        if not os.path.exists(filename):
           sys.exit("Error: filename does not exist!")
        
        ''' open xml file and extract metadata and annotations'''
        xml_tree = ET.parse(xml_file)
        xml_root = xml_tree.getroot()

        height = int(xml_root.find('scan_characteristics').find('size').find('y').text)
        width = int(xml_root.find('scan_characteristics').find('size').find('x').text)

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []        
        
        '''
          0) Inner retina = ILM-inner + OPL-outer              (20, 60)
          1) ONL = OPL-outer + ELM                             (60, 100)
          2) PR inner = ELM + IS/OS junction                   (100, 110)
          3) PR outer = IS/OS junction + PR seg-outer          (110, 120)
          4) RPE = RPE-inner + Choroid-inner                   (140, 150)
        '''
        ilm_inner = []
        opl_outer = []
        elm = []
        is_os = []
        pr_seg_outer = []
        rpe_inner = []
        ch_inner = []
        for boundary in xml_root.findall('surface'):
            if int(boundary.find('label').text) == 10:
                for y in boundary.findall('bscan')[25].findall('y'):
                    ilm_inner.append(int(y.text))
            if int(boundary.find('label').text) == 60:
                for y in boundary.findall('bscan')[25].findall('y'):
                    opl_outer.append(int(y.text))
            if int(boundary.find('label').text) == 100:
                for y in boundary.findall('bscan')[25].findall('y'):
                    elm.append(int(y.text))
            if int(boundary.find('label').text) == 110:
                for y in boundary.findall('bscan')[25].findall('y'):
                    is_os.append(int(y.text))
            if int(boundary.find('label').text) == 120:
                for y in boundary.findall('bscan')[25].findall('y'):
                    pr_seg_outer.append(int(y.text))
            if int(boundary.find('label').text) == 140:
                for y in boundary.findall('bscan')[25].findall('y'):
                    rpe_inner.append(int(y.text))
            if int(boundary.find('label').text) == 150:
                for y in boundary.findall('bscan')[25].findall('y'):
                    ch_inner.append(int(y.text)) #                    ch_inner.append(height-int(y.text))

        inner_retina = ilm_inner + opl_outer[::-1] + [ilm_inner[0]] # reverse the order of second boundary to keep the order of connecting points 
        onl = opl_outer + elm[::-1] + [opl_outer[0]]
        pr_inner = elm + is_os[::-1] + [elm[0]]
        pr_outer = is_os + pr_seg_outer[::-1] + [is_os[0]]
        rpe = rpe_inner + ch_inner[::-1] + [rpe_inner[0]]
        x1 = [i for i in range(0,width)]
        x = x1 + x1[::-1] + [x1[0]]
        layers_y = [inner_retina, onl, pr_inner, pr_outer, rpe]

        '''for each layer zip x and y in one list'''
        layers = []
        for i, l_y in enumerate(layers_y):
            l = [(x + 0.5, y + 0.5) for x, y in zip(x, l_y)]
            l = [p for x in l for p in x]
            layers.append(l)
 
        for i, l in enumerate(layers):
            obj = {
                "bbox": [0, np.min(layers_y[i]), width-1, np.max(layers_y[i])],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [l],
                "category_id": i,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# load the images
img_dir = "/projects/progstar/all_oct_imgs_progstar02/OCT_imgs_files/heyex_export_raw/renamed_vol/y49/png_dirs/" 
xml_dir = "/projects/parisa/data/test_detectron_oct_seg/"
for d in ["train", "val"]:
    DatasetCatalog.register("oct_" + d, lambda d=d: get_oct_dicts(img_dir, xml_dir + d))
    MetadataCatalog.get("oct_" + d).set(thing_classes=["Inner retina", "ONL", "PR inner", "PR outer", "RPE"])
oct_metadata = MetadataCatalog.get("oct_train")


# load the model
#from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # has five classes, one for each layer. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
#cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # to use those data with empty annotation
cfg.INPUT.FORMAT = "L" # input images are black and white
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128 , 256, 512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.002, 0.01, 0.02, 0.05]]


# inference and evaluation
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0001   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_oct_dicts(img_dir, xml_dir + "val")
for d in random.sample(dataset_dicts, 5):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    from detectron2.data import detection_utils as utils
    groundtruth_instances = utils.annotations_to_instances(d['annotations'], (d['height'],d['height']))
    v_pred = Visualizer(im[:, :, ::-1],
                   metadata=oct_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # This option is only available for segmentation models
    )
    v_groundtruth = Visualizer(im[:, :, ::-1],
                   metadata=oct_metadata,
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # This option is only available for segmentation models
    )
    out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_groundtruth = v_groundtruth.draw_dataset_dict(d)

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(1, 2, figsize=(20, 10))
    axis[0].imshow(out_pred.get_image()[:, :, ::-1])
    axis[1].imshow(out_groundtruth.get_image()[:, :, ::-1])
    axis[0].set_title('Predicition')
    axis[1].set_title('Ground Truth')
    plt.tight_layout()
    plt.savefig("./output/"+os.path.basename(d["file_name"]))     
    
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("oct_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "oct_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
