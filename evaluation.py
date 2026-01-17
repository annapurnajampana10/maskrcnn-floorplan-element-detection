from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

# --- Load config and trained weights 
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  
cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/floorplan_detection_output/model_final.pth"  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   
cfg.DATASETS.TEST = ("floorplan_val",) 

cfg.MODEL.DEVICE = 'cpu' 

# Create a predictor to load the model
predictor = DefaultPredictor(cfg)

# --- Create evaluator for your dataset ---
evaluator = COCOEvaluator("floorplan_val", cfg, False, output_dir="./output/")

# --- Build test loader ---
val_loader = build_detection_test_loader(cfg, "floorplan_val")

# --- Run evaluation ---
results = inference_on_dataset(predictor.model, val_loader, evaluator)
print(results)
