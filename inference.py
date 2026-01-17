import os, cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo

# --- Load config and trained weights ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Changed to 4 to match the trained model's class count
cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/floorplan_detection_output/model_final.pth"  # Updated path to your trained checkpoint
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("floorplan_val",)

# Configure to use CPU
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

# --- Get validation dataset ---
dataset_dicts = DatasetCatalog.get("floorplan_val")
metadata = MetadataCatalog.get("floorplan_train")

# --- Output directory ---
output_dir = "/content/inference_val_results"
os.makedirs(output_dir, exist_ok=True)

# --- Run inference on all validation images ---
for i, d in enumerate(dataset_dicts):
    img_path = d["file_name"]
    img = cv2.imread(img_path)

    # Handle cases where image might not be loaded correctly
    if img is None:
        print(f"Warning: Could not load image from {img_path}. Skipping.")
        continue

    outputs = predictor(img)

    # Visualize predictions
    v = Visualizer(img[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save result
    # Ensure unique filenames for saving
    base_name = os.path.basename(img_path)
    file_name_without_ext, file_ext = os.path.splitext(base_name)
    save_path = os.path.join(output_dir, f"{file_name_without_ext}_pred{file_ext}")
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

    # Optionally show inline in Colab, limit to first few images to avoid excessive output
    if i < 5: # Show only first 5 images for brevity
        plt.figure(figsize=(12, 12))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.axis("off")
        plt.title(os.path.basename(img_path))
        plt.show()
    else:
        print(f"Processed and saved: {save_path}")

print(f"Inference complete. Results saved to {output_dir}")
