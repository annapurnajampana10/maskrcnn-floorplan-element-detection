from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

# Define the datasets to be registered
datasets_info = [
    ("floorplan_train", "/content/cubicasa_dataset/cubicasa5k/annotations/train.json", "/content/cubicasa_dataset/cubicasa5k"),
    ("floorplan_val",   "/content/cubicasa_dataset/cubicasa5k/annotations/val.json",   "/content/cubicasa_dataset/cubicasa5k"),
    ("floorplan_test",  "/content/cubicasa_dataset/cubicasa5k/annotations/test.json",  "/content/cubicasa_dataset/cubicasa5k")
]

for name, json_file, image_root in datasets_info:
    if name in DatasetCatalog.list():
        print(f"Dataset '{name}' is already listed. Attempting to clean and re-register.")
        # Explicitly try to remove the 'json_file' attribute from metadata if it's there
        # This is a workaround for cases where DatasetCatalog.remove() doesn't fully clear it.
        if hasattr(MetadataCatalog.get(name), "json_file"):
            delattr(MetadataCatalog.get(name), "json_file")
            print(f"  Removed 'json_file' from MetadataCatalog for '{name}'.")
        DatasetCatalog.remove(name)
        print(f"  Dataset '{name}' removed from DatasetCatalog.")

    # Now, register the dataset
    register_coco_instances(name, {}, json_file, image_root)
    print(f"Dataset '{name}' registered successfully.")

    MetadataCatalog.get(name).thing_classes = categories
    MetadataCatalog.get(name).image_root = image_root 
    print(f"Metadata for '{name}' updated with thing_classes: {MetadataCatalog.get(name).thing_classes}")
    print(f"Metadata for '{name}' json_file: {MetadataCatalog.get(name).json_file}")

print("All COCO datasets are prepared.")
