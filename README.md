# maskrcnn-floorplan-element-detection
Mask R-CNN–based system for automated detection and segmentation of architectural elements from floorplans.

Project Overview
This project focuses on detecting and segmenting key architectural elements from 2D floorplan images, such as walls, doors, windows, and staircases, using deep learning–based computer vision techniques. The model is built using Mask R-CNN with Detectron2, enabling precise instance segmentation for architectural analysis and downstream applications like 2D-to-3D reconstruction.

Objectives
Detect and segment architectural elements from floorplan images
Improve accuracy for complex structures like staircases
Support preprocessing for 2D-to-3D layout conversion workflows

Approach
Used Mask R-CNN with Detectron2 framework
Custom dataset preprocessing and annotation (COCO format)
Trained and evaluated models using PyTorch
Analyzed performance using detection and segmentation metrics

Tech Stack
Programming Language: Python
Deep Learning: PyTorch, Detectron2
Computer Vision: OpenCV
Data Processing: NumPy, Pandas, lxml, BeautifulSoup
Visualization: Matplotlib
Utilities: tqdm, pycocotools



Results
Achieved ~73%+ accuracy for wall, door, and window detection
Identified challenges in staircase detection and actively worked on improvements
Generated visual predictions and performance graphs

Future Improvements
Improve staircase detection accuracy
Extend model to detect room labels and furniture
Optimize model for faster inference
Integrate with full 2D-to-3D reconstruction pipeline
