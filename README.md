# Object_detection-YOLO_V3
This repository contains an implementation of the YOLOv3 object detection model using Keras and TensorFlow. The model is trained on the COCO dataset and can detect 80 different classes of objects.
## Model Architecture
The YOLOv3 model architecture consists of a backbone convolutional neural network (Darknet-53) followed by detection layers. The backbone network extracts features from the input image, and the detection layers predict bounding boxes and class probabilities for multiple anchor boxes at 3 different scales.
## Usage
To use the YOLOv3 model for object detection, follow these steps:
1. Install the required dependencies.
2. Download the pre-trained weights file for the YOLOv3 model from [here](https://pjreddie.com/media/files/yolov3.weights) and place it in the project directory.
3. Run the `yolo_algo.ipynb` script to save your model in `model.h5`
4. Prepare the input image for object detection and placed in the `input_images` directory.
5. Run the `yolov3_out.ipynb` script to perform object detection on the input images:

    The script will process the images using the YOLOv3 model and save the annotated images with bounding boxes in the `output_images` directory.

6. The annotated images can be found in the `output_images` directory, showing the detected objects with their corresponding labels and confidence scores.
## Configuration

The YOLOv3 model can be configured using the following parameters:

- Input size: The width and height of the input images used in this model is 416*416.

- Confidence threshold: The minimum confidence score required for an object detection result to be considered valid can be adjusted by modifying the `class_threshold` variable. In this its value is `0.6`.

- Non-maximum suppression (NMS) threshold: The overlap threshold used for suppressing overlapping bounding boxes can be adjusted by modifying the `do_nms` function. In this its value taken is 0.5.
## Acknowledgments
This project is based on the YOLOv3 algorithm developed by Joseph Redmon and Ali Farhadi. The implementation used here was inspired by various open-source projects and tutorials.
