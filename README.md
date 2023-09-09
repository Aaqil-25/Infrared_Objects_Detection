# Infrared_Objects_detection
YOLOv4-based  Object Detection Model.

The model was trained with 5083 images in total.
On that 4499 images from day-time  IR images and 584 images from night-time IR images.
The model can be on tested  20% of the day-time image dataset(1124 images) and  8000 images night-time IR data set.

The model trained to detect classes as,

"category_id": 0 for  "people"

"category_id": 1 for  "buggy"

"category_id": 2 for  "motorcycle"

"category_id": 3 for  "car"

"category_id": 4 for  "ATV"

"category_id": 5 for  "bus"

"category_id": 6 for  "truck"

"category_id": 7 for  "van"

The included files are:

1. Weights
   
   The file contains the model.
   
3. Testing day IR  images
   
   The images used to test the model.
    
5. Testing night IR  images
   
   The images used to test the model. 
7. Python scripts
   
      (i)  detector_output.py
   
   	This file is used to output the detected objects in a JSON file format for day IR image
   
      (ii) detector_output_1.py
   
   	This file is used to output the detected objects in a JSON file format for night IR image


The model was tested on a Pycharm IDE.

The packages used to run the Python script.

(1) opencv-python 
	https://github.com/opencv/opencv-python

(2) NumPy
	https://github.com/numpy/numpy

(3) glob

(4) random

(5) JSON

