# Infrared_Objects_detection
YOLOv4-based  Object Detection Model.

The model was trained with 5083 images in total.
On that 4499 images from day-time  IR images and 584 images from knight-time IR images.
The model tested on 20% of the day-time image dataset(1124 images). From the Knight-time dataset, 8000 images are to be available for testing.


The model trained to detect classes as,

"category_id": 0 for  "people",
"category_id": 1 for  "buggy",
"category_id": 2 for  "motorcycle"
"category_id": 3 for  "car"
"category_id": 4 for  "ATV"
"category_id": 5 for  "bus"
"category_id": 6 for  "truck"
"category_id": 7 for  "van"

The included files are:

1. Weights
2. Testing day ir  images
3. Testing knight ir  images
4. python scripts
 	(
	detector_output.py,
	detector_output_1.py)


1. weights 

The file containe the model.

2.Testing images

The images used to test the model. 

3. Testing knight ir images

The images used to test the model.

4.python script to test the model.

detector_output.py,

this file used to output the detected objects in a json file format for day ir image

detector_output_1.py,

this file used to output the detected objects in a json file format for knight ir image


The model tested on a pycharm IDE.

The packages used to run the python script.

opencv-python 
https://github.com/opencv/opencv-python

numpy

glob

random

json

re
