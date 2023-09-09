import glob
import cv2
import numpy as np
import json
import re

# Load Yolo

net = cv2.dnn.readNet("weights/yolov4-tiny-custom_training_last.weights", "weights/yolov4-tiny-custom.cfg")
# Name custom objects
classes = ["people", "buggy", "motorcycle", "car", "ATV", "bus", "truck", "van"]

# Images path

images_path = sorted(glob.glob(r"Testing_1/*.png"), key=lambda path: int(''.join(filter(str.isdigit, path))))
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
num_classes = len(classes)

colors = np.random.uniform(0, 255, size=(num_classes, 3))

# Initialize dictionary to store results
results_dict = {}

count = 0
tracked_objects = {}  # dictionary to store tracking IDs of detected objects
image_id = []

# loop through all the images
for img_path in images_path:
    # Loading image
    image_id = re.findall(r"\((\d+)\)", img_path)[0]
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = float(detection[2] * width)
                h = float(detection[3] * height)

                # Rectangle coordinates
                x = float(center_x - w / 2)
                y = float(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Initialize list to store objects detected in image
    objects_list = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            label = str(classes[class_id])
            confidence = confidences[i]
            # check if the object has appeared before
            object_key = (label, tuple(img[int(y):int(y + h), int(x):int(x + w)].flatten()))
            if object_key in tracked_objects:
                # if yes, reuse its previous tracking ID
                tracking_id = tracked_objects[object_key]
                count += 1
            else:
                # if not, assign a new tracking ID and store it in the dictionary
                count += 1
                tracking_id = count
                tracked_objects[object_key] = tracking_id

            # Store object information in dictionary
            object_tracked = {
                "id": count,
                "image_id": int(image_id),
                "track_id": tracking_id,
                "bbox": [x, y, w, h],
                "score": confidence,
                "category_id": int(class_id)
            }

            objects_list.append(object_tracked)

    # Store objects list in results dictionary
    results_dict[image_id] = objects_list

# Save results to JSON file
with open("results1.json", "w") as outfile:
    json.dump(results_dict, outfile, indent=4)

cv2.destroyAllWindows()
