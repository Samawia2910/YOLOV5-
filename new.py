# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
#
# # Define the class to detect
# class_to_detect = 'person'
#
# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#
# # Define the image transformation
# transform = transforms.Compose([
#     transforms.Resize(size=640),
#     transforms.ToTensor()
# ])
#
# # Load the image
# img = r'E:\Project\yolov5_sample\yolov5_sample\images\new1.jpg'
# # img = Image.open('')
#
# # Apply the transformation to the image
# img = transform(img)
#
# # Run the detection on the image
# results = model(img.unsqueeze(0))
#
# # Filter the results to only show the desired class
# class_indices = np.where(results.pred[0].detach().cpu().numpy()[:, -1] == class_to_detect)[0]
# class_results = results.pred[0].detach().cpu().numpy()[class_indices]
#
# # Print the results
# print(class_results)
# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# # Set up YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#
# # Define the class you want to detect
# class_to_detect = 'person'
#
# # Load image
# image_path = Path(r'E:\Project\yolov5_sample\yolov5_sample\images\new1.jpg')
# image = cv2.imread(str(image_path))
#
# # Run YOLOv5 detection
# results = model(image)
#
# # Extract detected objects and their labels
# objects = results.pred[0]
# labels = objects[:, -1]
#
# # Filter objects to only include the class you want to detect
# class_indices = labels == class_to_detect
# filtered_objects = objects[class_indices]
#
# # Draw bounding boxes around the detected objects
# for obj in filtered_objects:
#     bbox = obj[:4].tolist()
#     confidence = obj[4]
#     label = obj[-1]
#     color = (0, 255, 0)  # green
#     thickness = 2
#     cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color, thickness)
#     cv2.putText(image, f'{label}: {confidence:.2f}', tuple(bbox[:2]), cv2.FONT_HERSHEY_PLAIN, 1, color, thickness)
#
# # Show the result
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

import cv2
import torch
import torch as torch

from models.experimental import attempt_load
# from utils.general import non_max_suppression, scale_coords, plot_one_box
model = attempt_load("yolov5s.pt", map_location=torch.device("cpu"))
class_label = "person"
img = cv2.imread(r'E:\Project\yolov5_sample\yolov5_sample\images\new1.jpg')
# img = cv2.resize(img, (640, 640))
# results = model(torch.from_numpy(img).float()[None], augment=False)
# boxes = non_max_suppression(results.pred[0], conf_thres=0.5, iou_thres=0.5, classes=[model.names.index(class_label)])
# for box in boxes:
#     box = scale_coords(img.shape[:2], box[:, :4], img.shape[:2]).round()
#     for x1, y1, x2, y2, conf, cls in box:
#         plot_one_box((x1, y1, x2, y2), img, label=model.names[int(cls)], color=(0, 255, 0))
# cv2.imshow("Detection result", img)
# cv2.waitKey(0)