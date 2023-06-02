import cv2
import numpy as np
import torch
from utils.datasets import letterbox
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from numpy import random

# from models.experimental import attempt_load
# from utils.general import non_max_suppression, scale_coords, plot_one_box


def detect(img0, model, imgsz,names):


    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16



    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


    img = letterbox(img0, 640, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    lbls=[]
    boxes=[]
    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):

                label = f'{names[int(cls)]} {conf:.2f}'
                # print(label)
                lbl= label.split(' ')[0]
                if lbl == 'person':
                    lbls.append(lbl)
                    x1 = int(xyxy[0].item())
                    y1 = int(xyxy[1].item())
                    x2 = int(xyxy[2].item())
                    y2 = int(xyxy[3].item())
                    boxes.append((x1, y1, x2, y2))
    return lbls, boxes
if __name__ == '__main__':
    device=''
    classes=None
    agnostic_nms=False
    conf_thres=0.5
    iou_thres=0.45
    augment=False
    imgsz=640
    weights='weights/yolov5s.pt'
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model



    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
   

    image_path = r'E:\Project\yolov5_sample\yolov5_sample\images\new1.jpg'
    img0 = cv2.imread(image_path)
    fresh_img = img0.copy()
    tl = round(0.002 * (img0.shape[0] + img0.shape[1]) / 2) + 1
    labels, boxes = detect(img0, model, imgsz, names)
    print("Boxes", boxes)
    i = 0
    for box in boxes:
        print(box)
        crop_obj = fresh_img[box[1]:box[3], box[0]:box[2]]
        cv2.imwrite(f'crop/obj_{i}.jpg', crop_obj)
        i+= 1
    for x1, y1, x2, y2 in boxes:
        # Crop the image using the bounding box coordinates
        crop_img = fresh_img[int(y1):int(y2), int(x1):int(x2)]

        # Save the cropped image as a separate file
        cv2.imwrite(f'crop/obj_{i}.jpg', crop_img)
    for box in boxes:
       indexing = boxes.index(box)
       label = labels[indexing]
       c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
       try:
        label_id = names.index(label)
        color = colors[label_id]
        cv2.rectangle(fresh_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(fresh_img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(fresh_img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)       # cv2.putText(fresh_img, "No of Persons: " + str(len(boxes)), (28, 50), 0, 2, (100, 20, 0), 3)
        cv2.putText(fresh_img, "No of Persons: " + str(len(boxes)), (28, 50), 0, 2, (100, 20, 0), 3)
        # print("persons in the image : ", len(boxes))
        # cv2.putText(fresh_img, f'{label}: {conf_thres:.2f}', tuple(boxes[:2]), cv2.FONT_HERSHEY_PLAIN, 1, color, thickness=tf, lineType=cv2.LINE_AA)

       except:
        color = [0, 0, 255]
        cv2.rectangle(fresh_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(fresh_img, c1, c2, color, -1, cv2.LINE_AA)
        # cv2.putText(fresh_img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(fresh_img, "No of Persons detect" + str(boxes.shape[0]), (0, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
        cv2.putText(fresh_img, "No of Person: " + str(len(boxes)), (28, 50), 0, 2, (100, 20, 0), 3)
        # print("persons in the image : ", len(boxes))
        # person_results = results.pandas().xyxy[results.pandas().xyxy['name'] == 'person']
        # cv2.putText(fresh_img, f'{label}: {conf_thres:.2f}', tuple(boxes[:2]), cv2.FONT_HERSHEY_PLAIN, 1, color, thickness=tf, lineType=cv2.LINE_AA)
        # print(person_results)
        # cv2.imshow('Inference', fresh_img)

    cv2.imshow('Image', fresh_img)
    cv2.waitKey()
