import os
import sys
py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.add_dll_directory(py_dll_path)
import numpy as np
import cv2
from PIL import Image, ImageDraw
from matplotlib import cm
from scipy import ndimage
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class KjnFastRCNN(object):
    def __init__(self, model_path='object_detection_and_tracing/models/model_OD.pth'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = self.get_model_object_detection(num_classes = 2)
        if self.model_path is not None:
            self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        self.model.eval()

    def get_model_object_detection(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def predict(self, img, iou_threshold=0.3):
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img = torchvision.transforms.functional.to_tensor(img)
        img = img.to(self.device)
        with torch.no_grad():
            prediction = self.model([img])
        nms_results = torchvision.ops.nms(boxes = prediction[0]['boxes'], scores = prediction[0]['scores'], iou_threshold= iou_threshold)
        nms_results = nms_results.tolist()
        boxes = prediction[0]['boxes'].cpu().detach().numpy()
        labels = prediction[0]['labels'].cpu().detach().numpy()
        scores = prediction[0]['scores'].cpu().detach().numpy()
        list_of_bbox_with_labels = []
        i = 0
        for idx, label in enumerate(labels):
            if idx not in nms_results:
                continue
            box = boxes[i]
            box = box.astype(int)
            bbox_dict = {
                'top_left': (box[0], box[1]),
                'top_right': (box[2], box[1]),
                'botom_right': (box[2], box[3]),
                'botom_left': (box[0], box[3]),
                'label': label,
                'probability': scores[i]
            }
            list_of_bbox_with_labels.append(bbox_dict)
            i += 1
        return list_of_bbox_with_labels

    def detect(self, img, iou_threshold=0.7):
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img = torchvision.transforms.functional.to_tensor(img)
        img = img.to(self.device)
        with torch.no_grad():
            prediction = self.model([img])
        nms_results = torchvision.ops.nms(boxes = prediction[0]['boxes'], scores = prediction[0]['scores'], iou_threshold= iou_threshold)
        nms_results = nms_results.tolist()
        boxes = prediction[0]['boxes'].cpu().detach().numpy()
        labels = prediction[0]['labels'].cpu().detach().numpy()
        scores = prediction[0]['scores'].cpu().detach().numpy()
        i = 0
        bbox_xcycwh, cls_conf, cls_ids = [], [], []
        for idx, label in enumerate(labels):
            if idx not in nms_results:
                continue
            box = boxes[i]
            score = scores[i]
            box = box.astype(int)
            x0, y0, x1, y1 = box
            bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
            cls_conf.append(score)
            cls_ids.append(label)
            i += 1
        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids)

    def predict_with_draw(self, img, iou_threshold=0.5):
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img = torchvision.transforms.functional.to_tensor(img)
        img = img.to(self.device)
        with torch.no_grad():
            prediction = self.model([img])
        nms_results = torchvision.ops.nms(boxes = prediction[0]['boxes'], scores = prediction[0]['scores'], iou_threshold= iou_threshold)
        nms_results = nms_results.tolist()
        img = img.cpu().detach()
        img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        boxes = prediction[0]['boxes'].cpu().detach().numpy()
        labels = prediction[0]['labels'].cpu().detach().numpy()
        scores = prediction[0]['scores'].cpu().detach().numpy()
        list_of_bbox_with_labels = []
        i = 0
        for idx, label in enumerate(labels):
            if idx not in nms_results:
                continue
            box = boxes[i]
            box = box.astype(int)
            bbox_dict = {
                'top_left': (box[0], box[1]),
                'top_right': (box[2], box[1]),
                'botom_right': (box[2], box[3]),
                'botom_left': (box[0], box[3]),
                'label': label,
                'probability': scores[i]
            }
            draw = ImageDraw.Draw(img)
            draw.rectangle(box.tolist(), outline ="green", width=10)
            list_of_bbox_with_labels.append(bbox_dict)
            i += 1
        img = np.asarray(img)
        return list_of_bbox_with_labels, img

if __name__ == "__main__":
    def load_images(path):
        images = []
        valid_images = ['.png', '.jpg', '.jpeg']
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            images.append(os.path.join(path, f))
        return images

    import pprint
    kjn = KjnFastRCNN()
    for idx, image_path in enumerate(sorted(load_images('E:/kjn_biedronka/biedronka_img_dataset3/'))):
        print("image_path: ", image_path)
        image = cv2.imread(image_path)
        bbox_xcycwh, cls_conf, cls_ids = kjn.detect(image)
        print(bbox_xcycwh)
