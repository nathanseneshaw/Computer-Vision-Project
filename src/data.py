"""
RDD2022 pothole dataset for the from-scratch YOLO (v1) detector
Uses RDD2022 road images, keeping only potholes (D40)
"""
import glob
import os
import random
import xml.etree.ElementTree as ET

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

# <repo>/data/RDD2022, filled by download_rdd2022.py
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'RDD2022')

# RDD2022 damage codes to detect. D40 is pothole
# (D00 longitudinal crack, D10 transverse crack, D20 alligator crack)
# Listing more than one code here merges them into the model's single class
TARGET_CLASSES = ('D40',)


# read a Pascal VOC xml annotation from RDD2022
# returns an (N, 4) float32 array of target boxes as (x1, y1, x2, y2) in original image pixels
def load_annotation(xml_path, target_classes=TARGET_CLASSES):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.iter('object'):
        if obj.findtext('name', '').strip() not in target_classes:
            continue
        bndbox = obj.find('bndbox')
        boxes.append([float(bndbox.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


# RDD2022 keeps <country>/train/annotations/xmls/X.xml next to <country>/train/images/X.jpg
def image_path(xml_path):
    train_dir = os.path.dirname(os.path.dirname(os.path.dirname(xml_path)))
    name = os.path.splitext(os.path.basename(xml_path))[0] + '.jpg'
    return os.path.join(train_dir, 'images', name)


# The dataset class
class RoadDamage(data.Dataset):
    # countries: optional list such as ['Japan', 'India'] to train on part of RDD2022
    def __init__(self, image_set='train', data_path=DATA_PATH, countries=None, val_fraction=0.1):

        self.name = 'rdd2022_' + image_set
        self.image_set = image_set
        self.data_path = os.path.normpath(data_path)
        self.countries = countries
        self.val_fraction = val_fraction
        self.classes = ('__background__', 'pothole')
        # RDD2022 image sizes vary by country, so there is no fixed width/height here:
        # compute the scale to 448 x 448 from each image's own shape in __getitem__
        self.yolo_image_size = 448
        self.yolo_grid_num = 7
        self.yolo_grid_size = self.yolo_image_size / self.yolo_grid_num
        # split images into training set and validation set
        self.gt_files_train, self.gt_files_val = self.list_dataset()
        # the pixel mean for normalization
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        # training set
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            # validation set
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)


    # list the annotation files that contain at least one pothole
    # RDD2022's test split has no annotations, so the labelled train split is divided
    # into train and val with a fixed seed (val_fraction of the images go to val)
    def list_dataset(self):

        filename = os.path.join(self.data_path, '**', 'train', 'annotations', 'xmls', '*.xml')
        gt_files = sorted(glob.glob(filename, recursive=True))
        if not gt_files:
            raise FileNotFoundError(
                'No RDD2022 annotations under %s. Run download_rdd2022.py first.' % self.data_path)

        if self.countries:
            wanted = [c.lower() for c in self.countries]
            gt_files = [f for f in gt_files
                        if any(c in os.path.relpath(f, self.data_path).split(os.sep)[0].lower() for c in wanted)]

        gt_files = [f for f in gt_files if len(load_annotation(f)) > 0]

        random.Random(0).shuffle(gt_files)
        num_val = int(len(gt_files) * self.val_fraction)
        gt_files_train = gt_files[num_val:]
        gt_files_val = gt_files[:num_val]

        return gt_files_train, gt_files_val


    def __getitem__(self, idx):

        # gt file and its image
        filename_gt = self.gt_paths[idx]
        filename_image = image_path(filename_gt)
        # (N, 4) array of pothole boxes as (x1, y1, x2, y2) in original image pixels
        boxes = load_annotation(filename_gt)

        im = cv2.imread(filename_image)
        height, width = im.shape[:2]
        scale_width = self.yolo_image_size / width
        scale_height = self.yolo_image_size / height
        im = cv2.resize(im, (self.yolo_image_size, self.yolo_image_size))
        im = (im - self.pixel_mean) / 255.0
        image_blob = torch.from_numpy(im.transpose((2, 0, 1)).astype(np.float32))

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes = boxes[np.argsort(areas)]

        gt_box_blob = np.zeros((5, self.yolo_grid_num, self.yolo_grid_num), dtype=np.float32)
        gt_mask_blob = np.zeros((self.yolo_grid_num, self.yolo_grid_num), dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) * 0.5 * scale_width
            cy = (y1 + y2) * 0.5 * scale_height
            w = (x2 - x1) * scale_width
            h = (y2 - y1) * scale_height
            gx = min(int(cx / self.yolo_grid_size), self.yolo_grid_num - 1)
            gy = min(int(cy / self.yolo_grid_size), self.yolo_grid_num - 1)
            gt_box_blob[0, gy, gx] = cx / self.yolo_grid_size - gx
            gt_box_blob[1, gy, gx] = cy / self.yolo_grid_size - gy
            gt_box_blob[2, gy, gx] = w / self.yolo_image_size
            gt_box_blob[3, gy, gx] = h / self.yolo_image_size
            gt_box_blob[4, gy, gx] = 1
            gt_mask_blob[gy, gx] = 1
        gt_box_blob = torch.from_numpy(gt_box_blob)
        gt_mask_blob = torch.from_numpy(gt_mask_blob)

        # this is the sample dictionary to be returned from this function
        sample = {'image': image_blob,
                  'gt_box': gt_box_blob,
                  'gt_mask': gt_mask_blob}

        return sample


    # len of the dataset
    def __len__(self):
        return self.size


# draw grid on images for visualization
def draw_grid(image, line_space=64):
    H, W = image.shape[:2]
    image[0:H:line_space] = [255, 255, 0]
    image[:, 0:W:line_space] = [255, 255, 0]


# the main function for testing
if __name__ == '__main__':
    dataset_train = RoadDamage('train')
    dataset_val = RoadDamage('val')

    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

    # visualize the training data
    for i, sample in enumerate(train_loader):

        image = sample['image'][0].numpy().transpose((1, 2, 0))
        gt_box = sample['gt_box'][0].numpy()
        gt_mask = sample['gt_mask'][0].numpy()

        y, x = np.where(gt_mask == 1)
        cx = gt_box[0, y, x] * dataset_train.yolo_grid_size + x * dataset_train.yolo_grid_size
        cy = gt_box[1, y, x] * dataset_train.yolo_grid_size + y * dataset_train.yolo_grid_size
        w = gt_box[2, y, x] * dataset_train.yolo_image_size
        h = gt_box[3, y, x] * dataset_train.yolo_image_size

        x1 = cx - w * 0.5
        x2 = cx + w * 0.5
        y1 = cy - h * 0.5
        y2 = cy + h * 0.5

        print(image.shape, gt_box.shape, '%d potholes' % len(cx))

        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        im = image * 255.0 + dataset_train.pixel_mean
        im = im.astype(np.uint8)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.title('input image (448x448)', fontsize=16)

        ax = fig.add_subplot(1, 3, 2)
        draw_grid(im)
        plt.imshow(im[:, :, (2, 1, 0)])
        for j in range(len(cx)):
            rect = patches.Rectangle((x1[j], y1[j]), x2[j]-x1[j], y2[j]-y1[j], linewidth=2, edgecolor='g', facecolor="none")
            ax.add_patch(rect)
        plt.plot(cx, cy, 'ro', markersize=12)
        plt.title('Ground truth bounding boxes in YOLO format', fontsize=16)

        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(gt_mask)
        plt.title('Ground truth mask in YOLO format (7x7)', fontsize=16)
        plt.show()
