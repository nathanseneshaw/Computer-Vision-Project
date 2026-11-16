"""
Fine-tunes a COCO-pretrained Ultralytics YOLO model on the RDD2022 potholes and reports its AP@0.5 on the val split
"""
import os

from ultralytics import YOLO

from prepare_labels import YOLO_DIR

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'ultralytics'))


if __name__ == '__main__':
    model = YOLO('yolo11s.pt')
    model.train(
        data=os.path.join(YOLO_DIR, 'potholes.yaml'),
        imgsz=640,
        epochs=100,
        patience=30,
        batch=16,
        workers=4,
        project=OUTPUT_DIR,
        name='yolo11s_potholes',
    )

    metrics = model.val()
    print('pothole AP@0.5: %.3f' % metrics.box.map50)
