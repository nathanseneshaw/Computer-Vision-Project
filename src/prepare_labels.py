"""
Converts the RDD2022 pothole annotations to Ultralytics YOLO labels, train/val image lists and a dataset yaml
"""
import os

from PIL import Image

from data import DATA_PATH, RoadDamage, image_path, load_annotation

YOLO_DIR = os.path.normpath(os.path.join(DATA_PATH, '..', 'yolo'))


def label_path(image_file):
    images_dir, name = os.path.split(image_file)
    return os.path.join(os.path.dirname(images_dir), 'labels', os.path.splitext(name)[0] + '.txt')


def write_labels(xml_file):
    image_file = image_path(xml_file)
    with Image.open(image_file) as im:
        width, height = im.size

    lines = []
    for x1, y1, x2, y2 in load_annotation(xml_file):
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append('0 %.6f %.6f %.6f %.6f' % ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                                                (x2 - x1) / width, (y2 - y1) / height))

    out = label_path(image_file)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return os.path.abspath(image_file)


if __name__ == '__main__':
    dataset = RoadDamage('train')
    os.makedirs(YOLO_DIR, exist_ok=True)

    for split, xml_files in (('train', dataset.gt_files_train), ('val', dataset.gt_files_val)):
        image_files = [write_labels(f) for f in xml_files]
        with open(os.path.join(YOLO_DIR, split + '.txt'), 'w') as f:
            f.write('\n'.join(image_files) + '\n')
        print('%s: %d images' % (split, len(image_files)))

    yaml_file = os.path.join(YOLO_DIR, 'potholes.yaml')
    with open(yaml_file, 'w') as f:
        f.write('train: %s\n' % os.path.join(YOLO_DIR, 'train.txt').replace('\\', '/'))
        f.write('val: %s\n' % os.path.join(YOLO_DIR, 'val.txt').replace('\\', '/'))
        f.write('names:\n  0: pothole\n')
    print('wrote %s' % yaml_file)
