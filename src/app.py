"""
Web interface for the road damage detector: upload a road photo, get the damage marked on it and a severity rating

Run from the repo root:
    python src/app.py
    python src/app.py --weights path/to/best.pt
"""
import argparse
import glob
import os

from PIL import Image, ImageDraw, ImageFont, ImageOps

from severity import Assessment, Detection, assess_image

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MODELS_DIR = os.path.join(REPO_DIR, 'models', 'ultralytics')

# the size train_ultralytics.py trains at
IMAGE_SIZE = 1024

LEVEL_COLORS = {'Minor': (250, 204, 21), 'Moderate': (249, 115, 22), 'Severe': (220, 38, 38)}
LEVEL_ICONS = {'None': '🟢', 'Minor': '🟡', 'Moderate': '🟠', 'Severe': '🔴'}


# newest best.pt under models/ultralytics, i.e. the model train_ultralytics.py produced
def find_weights():
    candidates = glob.glob(os.path.join(MODELS_DIR, '**', 'weights', 'best.pt'), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            'No trained weights found under %s.\n'
            'Train a model first (python src/prepare_labels.py, then python src/train_ultralytics.py), '
            'or point to one with --weights path/to/best.pt.' % MODELS_DIR)
    return max(candidates, key=os.path.getmtime)


# run the model on a PIL image and return a list of Detection
def detect(model, image, confidence):
    result = model.predict(image, conf=confidence, imgsz=IMAGE_SIZE, verbose=False)[0]
    detections = []
    for box, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        detections.append(Detection(name=str(model.names[int(cls)]), confidence=float(conf), box=tuple(box)))
    return detections


# draw each defect's box coloured by its own severity
def annotate(image, assessment):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    short_side = min(out.size)
    line_width = max(2, short_side // 250)
    try:
        font = ImageFont.load_default(size=max(12, short_side // 40))
    except TypeError:  # Pillow < 10.1 has a fixed-size default font
        font = ImageFont.load_default()

    for d in assessment.detections:
        color = LEVEL_COLORS[d.level]
        x1, y1, x2, y2 = d.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text = '%s %.0f%%' % (d.label, d.confidence * 100)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        label_top = max(0, y1 - text_h - 6)
        draw.rectangle((x1, label_top, x1 + text_w + 8, label_top + text_h + 6), fill=color)
        draw.text((x1 + 4, label_top + 2 - top), text, fill=(0, 0, 0), font=font)
    return out


def summarize(assessment: Assessment, confidence):
    if assessment.level == 'None':
        return ('### %s No road damage detected\n'
                'Nothing found at confidence ≥ %.2f. If you can see damage the model missed, '
                'try lowering the confidence threshold.' % (LEVEL_ICONS['None'], confidence))
    return ('### %s Severity: %s (score %.0f / 100)\n'
            '- **%d** defect%s detected\n'
            '- Damage covers **%.1f%%** of the image\n'
            '- Largest single defect covers **%.1f%%** of the image'
            % (LEVEL_ICONS[assessment.level], assessment.level, assessment.score, assessment.num_defects,
               '' if assessment.num_defects == 1 else 's', assessment.coverage * 100, assessment.largest * 100))


def make_predict_fn(model):
    def predict(image, confidence):
        if image is None:
            return None, 'Upload an image to get started.', []
        image = ImageOps.exif_transpose(image).convert('RGB')
        assessment = assess_image(detect(model, image, confidence), *image.size)
        rows = [[d.label, '%.2f' % d.confidence, '%.2f%%' % (d.area_fraction * 100), d.level]
                for d in assessment.detections]
        return annotate(image, assessment), summarize(assessment, confidence), rows
    return predict


def build_ui(model):
    import gradio as gr

    with gr.Blocks(title='Road damage detector') as ui:
        gr.Markdown('# Road Damage Detector Project\n'
                    'Input a road photo and our model will look for potential damage.')
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type='pil', label='Road image')
                confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label='Confidence threshold',
                                       info='Lower finds more damage but also more false alarms')
                button = gr.Button('Analyze', variant='primary')
            with gr.Column():
                image_out = gr.Image(label='Detected damage', interactive=False)
                verdict = gr.Markdown()
        table = gr.Dataframe(headers=['Damage type', 'Confidence', 'Image area', 'Severity'],
                             label='Detections (worst first)', interactive=False)
        predict = make_predict_fn(model)
        button.click(predict, [image_in, confidence], [image_out, verdict, table])
        image_in.upload(predict, [image_in, confidence], [image_out, verdict, table])
    return ui


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', help='path to a trained .pt file (default: newest best.pt under models/ultralytics)')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true', help='create a temporary public link')
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = args.weights or find_weights()
    print('Loading %s' % weights)
    model = YOLO(weights)
    build_ui(model).launch(server_port=args.port, share=args.share, footer_links=[])


if __name__ == '__main__':
    main()