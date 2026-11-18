"""
Turns raw road damage detections into a severity rating

RDD2022 only labels *what* and *where* the damage is, not how bad it is, so severity here is a
transparent heuristic computed from the detections, not something the model learned. It is based on
how much of the image the damage covers, how large the worst single defect is, and how many defects
there are, weighted by damage type. Tune the constants below to match your own definition of "bad".
"""
from dataclasses import dataclass

import numpy as np

# how much each damage type counts toward severity (1.0 = pothole)
# D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole
# the single-class model from train_ultralytics.py names its class "pothole"
CLASS_WEIGHTS = {
    'D40': 1.0, 'pothole': 1.0,
    'D20': 0.8,
    'D10': 0.4,
    'D00': 0.4,
}
DEFAULT_WEIGHT = 0.6  # for any class name not listed above

# human-readable names for the RDD2022 codes (other names are shown as they are)
CLASS_LABELS = {
    'D00': 'longitudinal crack',
    'D10': 'transverse crack',
    'D20': 'alligator crack',
    'D40': 'pothole',
}

# score components (see assess_image)
FULL_COVERAGE = 0.10    # weighted damage covering 10% of the image maxes out the coverage part
FULL_LARGEST = 0.04     # a single weighted defect covering 4% of the image maxes out the size part
FULL_COUNT = 5.0        # five full-weight defects max out the count part
POINTS_COVERAGE, POINTS_LARGEST, POINTS_COUNT = 60, 25, 15

# image-level score (0-100) to level
MINOR_BELOW = 25
MODERATE_BELOW = 55

# per-defect level from weighted image area fraction
# these roughly match the image-level thresholds above for an image with a single defect
DEFECT_MINOR_BELOW = 0.02
DEFECT_MODERATE_BELOW = 0.045

LEVELS = ('None', 'Minor', 'Moderate', 'Severe')


@dataclass
class Detection:
    name: str          # class name as reported by the model
    confidence: float
    box: tuple         # (x1, y1, x2, y2) in image pixels
    area_fraction: float = 0.0
    weight: float = DEFAULT_WEIGHT
    level: str = 'Minor'

    @property
    def label(self):
        return CLASS_LABELS.get(self.name, self.name)


@dataclass
class Assessment:
    level: str                 # one of LEVELS
    score: float               # 0-100
    num_defects: int
    coverage: float            # fraction of the image covered by any damage box (overlaps counted once)
    largest: float             # area fraction of the biggest single defect
    detections: list


def _weight(name):
    return CLASS_WEIGHTS.get(name, DEFAULT_WEIGHT)


def _defect_level(weighted_area):
    if weighted_area < DEFECT_MINOR_BELOW:
        return 'Minor'
    if weighted_area < DEFECT_MODERATE_BELOW:
        return 'Moderate'
    return 'Severe'


def _union_fraction(boxes, width, height, grid=256):
    # fraction of the image covered by the union of the boxes, on a coarse grid so overlaps count once
    mask = np.zeros((grid, grid), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        c1, c2 = int(np.floor(x1 / width * grid)), int(np.ceil(x2 / width * grid))
        r1, r2 = int(np.floor(y1 / height * grid)), int(np.ceil(y2 / height * grid))
        mask[max(r1, 0):min(r2, grid), max(c1, 0):min(c2, grid)] = True
    return float(mask.mean())


def assess_image(detections, width, height):
    """
    detections: list of Detection (name, confidence, box). Fills in area_fraction, weight and level on each.
    width, height: image size in pixels
    """
    if not detections:
        return Assessment('None', 0.0, 0, 0.0, 0.0, [])

    image_area = float(width * height)
    for d in detections:
        x1, y1, x2, y2 = d.box
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        d.area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
        d.weight = _weight(d.name)
        d.level = _defect_level(d.area_fraction * d.weight)

    weighted_areas = [d.area_fraction * d.weight for d in detections]
    coverage_part = min(1.0, sum(weighted_areas) / FULL_COVERAGE) * POINTS_COVERAGE
    largest_part = min(1.0, max(weighted_areas) / FULL_LARGEST) * POINTS_LARGEST
    count_part = min(1.0, sum(d.weight for d in detections) / FULL_COUNT) * POINTS_COUNT
    score = coverage_part + largest_part + count_part

    if score < MINOR_BELOW:
        level = 'Minor'
    elif score < MODERATE_BELOW:
        level = 'Moderate'
    else:
        level = 'Severe'
    # an image is never rated milder than its worst single defect
    level = max([level] + [d.level for d in detections], key=LEVELS.index)

    detections = sorted(detections, key=lambda d: d.area_fraction * d.weight, reverse=True)
    return Assessment(
        level=level,
        score=round(score, 1),
        num_defects=len(detections),
        coverage=_union_fraction([d.box for d in detections], width, height),
        largest=max(d.area_fraction for d in detections),
        detections=detections,
    )
