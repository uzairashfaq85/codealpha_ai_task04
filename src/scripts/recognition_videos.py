"""Visualize YOLO annotations with human-readable class names on one image.

Created: Aug 2024
Purpose: Draws bounding boxes with label names and saves a preview image.
"""

from pathlib import Path
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "traffic-signs-dataset-in-yolo-format" / "ts" / "ts"
LABELS_PATH = PROJECT_ROOT / "data" / "traffic-signs-preprocessed" / "label_names.csv"


def generate_demo_labeled_image(output_name: str = "results3_demo.png") -> Path:
    width, height = 960, 540
    image_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    image_bgr[:] = (28, 28, 28)

    x_min, y_min, box_width, box_height = 320, 140, 240, 240
    cv2.rectangle(image_bgr, (x_min, y_min), (x_min + box_width, y_min + box_height), (255, 0, 0), 3)
    cv2.circle(image_bgr, (x_min + 120, y_min + 120), 80, (0, 255, 255), -1)

    cv2.putText(image_bgr, "Class: Speed limit (30km/h) [demo]", (x_min - 40, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(image_bgr, "Demo Mode: synthetic labeled annotation", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    plt.rcParams["figure.figsize"] = (12, 7)
    figure = plt.figure()
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Demo Traffic Sign Labeled Annotation", fontsize=16)

    output_path = PROJECT_ROOT / output_name
    figure.savefig(output_path)
    plt.close()
    return output_path


def visualize_labeled_image(image_num: str, output_name: str = "results3.png") -> Path:
    image_path = DATASET_DIR / f"{image_num}.jpg"
    label_path = DATASET_DIR / f"{image_num}.txt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Annotation not found: {label_path}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Label names file not found: {LABELS_PATH}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    labels_df = pd.read_csv(LABELS_PATH)
    label_map = dict(zip(labels_df["ClassId"], labels_df["SignName"]))

    height, width = image_bgr.shape[:2]

    with open(label_path, "r", encoding="utf-8") as file:
        annotations = [line.rstrip() for line in file if line.strip()]

    for annotation in annotations:
        class_id, x_center, y_center, box_width, box_height = annotation.split()
        class_id = int(class_id)

        x_center = int(float(x_center) * width)
        y_center = int(float(y_center) * height)
        box_width = int(float(box_width) * width)
        box_height = int(float(box_height) * height)

        x_min = int(x_center - (box_width / 2))
        y_min = int(y_center - (box_height / 2))

        cv2.rectangle(
            image_bgr,
            (x_min, y_min),
            (x_min + box_width, y_min + box_height),
            (255, 0, 0),
            2,
        )

        class_name = label_map.get(class_id, str(class_id))
        cv2.putText(
            image_bgr,
            f"Class: {class_name}",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    plt.rcParams["figure.figsize"] = (15, 15)
    figure = plt.figure()
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Image {image_num}.jpg with Traffic Signs", fontsize=18)

    output_path = PROJECT_ROOT / output_name
    figure.savefig(output_path)
    plt.show()
    plt.close()
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize labeled annotations on one image")
    parser.add_argument("--image-num", default="00074", help="Image ID without extension")
    parser.add_argument("--output", default="results3.png", help="Output image file name")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo mode without dataset files")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.demo:
        saved_to = generate_demo_labeled_image(args.output)
        print("Running in demo mode (no dataset files required).")
    else:
        saved_to = visualize_labeled_image(args.image_num, args.output)
    print(f"Saved output to: {saved_to}")


if __name__ == "__main__":
    main()
