"""Visualize YOLO annotations on a single traffic sign image.

Created: Aug 2024
Purpose: Draws annotation bounding boxes and saves a preview image.
"""

from pathlib import Path
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "traffic-signs-dataset-in-yolo-format" / "ts" / "ts"


def generate_demo_image(output_name: str = "example_demo.png") -> Path:
    width, height = 960, 540
    image_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    image_bgr[:] = (32, 32, 32)

    x_min, y_min, box_width, box_height = 330, 150, 220, 220
    cv2.rectangle(image_bgr, (x_min, y_min), (x_min + box_width, y_min + box_height), (172, 10, 127), 3)
    cv2.circle(image_bgr, (x_min + 110, y_min + 110), 70, (0, 0, 255), -1)
    cv2.putText(image_bgr, "STOP", (x_min + 68, y_min + 125), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(image_bgr, "Class: 14 (demo)", (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (172, 10, 127), 2)
    cv2.putText(image_bgr, "Demo Mode: synthetic annotation image", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    plt.rcParams["figure.figsize"] = (12, 7)
    figure = plt.figure()
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Demo Traffic Sign Annotation", fontsize=16)

    output_path = PROJECT_ROOT / output_name
    figure.savefig(output_path)
    plt.close()
    return output_path


def visualize_image(image_num: str, output_name: str = "example.png") -> Path:
    image_path = DATASET_DIR / f"{image_num}.jpg"
    label_path = DATASET_DIR / f"{image_num}.txt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Annotation not found: {label_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image_bgr.shape[:2]

    with open(label_path, "r", encoding="utf-8") as file:
        annotations = [line.rstrip() for line in file if line.strip()]

    for annotation in annotations:
        class_id, x_center, y_center, box_width, box_height = annotation.split()
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
            (172, 10, 127),
            2,
        )
        cv2.putText(
            image_bgr,
            f"Class: {class_id}",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (172, 10, 127),
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
    parser = argparse.ArgumentParser(description="Visualize annotations on one image")
    parser.add_argument("--image-num", default="00001", help="Image ID without extension")
    parser.add_argument("--output", default="example.png", help="Output image file name")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo mode without dataset files")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.demo:
        saved_to = generate_demo_image(args.output)
        print("Running in demo mode (no dataset files required).")
    else:
        saved_to = visualize_image(args.image_num, args.output)
    print(f"Saved output to: {saved_to}")


if __name__ == "__main__":
    main()
