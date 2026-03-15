"""Traffic sign recognition pipeline for videos.

Created: Aug 2024
Purpose: Runs YOLO detection + CNN classification on an input video.
"""

from pathlib import Path
import argparse
import pickle
import time

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_demo_video(output_name: str = "result_demo.mp4", frame_count: int = 120):
    width, height = 960, 540
    fps = 24.0
    output_path = PROJECT_ROOT / output_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)

    if not writer.isOpened():
        raise ValueError(f"Could not create output video: {output_path}")

    start = time.time()
    for frame_idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (25, 25, 25)

        offset = int((frame_idx * 6) % (width - 220))
        x_min, y_min = 80 + offset, 160
        x_max, y_max = x_min + 140, y_min + 140

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 140, 255), 3)
        cv2.circle(frame, (x_min + 70, y_min + 70), 45, (0, 0, 255), -1)
        cv2.putText(frame, "STOP", (x_min + 28, y_min + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(
            frame,
            "Demo Mode: synthetic traffic-sign detection/classification",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
        )
        cv2.putText(
            frame,
            "Predicted: Stop (demo)",
            (x_min, y_min - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 140, 255),
            2,
        )

        writer.write(frame)

    writer.release()
    duration = time.time() - start
    fps_effective = round(frame_count / duration, 1) if duration > 0 else 0.0

    print("Running in demo mode (no external model/data assets required).")
    print(f"Frames processed: {frame_count}/{frame_count}")
    print(f"Total processing time: {duration:.5f} seconds")
    print(f"Average throughput: {fps_effective} FPS")
    print(f"Output saved to: {output_path}")


def set_network(config_path: Path, weights_path: Path):
    net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()

    output_layer_indices = []
    for layer in unconnected:
        output_layer_indices.append(int(layer[0] if isinstance(layer, (list, tuple, np.ndarray)) else layer))

    layers_names_output = [str(layer_names[index - 1]) for index in output_layer_indices]
    return net, layers_names_output


def set_output_stream(video_capture, frame_height: int, frame_width: int, output_name: str):
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = PROJECT_ROOT / output_name
    output_video = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height), True)
    return output_video, output_path


def class_names_fun(data_dir: Path):
    class_names_path = data_dir / "traffic-signs-preprocessed" / "label_names.csv"
    if not class_names_path.exists():
        raise FileNotFoundError(f"Label names file not found: {class_names_path}")

    labels_df = pd.read_csv(class_names_path)
    return dict(zip(labels_df["ClassId"], labels_df["SignName"]))


def get_predictions(net_output, confidence_threshold, nms_threshold, frame_width, frame_height):
    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in net_output:
        for detected_object in result:
            scores = detected_object[5:]
            class_current = int(np.argmax(scores))
            confidence_current = float(scores[class_current])

            if confidence_current > confidence_threshold:
                box_current = detected_object[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(confidence_current)
                class_numbers.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, confidence_threshold, nms_threshold)
    return results, bounding_boxes, class_numbers


def draw_markers(frame, results, bounding_boxes, scale_factor, mean, class_numbers, model, labels, colors):
    if results is None or len(results) == 0:
        return frame

    for result_idx in np.array(results).flatten():
        x_min, y_min, box_width, box_height = bounding_boxes[result_idx]

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_min + int(box_width))
        y_max = min(frame.shape[0], y_min + int(box_height))

        frame_ts = frame[y_min:y_max, x_min:x_max, :]
        if frame_ts.size == 0:
            continue

        blob_ts = cv2.dnn.blobFromImage(frame_ts, scale_factor, size=(32, 32), swapRB=True, crop=False)
        blob_ts[0] = blob_ts[0, :, :, :] - mean["mean_image_rgb"]
        blob_ts = blob_ts.transpose(0, 2, 3, 1)

        scores = model.predict(blob_ts, verbose=0)
        prediction = int(np.argmax(scores))

        class_idx = class_numbers[result_idx] % len(colors)
        color_box_current = colors[class_idx].tolist()

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_box_current, 2)
        label_text = labels.get(prediction, str(prediction))
        cv2.putText(frame, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box_current, 2)

    return frame


def validate_required_files(video_path: Path):
    required_paths = [
        video_path,
        PROJECT_ROOT / "data" / "mean_image_rgb.pickle",
        PROJECT_ROOT / "model" / "model-5x5.h5",
        PROJECT_ROOT / "weights" / "signs.weights",
        PROJECT_ROOT / "weights" / "yolov3_ts_test.cfg",
        PROJECT_ROOT / "data" / "traffic-signs-preprocessed" / "label_names.csv",
    ]

    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_text}")


def main(args):
    if args.get("demo"):
        run_demo_video(args.get("output", "result_demo.mp4"), args.get("demo_frames", 120))
        return

    video_argument = args.get("video")
    if not video_argument:
        raise ValueError("Please provide an input video using --video (or use --demo)")

    video_path = Path(video_argument)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    validate_required_files(video_path)

    data_dir = PROJECT_ROOT / "data"
    labels = class_names_fun(data_dir)

    with open(data_dir / "mean_image_rgb.pickle", "rb") as file:
        mean = pickle.load(file, encoding="latin1")

    model = load_model(str(PROJECT_ROOT / "model" / "model-5x5.h5"))

    weights_path = PROJECT_ROOT / "weights" / "signs.weights"
    config_path = PROJECT_ROOT / "weights" / "yolov3_ts_test.cfg"
    net, layers_names_output = set_network(config_path, weights_path)

    colors = np.random.randint(0, 255, size=(max(len(labels), 1), 3), dtype="uint8")
    confidence_threshold = 0.06
    nms_threshold = 0.08

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_tot = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    current_time = 0.0

    ret, frame = video.read()
    if not ret or frame is None:
        video.release()
        raise ValueError(f"Could not read first frame from video: {video_path}")

    frame_height, frame_width = frame.shape[:2]
    output_video, output_path = set_output_stream(video, frame_height, frame_width, args.get("output", "result.mp4"))

    while video.isOpened():
        start = time.time()

        scale_factor = 1 / 255.0
        blob = cv2.dnn.blobFromImage(frame, scale_factor, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        net_output = net.forward(layers_names_output)

        end = time.time()
        dt = end - start

        current_frame += 1
        current_time += dt

        results, bounding_boxes, class_numbers = get_predictions(
            net_output,
            confidence_threshold,
            nms_threshold,
            frame_width,
            frame_height,
        )

        if results is not None and len(results) > 0:
            frame = draw_markers(frame, results, bounding_boxes, scale_factor, mean, class_numbers, model, labels, colors)

        output_video.write(frame)

        ret, frame = video.read()
        if not ret:
            break

    video.release()
    output_video.release()
    cv2.destroyAllWindows()

    print(f"Frames processed: {current_frame}/{frame_tot if frame_tot > 0 else '?'}")
    print(f"Total processing time: {current_time:.5f} seconds")
    if current_time > 0:
        print(f"Average frames per second: {round((current_frame / current_time), 1)}")
    print(f"Output saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic sign recognition in video")
    parser.add_argument("-i", "--video", help="Path to input video")
    parser.add_argument("-o", "--output", default="result.mp4", help="Output video file name")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo mode without external assets")
    parser.add_argument("--demo-frames", type=int, default=120, help="Number of frames to generate in demo mode")
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(parse_args())
