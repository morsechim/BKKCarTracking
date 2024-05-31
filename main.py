import os
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
import cv2

import ultralytics

ultralytics.checks()

# Define video paths
input_path = os.path.join(".", "videos", "road.mp4")
output_path = os.path.join(".", "videos", "output.mp4")

# Define processor device
device = "mps:0" if torch.backends.mps.is_available() else "cpu"
print(f"Processor device: {device}")

# Define model path
model_path = os.path.join(".", "weights", "yolov8x.pt")
# Define YOLO model
model = YOLO(model_path).to(device)
model.fuse()

# Define tracker 
tracker = sv.ByteTrack()

# Define bbox annotator
# box_annotator = sv.BoundingBoxAnnotator()

# Define ellipse_annotator
ellipse_annotator = sv.EllipseAnnotator()
# Define label annotator
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_RIGHT)

selected_classes = [2, 5, 7]

# Define callback function for processing frames
def callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, selected_classes)]
    detections = tracker.update_with_detections(detections)
    
    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]
    
    annotated_frame = ellipse_annotator.annotate(scene=frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)

# Process video with the callback function
def process_video_with_progress(source_path: str, target_path: str, callback) -> None:
    cap = cv2.VideoCapture(source_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = callback(frame, frame_idx)
            out.write(annotated_frame)
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()

# Process video with progress bar
process_video_with_progress(input_path, output_path, callback)
