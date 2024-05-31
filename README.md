# BKK Car Detection and Tracking

This project utilizes YOLO (You Only Look Once) model for detecting and tracking objects on the road in a video. It annotates the detected objects with ellipses and labels, and then processes the video to output a new video with annotations.

## Table of Contents
- [BKK Car Detection and Tracking](#BKK-Car-Detection-and-Tracking)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Advanced Usage](#advanced-usage)
  - [Acknowledgments](#acknowledgments)

## Requirements

Ensure you have the following installed:
- Python 3.8 or later
- `pip` package installer
- `torch` (PyTorch)
- `supervision`
- `ultralytics`
- `tqdm`
- `opencv-python`

You can install the required packages using `pip`:

```bash
pip install torch supervision ultralytics tqdm opencv-python
```

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/morsechim/BKKCarTracking.git
cd BKKCarTracking
```

2. **Ensure you have the necessary video file:**
   - Place your video file in the `videos` directory and name it `road.mp4`.

3. **Ensure you have the necessary model weights:**
   - Place the YOLOv8 model weights file in the `weights` directory and name it `yolov8x.pt`.

4. **Run the script:**

```bash
python main.py
```

5. **View the output:**
   - The processed video with annotations will be saved as `output.mp4` in the `videos` directory.

## Configuration

- **Model and Device:**
  - The YOLO model weights are expected to be located at `./weights/yolov8x.pt`.
  - The script automatically selects the device (`mps:0` if available, otherwise `cpu`).

- **Video Input/Output:**
  - Input video path: `./videos/road.mp4`
  - Output video path: `./videos/output.mp4`

- **Selected Classes:**
  - The classes of objects to track are specified by their class IDs. By default, the script tracks objects with class IDs `[2, 5, 7]`.

- **Callback Function:**
  - The `callback` function processes each frame of the video. It detects objects using the YOLO model, updates object tracks with a tracker, annotates objects with ellipses and labels, and returns the annotated frame.

## Advanced Usage

- **Customizing Detection Classes:**
  - You can modify the `selected_classes` variable in the script to specify the classes of objects you want to detect and track.

- **Adjusting Annotation Parameters:**
  - You can customize the appearance and position of annotations by modifying the annotator parameters in the script.

- **Fine-tuning Model:**
  - If needed, you can fine-tune the YOLO model with additional training data for better performance on specific object detection tasks.

## Acknowledgments

This project utilizes the following libraries:
- **[PyTorch](https://pytorch.org/)**
- **[Supervision](https://github.com/roboflow/supervision)**
- **[Ultralytics](https://github.com/ultralytics/)**

This project is inspired by the need for real-time object detection and tracking for various applications in surveillance and traffic monitoring.

---

*Note: Customize the repository URL, paths, and any other project-specific details as needed.*
