# High-Altitude Human Deductions with YOLOv8

## Overview

This project uses the YOLOv8 object detection model to detect humans in high-altitude drone footage. The model is trained to recognize human figures from a distance and can be applied to video footage for real-time detection and annotation.

## Features

- Load a pre-trained YOLOv8 model for object detection.
- Process video files to detect and annotate human figures.
- Save annotated video output.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/High-Altitude-Human-Deductions-YOLOv8.git
    cd High-Altitude-Human-Deductions-YOLOv8
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Place your model file and input video in the appropriate directories.**

2. **Update the paths in the `process_video.py` script:**

    ```python
    model_path = r'path\to\your\model.pt'
    input_video_path = r'path\to\your\input_video.mp4'
    output_video_path = r'path\to\your\output_video.mp4'
    ```

3. **Run the script:**
    ```bash
    python process_video.py
    ```

4. **The output video with annotations will be saved to the specified `output_video_path`.**

## Script Details

- `process_video.py`: Main script for video processing.
  - Loads the YOLOv8 model.
  - Processes the input video frame by frame.
  - Annotates detected humans with bounding boxes and confidence scores.
  - Saves the annotated video to the specified output path.

## Requirements

- `ultralytics==8.0.114`
- `torch==2.0.2`
- `numpy==1.24.3`
- `opencv-python==4.7.0.68`
- `matplotlib==3.8.3`
- `pandas==2.0.2`
- `scipy==1.10.1`
- `tqdm==4.66.2`
- `seaborn==0.13.2`
- `tensorboard==2.13.0`

## Acknowledgements

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/yolov5) for providing the state-of-the-art object detection models.
- **OpenCV**: [OpenCV](https://opencv.org/) for image and video processing tools.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
