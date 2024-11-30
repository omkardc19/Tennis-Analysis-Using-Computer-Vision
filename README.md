# Tennis Analysis with YOLO, PyTorch, and Keypoint Extraction

## Introduction
This project develops a tennis analysis system using computer vision and machine learning techniques to analyze tennis matches. The system detects and tracks players, tennis balls, and key points on the court. It also calculates player speed, ball shot speed, and measures player coverage during the match. Additionally, it determines if the ball lands in or out of the court.

## Output Video 
Once the analysis is complete, the output video will be saved as output_video.avi. You can view the sample output video here:



https://github.com/user-attachments/assets/ab74642a-2a28-4839-b689-8a3e08fd4e0c



## Models Used
The project uses the following models:
- **YOLO (You Only Look Once)**: For real-time object detection of players and tennis balls.
- **CNN (Convolutional Neural Network)**: For court keypoint extraction, estimating key points on the tennis court to understand player positioning.
- **Custom YOLO Model**: Fine-tuned to detect tennis balls moving at high speeds.

## Training
- The YOLO model is trained on a dataset using **Roboflow** to improve tennis ball detection accuracy.
- The court keypoint extraction model is also trained using a custom dataset, which includes labeled data of key points on a tennis court.

To train the models:
1. **YOLO Ball Detector**: Fine-tuned using a Roboflow dataset containing tennis ball images.
2. **Court Keypoint Model**: Trained using a PyTorch-based convolutional neural network to predict key points on the court (e.g., baseline, net).

Training is done on Google Colab using the pre-trained YOLOv5 and YOLOv8 models.

## Installation Steps

### Clone the repository:
```bash
git clone https://github.com/your-username/tennis_analysis.git
cd tennis_analysis
```

### Requirements
Ensure Python 3.8 or higher is installed. The required libraries are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Requirements
The `requirements.txt` includes:

- torch
- ultralytics
- opencv-python
- numpy
- pandas









