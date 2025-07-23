# Collision Detection
A Resnet3D_18 model-based project for classifying crash vs. no-crash dashcam video

# Project Overview

This project detects collisions in dashcam videos using a fine-tuned 3D CNN (r3d_18). It processes short clips by extracting 16 frames around ```time_of_event``` and classifies them as crash or no-crash. This system addresses a growing need for **automated crash detection** in road safety applications, insurance tech, and fleet monitoring. By analyzing raw dashcam footage in real time, this offers a scalable solution to detect incidents **without relying on external sensors while minimizing the need for manual review**.


The full pipeline from preprocessing, prediction to demo UI—is **containerized and deployed automatically to Google Cloud** using a CI/CD workflow with **GitHub Actions**, **Docker**, and **Google Cloud Run**.

# Dataset

This project uses labeled dashcam video clips from the [Nexar AI Dashcam Challenge dataset]((https://www.kaggle.com/competitions/nexar-collision-prediction))]. Clips were trimmed and preprocessed into 16-frame sequences centered around annotated collision events.

- Videos were sampled to include both **crash** and **no crash** events.

### Model

This project uses **ResNet3D-18 (`r3d_18`)**, a 3D convolutional neural network designed to recognize actions in video by capturing both **spatial** (what something looks like) and **temporal** (how it moves) features.
Unlike 2D CNNs that operate on individual frames, **3D CNNs capture both spatial and temporal features** by applying convolutions across time (video frames).

### How it works:
- Input shape: **(batch_size, channels=3, frames=16, height, width)**
- Applies **3D convolutions** over time and space.
- Based on the **ResNet-18** architecture (with skip connections) to handle 3D data.
- Final outputs `crash` or `no_crash` class probabilities.

This allows the model to detect subtle patterns like sudden stops, collisions, or motion changes that are only visible when analyzing a sequence of frames.

### Paper Reference:
> Kensho Hara, Hirokatsu Kataoka, Yutaka Satoh (2017). [**Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?**](
https://doi.org/10.48550/arXiv.1708.07632) — ICCV 2017.

# Project Flow
<img width="1280" height="377" alt="image" src="https://github.com/user-attachments/assets/1a48d280-eb86-4715-9b77-9104e4aebfb9" />


# File Overview

| Path                         | File                          | Description                                                                |
|------------------------------|-------------------------------|----------------------------------------------------------------------------|
| `scripts/`                   | `cnn_frame_extractor.ipynb`   | Notebook to extract 16-frame clips from videos for CNN input.              |
|                              | `join_clip_to_video.py`       | Reassembles predicted clips into a full video.                             |
|                              | `process_dataset.py`          | Processes raw video data into model-ready format.                          |
| `src/`                       | `dataset.py`                  | Loads and formats video clips as tensors for the model.                    |
|                              | `model.py`                    | Loads the pretrained 3D CNN (`r3d_18`) for crash detection.                |
|                              | `evaluate.ipynb`              | Notebook to evaluate the model and compute performance metrics.            |
|                              | `generate_demo.py`            | Builds demo videos with predicted overlays.                                |
|                              | `generate_demo.ipynb`         | Notebook version for interactive demo generation.                          |
|                              | `train_model.py`              | Fine-tunes the 3D CNN on labeled video clips.                              |
|                              | `train_model.ipynb`           | Notebook version for training the model with visualization.                |
| `src/confusion_matrix/`      | *(various)*                   | Stores confusion matrix images or generation scripts.                      |
| `src/metrics_log/`           | *(various)*                   | Logs of evaluation metrics (accuracy, loss, etc.).                         |
| `src/plots/`                 | *(various)*                   | Output plots like accuracy curves or training history.                     |
| `src/test_metrics/`          | *(various)*                   | Scripts or files for testing and validating metric calculations.           |


# 📽️ Demo

> ⚠️ **Note:** This app is hosted on Google Cloud Run and may take up to 1 minute to load due to cold starts when inactive.

🔗 [🌐 Open Live Demo](https://collision-detection-1035539878985.us-central1.run.app/)

<img width="975" height="797" alt="Collision Detection Demo Screenshot" src="https://github.com/user-attachments/assets/030c8bd9-2d64-48ab-bc0a-67e72913f3a7" />


# Setup Instructions

### Run with Docker

```bash
# Clone the repo
git clone https://github.com/KosiAtupulazi/collision-detection.git
cd collision-detection

# Build and run the container
docker build -t collision-app .
docker run -p 8501:8501 collision-app
