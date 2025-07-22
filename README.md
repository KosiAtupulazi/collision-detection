# Project Overview

This project detects collisions in dashcam videos using a fine-tuned 3D CNN (r3d_18). It processes short clips by extracting 16 frames around ```time_of_event``` and classifies them as crash or no-crash.

### 🧠 Model Overview

This project uses **ResNet3D-18 (`r3d_18`)**, a 3D convolutional neural network designed for action recognition in videos. Unlike 2D CNNs that operate on individual frames, **3D CNNs capture both spatial and temporal features** by applying convolutions across time (video frames) as well as space (width & height).

#### 📦 Key Characteristics:
- Accepts **video clips** (e.g., 16 frames × 3 channels × H × W) as input.
- Uses **3D convolutional kernels** to extract motion patterns across frames.
- Based on the classic **ResNet-18** architecture but extended to 3D convolutions.
- Outputs class probabilities (e.g., `crash` vs. `no_crash`) through a final softmax layer.

This allows the model to detect subtle patterns like sudden stops, collisions, or motion changes that are only visible when analyzing a sequence of frames.

#### 📚 Paper Reference:
> Hara, K., Kataoka, H., & Satoh, Y. (2018). [**Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?**](https://arxiv.org/abs/1711.11248) — CVPR 2018.


### 📁 File Overview

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

### 📽️ Demo

> ⚠️ **Note:** This app is hosted on Google Cloud Run and may take up to 1 minute to load due to cold starts when inactive.

🔗 [🌐 Open Live Demo](https://collision-detection-1035539878985.us-central1.run.app/)

<img width="975" height="797" alt="Collision Detection Demo Screenshot" src="https://github.com/user-attachments/assets/030c8bd9-2d64-48ab-bc0a-67e72913f3a7" />


