# Project Overview

This project detects collisions in dashcam videos using a fine-tuned 3D CNN (r3d_18). It processes short clips by extracting 16 frames around ```time_of_event``` and classifies them as crash or no-crash.

### 📁 File Overview

| Path                         | File                          | Description                                                                |
|------------------------------|-------------------------------|----------------------------------------------------------------------------|
| `scripts/`                   | `cnn_frame_extractor.ipynb`   | Notebook to extract 16-frame clips from videos for CNN input.              |
|                              | `join_clip_to_video.py`       | Reassembles predicted clips into a full video.                             |
|                              | `process_dataset.py`          | Processes raw video data into model-ready format.                          |
| `src/`                       | `dataset.py`                  | Loads and formats video clips as tensors for the model.                    |
|                              | `model.py`                    | Loads the pretrained 3D CNN (`r3d_18`) for crash detection.                |
|                              | `evaluate.ipynb`              | Runs evaluation on test data and outputs predictions.                      |
|                              | `generate_demo.ipynb`         | Builds demo videos with predictions                                        |
| `src/confusion_matrix/`      | *(various)*                   | Stores confusion matrix images or generation scripts.                      |
| `src/metrics_log/`           | *(various)*                   | Logs of evaluation metrics (accuracy, loss, etc.).                         |
| `src/plots/`                 | *(various)*                   | Output plots like accuracy curves or training history.                     |
| `src/test_metrics/`          | *(various)*                   | Scripts or files for testing and validating metric calculations.           |
