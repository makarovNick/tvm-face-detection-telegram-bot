# Face Detection Bot Using TVM

## Model Overview

This project utilizes a face detection model converted to TVM IR, optimized, and compiled for `x86_64-linux-gnu`. The model is based on a **TensorFlow Lite** implementation of the BlazeFace face detector. The project performs real-time face detection using a Telegram bot.

### Model Details

- **Input**: A normalized 128x128 RGB image.
- **Output**: Bounding boxes, keypoints, and detection probabilities.
- **Classes Detected**: Single-face detection.

## TVM Model Conversion and Optimization

### Conversion Process
The original TensorFlow Lite model was converted to TVM IR. The full process is documented in notebook.

### Tuning
The model was tuned using TVM's AutoTVM with these parameters:

| Parameter      | Value | Description                                               |
|----------------|-------|-----------------------------------------------------------|
| tuner          | xgb   | XGBoost provides efficient and accurate tuning.          |
| early_stopping | False | No early stopping; trials limited manually.               |
| n_trial        | 100   | A balanced number of trials to ensure optimal tuning.     |

### Exporting Artifacts

The optimized model was exported using TVM’s built-in functions:

- Model library: `face_detection_optimized.so`
- Export method: `lib.export_library`

The combined export ensured the model and its parameters were encapsulated in a single file for deployment.

## Telegram Bot Implementation

The bot performs face detection on user-uploaded images and returns annotated results.

### Supported Commands

1. `/detect_faces`: Upload an image to detect faces.

### Processing Details

- The bot accepts uploaded images.
- It preprocesses images by resizing them to 128x128.
- The model performs inference and returns:
  - Bounding boxes
  - Facial keypoints
  - Detection probabilities
- Annotated images are sent back to the user.

### Example Output

| Original Image | Detected Faces |
|----------------|-----------------|
| ![image](https://github.com/user-attachments/assets/edbd2dec-13cf-4cbe-b57a-3ce57b5b7ca4) | ![image](https://github.com/user-attachments/assets/15c322ab-4c2e-4f5a-84b3-befa2d713e7b) |
| ![image](https://github.com/user-attachments/assets/535a7826-b662-4483-9a6c-05d590fc6098) | ![image](https://github.com/user-attachments/assets/3c211cb1-56c1-4297-891a-7c7201af3e1d) |


## Deployment Instructions

1. **Setup Environment**:
   - Create an `.env` file with the following content:
     ```bash
     TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
     ```

2. **Build Docker Image**:
   ```bash
   docker-compose --project-name tvm_face_bot build
   ```

3. **Run the Bot**:
   ```bash
   docker-compose --project-name tvm_face_bot --env-file .env up
   ```

The bot will go online and be ready to accept image uploads for face detection.

---

For more information, check the detailed conversion and tuning process in notebook.

