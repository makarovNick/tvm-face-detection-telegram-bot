import os
import telebot
import cv2
import numpy as np
from telebot import types
from tvm import relay
from tvm.contrib import graph_executor
import tvm
import logging

# Configure the logger
logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)

# Initialize the bot
bot = telebot.TeleBot(os.getenv('TELEGRAM_BOT_TOKEN_ENV'))

# Load TVM module
package_path = "face_detection_optimized.so"
loaded_lib = tvm.runtime.load_module(package_path)
module = graph_executor.GraphModule(loaded_lib["default"](tvm.cpu()))

# Utility functions
def run_inference(module, image):
    module.set_input("input", tvm.nd.array(image))
    module.run()
    bboxes_and_keypoints = module.get_output(0).numpy()
    detection_scores = module.get_output(1).numpy()
    return bboxes_and_keypoints, detection_scores

def load_image_for_inference(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    image = cv2.resize(image, (128, 128))
    image = (image.astype(np.float32) - 128) / 128.0
    return np.expand_dims(image, axis=0), original_w, original_h

def create_blazeface_anchors(input_w, input_h):
    strides = [8, 16]
    anchors_per_layer = [2, 6]
    anchors = []
    for stride, num_anchors in zip(strides, anchors_per_layer):
        grid_cols = (input_w + stride - 1) // stride
        grid_rows = (input_h + stride - 1) // stride
        for grid_y in range(grid_rows):
            anchor_y = stride * (grid_y + 0.5)
            for grid_x in range(grid_cols):
                anchor_x = stride * (grid_x + 0.5)
                for _ in range(num_anchors):
                    anchors.append((anchor_x, anchor_y))
    return np.array(anchors, dtype=np.float32)

def non_max_suppression(faces, iou_thresh=0.3):
    faces = sorted(faces, key=lambda x: x["score"], reverse=True)
    selected_faces = []
    def calc_iou(face0, face1):
        xmin0, ymin0, xmax0, ymax0 = face0["bbox"]
        xmin1, ymin1, xmax1, ymax1 = face1["bbox"]
        intersect_xmin = max(xmin0, xmin1)
        intersect_ymin = max(ymin0, ymin1)
        intersect_xmax = min(xmax0, xmax1)
        intersect_ymax = min(ymax0, ymax1)
        intersect_area = max(0, intersect_xmax - intersect_xmin) * max(0, intersect_ymax - intersect_ymin)
        area0 = (xmax0 - xmin0) * (ymax0 - ymin0)
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        return intersect_area / (area0 + area1 - intersect_area)
    for face in faces:
        keep = True
        for sel_face in selected_faces:
            if calc_iou(face, sel_face) >= iou_thresh:
                keep = False
                break
        if keep:
            selected_faces.append(face)
    return selected_faces

def draw_detections(image, faces, original_w, original_h):
    for face in faces:
        xmin, ymin, xmax, ymax = face['bbox']
        cv2.rectangle(
            image, 
            (int(xmin * original_w), int(ymin * original_h)), 
            (int(xmax * original_w), int(ymax * original_h)), 
            (0, 255, 0), 2
        )
        for kp_x, kp_y in face['keypoints']:
            cv2.circle(
                image, 
                (int(kp_x * original_w), int(kp_y * original_h)), 
                3, (255, 0, 0), -1
            )
        cv2.putText(
            image, f"Score: {face['score']:.2f}", 
            (int(xmin * original_w), int(ymin * original_h) - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
    return image

def process_outputs(bboxes_and_keypoints, detection_scores, anchors, input_w, input_h, score_threshold=0.75, iou_threshold=0.3):
    scores = 1 / (1 + np.exp(-detection_scores[0, :, 0]))
    valid_indices = np.where(scores >= score_threshold)[0]
    valid_bboxes = bboxes_and_keypoints[0, valid_indices]
    valid_scores = scores[valid_indices]
    detected_faces = []
    for idx, bbox_keypoint in enumerate(valid_bboxes):
        sx, sy, w, h = bbox_keypoint[:4]
        score = valid_scores[idx]
        anchor = anchors[valid_indices[idx]]
        cx = sx + anchor[0]
        cy = sy + anchor[1]
        cx /= input_w
        cy /= input_h
        w /= input_w
        h /= input_h
        xmin = max(0, cx - w / 2)
        ymin = max(0, cy - h / 2)
        xmax = min(1, cx + w / 2)
        ymax = min(1, cy + h / 2)
        keypoints = [
            ((bbox_keypoint[4 + 2 * kp_idx] + anchor[0]) / input_w, 
             (bbox_keypoint[4 + 2 * kp_idx + 1] + anchor[1]) / input_h)
            for kp_idx in range(6)
        ]
        detected_faces.append({"score": score, "bbox": (xmin, ymin, xmax, ymax), "keypoints": keypoints})
    return non_max_suppression(detected_faces, iou_threshold)

def handler_face_detection(message):
    if message.content_type == "photo":
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        input_file = os.path.join(".", "input_image.jpg")
        with open(input_file, "wb") as f:
            f.write(downloaded_file)
        bot.send_message(message.chat.id, "Processing image...")
        try:
            image = cv2.imread(input_file)
            processed_image, original_w, original_h = load_image_for_inference(input_file)
            bboxes_keypoints, detection_scores = run_inference(module, processed_image)
            anchors = create_blazeface_anchors(128, 128)
            detected_faces = process_outputs(bboxes_keypoints, detection_scores, anchors, 128, 128)
            image_with_detections = draw_detections(image, detected_faces, original_w, original_h)
            output_file = os.path.join(".", "output_image.jpg")
            cv2.imwrite(output_file, image_with_detections)
            bot.send_photo(message.chat.id, photo=open(output_file, "rb"))
        except Exception as e:
            bot.send_message(message.chat.id, f"Error during processing: {str(e)}")
    else:
        bot.send_message(message.chat.id, "Please send a valid image.")

@bot.message_handler(commands=["detect_faces"])
def command_face_detection(message):
    prompt = bot.send_message(message.chat.id, "Send an image for face detection:")
    bot.register_next_step_handler(prompt, handler_face_detection)

bot.polling(none_stop=True, interval=0)
