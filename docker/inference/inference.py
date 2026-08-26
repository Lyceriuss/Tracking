import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
import sys
import time
import logging
import threading
import supervision as sv
from collections import defaultdict, deque
from flask import Flask, Response, render_template_string

# --- LOGGING SETUP ---
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("/app/inference_engine.log"), logging.StreamHandler(sys.stdout)]
)

SCHEMA_KEYS = [
    "View_Front", "View_Back", "View_Side", "Female", "Male",
    "Age_Child", "Age_Adult", "Age_Senior", "Bald", "Short_Hair", "Long_Hair",
    "Backpack", "Hat", "Glasses", "Muffler_Scarf", "ShoulderBag", 
    "HandBag", "PlasticBag", "CarryingOther", "ShortSleeve", "LongSleeve", 
    "Tshirt", "Jacket", "LongCoat", "Logo", "Plaid", "Stripe",
    "Trousers", "Jeans", "Shorts", "Skirt_or_Dress",
    "Boots", "Sneakers", "LeatherShoes", "Sandals"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Pure ONNX Web Control Center")
    parser.add_argument("--source", type=str, default=os.environ.get("STREAM_SOURCE", "rtsp://streamer_1:8554/live/stream"), help="Video stream source")
    parser.add_argument("--par-model", type=str, default="baseline_v4_prod.onnx", help="Path to PAR ONNX weights")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.onnx", help="Path to YOLO ONNX weights")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    return parser.parse_args()

# --- YOLO ONNX UTILS ---
def preprocess_yolo(img, input_size=(640, 640)):
    shape = img.shape[:2]
    r = min(input_size[0] / shape[0], input_size[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = input_size[1] - new_unpad[0], input_size[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(img_chw, axis=0), r, dw, dh

def postprocess_yolo(outputs, r, dw, dh, conf_threshold=0.4, iou_threshold=0.4):
    predictions = np.squeeze(outputs).T
    scores = predictions[:, 4]  # Class 0 (Person)
    mask = scores > conf_threshold
    
    valid_preds = predictions[mask]
    valid_scores = scores[mask]
    
    if len(valid_preds) == 0: return np.empty((0, 4)), np.empty(0)
        
    cx, cy, w, h = valid_preds[:, 0], valid_preds[:, 1], valid_preds[:, 2], valid_preds[:, 3]
    
    x1 = (cx - w / 2 - dw) / r
    y1 = (cy - h / 2 - dh) / r
    x2 = (cx + w / 2 - dw) / r
    y2 = (cy + h / 2 - dh) / r
    
    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes, valid_scores.tolist(), conf_threshold, iou_threshold)
    
    if len(indices) == 0: return np.empty((0, 4)), np.empty(0)
        
    indices = indices.flatten()
    final_boxes = np.stack([x1[indices], y1[indices], x2[indices], y2[indices]], axis=1)
    return final_boxes, valid_scores[indices]

# --- PAR ONNX UTILS (Identical to your baseline) ---
def preprocess_crop(crop_bgr):
    resized = cv2.resize(crop_bgr, (128, 256))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.transpose((img - mean) / std, (2, 0, 1))

def parse_attributes(agg_probs, schema_keys=SCHEMA_KEYS):
    detected = []
    view_names = ["Front", "Back", "Side"]
    detected.append(f"View: {view_names[np.argmax(agg_probs[0:3])]}")
    detected.append("Female" if agg_probs[3] > agg_probs[4] else "Male")
    if agg_probs[5] > 0.40: detected.append("Age_Child")
    elif agg_probs[7] > 0.40: detected.append("Age_Senior")
    else: detected.append("Age_Adult")
    if agg_probs[8] > 0.25: detected.append("Bald")
    elif agg_probs[10] > 0.45: detected.append("Long_Hair")
    elif agg_probs[9] > 0.45: detected.append("Short_Hair")
    if agg_probs[23] > 0.40: detected.append("LongCoat")
    elif agg_probs[22] > 0.20: detected.append("Jacket")
    elif agg_probs[21] > 0.20: detected.append("Tshirt")
    elif agg_probs[24] > 0.20: detected.append("Logo")
    elif agg_probs[25] > 0.50: detected.append("Plaid")
    elif agg_probs[26] > 0.50: detected.append("Stripe")
    elif agg_probs[20] > agg_probs[19] and agg_probs[20] > 0.50: detected.append("LongSleeve")
    elif agg_probs[19] > 0.80: detected.append("ShortSleeve")
    if agg_probs[30] > 0.30: detected.append("Skirt_or_Dress")
    elif agg_probs[29] > 0.30: detected.append("Shorts")
    elif agg_probs[28] > 0.30: detected.append("Jeans")
    elif agg_probs[27] > 0.50: detected.append("Trousers")
    shoe_indices = [31, 32, 33, 34]
    best_shoe_idx = shoe_indices[np.argmax([agg_probs[i] for i in shoe_indices])]
    if agg_probs[best_shoe_idx] > 0.45: detected.append(schema_keys[best_shoe_idx])
    if agg_probs[11] > 0.70 and agg_probs[11] > agg_probs[15]: detected.append("Backpack")
    elif agg_probs[15] > 0.70: detected.append("ShoulderBag")
    hand_indices = [16, 17, 18]
    best_hand_idx = hand_indices[np.argmax([agg_probs[i] for i in hand_indices])]
    if agg_probs[best_hand_idx] > 0.70: detected.append(schema_keys[best_hand_idx])
    if agg_probs[12] > 0.50: detected.append("Hat")
    if agg_probs[13] > 0.15: detected.append("Glasses")
    if agg_probs[14] > 0.65: detected.append("Muffler_Scarf")
    return detected

class PedestrianState:
    def __init__(self, top_k=10, fps=30):
        self.top_k = top_k
        self.buffer = []
        self.locked_attrs = []
        self.is_collecting = True
        self.active_view = None
        self.view_history = deque(maxlen=3 * fps)

    def update(self, q_score, probs, schema_keys):
        views = ["Front", "Back", "Side"]
        current_view = views[np.argmax(probs[0:3])]
        self.view_history.append(current_view)

        if not self.is_collecting:
            view_counts = defaultdict(int)
            for v in self.view_history: view_counts[v] += 1
            for v, count in view_counts.items():
                if v != self.active_view and count >= 5:
                    self.is_collecting = True
                    self.buffer, self.view_history = [], deque(maxlen=self.view_history.maxlen)
                    break
        
        if self.is_collecting:
            self.buffer.append((q_score, probs))
            self.buffer.sort(key=lambda x: x[0], reverse=True)
            if len(self.buffer) > self.top_k: self.buffer = self.buffer[:self.top_k]
            
            if len(self.buffer) == self.top_k:
                agg_probs = np.mean([p for _, p in self.buffer], axis=0)
                self.locked_attrs = parse_attributes(agg_probs, schema_keys)
                self.active_view = views[np.argmax(agg_probs[0:3])]
                self.is_collecting = False
                self.view_history.clear()

        if self.locked_attrs: return self.locked_attrs, len(self.buffer)
        return parse_attributes(np.mean([p for _, p in self.buffer], axis=0) if self.buffer else probs, schema_keys), len(self.buffer)

def draw_pedestrian_data(frame, track_id, bbox, display_attrs, buf_count, top_k):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"ID: {track_id} (Opt: {buf_count}/{top_k})", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    y_text = y1 + 15
    for attr in display_attrs:
        cv2.putText(frame, attr, (x2 + 6, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        y_text += 20

# --- FLASK WEB SERVER ---
app = Flask(__name__)
output_frame, lock = None, threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>PAR Inference Control Center</title>
<style>
    body { font-family: 'Segoe UI', sans-serif; background-color: #121212; color: #e0e0e0; text-align: center; margin: 0; }
    header { background-color: #1e1e1e; padding: 20px; border-bottom: 2px solid #00ffcc; }
    h1 { margin: 0; color: #00ffcc; letter-spacing: 1.5px; }
    .video-container { margin: 30px auto; max-width: 900px; border: 4px solid #333; border-radius: 12px; overflow: hidden; background: #000; }
    img { width: 100%; height: auto; display: block; }
</style></head><body>
<header><h1>PAR CONTROL CENTER (PURE ONNX)</h1></header>
<div class="video-container"><img src="{{ url_for('video_feed') }}" alt="Live Inference Stream Connecting..."></div>
</body></html>
"""
@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

def generate_web_stream():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            frame_data = output_frame
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- INFERENCE DAEMON ---
def run_inference_engine(args):
    global output_frame, lock
    
    if not os.path.exists(args.par_model) or not os.path.exists(args.yolo_model):
        logging.error("Model files missing. Ensure both ONNX models are present.")
        sys.exit(1)
        
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        logging.info("Loading Pure ONNX Architecture...")
        yolo_session = ort.InferenceSession(args.yolo_model, providers=providers)
        par_session = ort.InferenceSession(args.par_model, providers=providers)
        yolo_input_name, par_input_name = yolo_session.get_inputs()[0].name, par_session.get_inputs()[0].name
        
        cap = cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
        
        byte_tracker = sv.ByteTrack(track_activation_threshold=0.4, frame_rate=int(fps))
        track_states = defaultdict(lambda: PedestrianState(top_k=10, fps=int(fps)))
        missing_tracks = defaultdict(int)
        
        logging.info(f"Connected to RTSP stream: {args.source}")
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Stream dropped. Reconnecting...")
                time.sleep(2)
                cap = cv2.VideoCapture(args.source)
                byte_tracker = sv.ByteTrack(track_thresh=0.4, frame_rate=int(fps))
                track_states.clear()
                missing_tracks.clear()
                continue
                
            frame_idx += 1
            current_ids_in_frame, batch_tensors, batch_metadata = set(), [], []
            
            # YOLO Preprocessing & ONNX Execution
            input_tensor, r, dw, dh = preprocess_yolo(frame)
            outputs = yolo_session.run(None, {yolo_input_name: input_tensor})[0]
            boxes, scores = postprocess_yolo(outputs, r, dw, dh)
            
            if len(boxes) > 0:
                detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=np.zeros(len(boxes), dtype=int))
                tracked_detections = byte_tracker.update_with_detections(detections)
                
                if tracked_detections.tracker_id is not None:
                    for box, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                        track_id = int(track_id)
                        current_ids_in_frame.add(track_id)
                        
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        bbox_height, bbox_width = y2 - y1, x2 - x1
                        if bbox_height > 25 and (ped_crop := frame[y1:y2, x1:x2]).size > 0:
                            batch_tensors.append(preprocess_crop(ped_crop))
                            batch_metadata.append((track_id, (x1, y1, x2, y2), bbox_height * bbox_width))
            
            # PAR ONNX Execution
            if batch_tensors:
                outputs = par_session.run(None, {par_input_name: np.stack(batch_tensors)})
                batch_probs = 1 / (1 + np.exp(-outputs[0]))
                
                for meta, probs in zip(batch_metadata, batch_probs):
                    t_id, bbox, q_score = meta
                    display_attrs, buf_count = track_states[t_id].update(q_score, probs, SCHEMA_KEYS)
                    draw_pedestrian_data(frame, t_id, bbox, display_attrs, buf_count, 10)

            # Cleanup Stale Tracks
            stale_ids = set(track_states.keys()) - current_ids_in_frame
            for stale_id in stale_ids:
                missing_tracks[stale_id] += 1
                if missing_tracks[stale_id] > 30:
                    del track_states[stale_id]
                    del missing_tracks[stale_id]
            for active_id in current_ids_in_frame:
                missing_tracks.pop(active_id, None)

            # Blast to Web Server
            ret_enc, buffer = cv2.imencode('.jpg', frame)
            if ret_enc:
                with lock:
                    output_frame = buffer.tobytes()
            
            if frame_idx % 100 == 0:
                logging.info(f"Tracking {len(current_ids_in_frame)} pedestrians via Dual ONNX.")
                
    except Exception as e:
        logging.error(f"Fatal inference error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    args = parse_args()
    threading.Thread(target=run_inference_engine, args=(args,), daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)