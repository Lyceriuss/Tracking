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

import data_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("/app/inference_engine.log"), logging.StreamHandler(sys.stdout)]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Pure ONNX Inference Engine")
    parser.add_argument("--source", type=str, default=os.environ.get("STREAM_SOURCE", "rtsp://streamer_1:8554/live/stream"))
    parser.add_argument("--par-model", type=str, default=os.environ.get("MODEL_PATH", "baseline_v4_prod.onnx"))
    parser.add_argument("--yolo-model", type=str, default="yolo11n.onnx")
    # --- NEW: ReID Model Argument ---
    parser.add_argument("--reid-model", type=str, default=os.environ.get("REID_MODEL_PATH", "osnet_x0_25_msmt17.onnx"))
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()

def preprocess_yolo(img, input_size=(640, 640)):
    shape = img.shape[:2]
    r = min(input_size[0] / shape[0], input_size[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = input_size[1] - new_unpad[0], input_size[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad: img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(img_chw, axis=0), r, dw, dh

def postprocess_yolo(outputs, r, dw, dh, conf_threshold=0.4, iou_threshold=0.4):
    predictions = np.squeeze(outputs).T
    scores = predictions[:, 4]  
    mask = scores > conf_threshold
    valid_preds, valid_scores = predictions[mask], scores[mask]
    if len(valid_preds) == 0: return np.empty((0, 4)), np.empty(0)
    cx, cy, w, h = valid_preds[:, 0], valid_preds[:, 1], valid_preds[:, 2], valid_preds[:, 3]
    x1, y1 = (cx - w / 2 - dw) / r, (cy - h / 2 - dh) / r
    x2, y2 = (cx + w / 2 - dw) / r, (cy + h / 2 - dh) / r
    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes, valid_scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) == 0: return np.empty((0, 4)), np.empty(0)
    indices = indices.flatten()
    return np.stack([x1[indices], y1[indices], x2[indices], y2[indices]], axis=1), valid_scores[indices]

def preprocess_crop(crop_bgr):
    # Perfect shape for both PAR and OSNet (128x256)
    resized = cv2.resize(crop_bgr, (128, 256))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.transpose((img - mean) / std, (2, 0, 1))

def compute_similarity(vec1, vec2):
    """Cosine Similarity: 1.0 is exact match, 0.0 is entirely different."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def draw_pedestrian_data(frame, display_id, bbox, display_attrs, buf_count, top_k):
    x1, y1, x2, y2 = bbox
    color = (0, 165, 255) if "Scan" in str(display_id) else (0, 255, 0)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID: {display_id} (Opt: {buf_count}/{top_k})", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    
    y_text = y1 + 15
    for attr in display_attrs:
        cv2.putText(frame, attr, (x2 + 6, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        y_text += 20

class PedestrianState:
    def __init__(self, top_k=10, fps=30):
        self.top_k = top_k
        self.buffer = []
        self.locked_attrs = []
        self.is_collecting = True
        self.active_view = None
        self.view_history = deque(maxlen=3 * fps)
        self.best_crops = {} 
        
        self.entry_time = time.time()
        self.last_seen_time = time.time()
        self.entry_point = None
        self.last_point = None

    def update(self, q_score, probs, schema_keys, crop_bgr):
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
            self.buffer.append((q_score, probs, crop_bgr.copy()))
            self.buffer.sort(key=lambda x: x[0], reverse=True)
            if len(self.buffer) > self.top_k: self.buffer = self.buffer[:self.top_k]
            
            if len(self.buffer) == self.top_k:
                agg_probs = np.mean([p for _, p, _ in self.buffer], axis=0)
                self.locked_attrs = data_manager.parse_attributes(agg_probs, schema_keys)
                self.active_view = views[np.argmax(agg_probs[0:3])]
                
                best_q, _, best_crop = self.buffer[0]
                if self.active_view not in self.best_crops or best_q > self.best_crops[self.active_view]['score']:
                    self.best_crops[self.active_view] = { 
                        'score': best_q, 
                        'crop': best_crop, 
                        'probs': agg_probs 
                    }
                self.is_collecting = False
                self.view_history.clear()

        if self.locked_attrs: return self.locked_attrs, len(self.buffer)
        
        fallback_probs = np.mean([p for _, p, _ in self.buffer], axis=0) if self.buffer else probs
        return data_manager.parse_attributes(fallback_probs, schema_keys), len(self.buffer)


def run_inference_engine(args):
    try:
        providers = ['CPUExecutionProvider']
        yolo_session = ort.InferenceSession(args.yolo_model, providers=providers)
        par_session = ort.InferenceSession(args.par_model, providers=providers)
        
        # --- NEW: Initialize OSNet ReID Session ---
        reid_session = ort.InferenceSession(args.reid_model, providers=providers)
        reid_input_name = reid_session.get_inputs()[0].name
        # ------------------------------------------

        yolo_input_name, par_input_name = yolo_session.get_inputs()[0].name, par_session.get_inputs()[0].name
        
        cap = cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
        
        byte_tracker = sv.ByteTrack(track_activation_threshold=0.4, frame_rate=int(fps))
        track_states = defaultdict(lambda: PedestrianState(top_k=10, fps=int(fps)))
        missing_tracks = defaultdict(int)
        
        reid_gallery = {}       
        track_to_global = {}    
        next_global_id = 1
        
        logging.info(f"Connected to RTSP stream: {args.source}")
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(2)
                cap = cv2.VideoCapture(args.source)
                byte_tracker, track_states, missing_tracks = sv.ByteTrack(track_thresh=0.4, frame_rate=int(fps)), defaultdict(lambda: PedestrianState(top_k=10, fps=int(fps))), defaultdict(int)
                continue
                
            frame_idx += 1
            current_ids_in_frame, batch_tensors, batch_metadata = set(), [], []
            
            # YOLO 
            input_tensor, r, dw, dh = preprocess_yolo(frame)
            outputs = yolo_session.run(None, {yolo_input_name: input_tensor})[0]
            boxes, scores = postprocess_yolo(outputs, r, dw, dh)
            
            if len(boxes) > 0:
                tracked_detections = byte_tracker.update_with_detections(sv.Detections(xyxy=boxes, confidence=scores, class_id=np.zeros(len(boxes), dtype=int)))
                if tracked_detections.tracker_id is not None:
                    for box, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                        track_id = int(track_id)
                        current_ids_in_frame.add(track_id)
                        x1, y1, x2, y2 = map(int, box)
                        
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if track_states[track_id].entry_point is None:
                            track_states[track_id].entry_point = (cx, cy)
                        track_states[track_id].last_point = (cx, cy)
                        track_states[track_id].last_seen_time = time.time()

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        bbox_height, bbox_width = y2 - y1, x2 - x1
                        if bbox_height > 25 and (ped_crop := frame[y1:y2, x1:x2].copy()).size > 0:
                            batch_tensors.append(preprocess_crop(ped_crop))
                            batch_metadata.append((track_id, (x1, y1, x2, y2), bbox_height * bbox_width, ped_crop))
            
            # PAR & REID
            if batch_tensors:
                outputs = par_session.run(None, {par_input_name: np.stack(batch_tensors)})
                batch_probs = 1 / (1 + np.exp(-outputs[0]))
                
                for meta, probs in zip(batch_metadata, batch_probs):
                    t_id, bbox, q_score, ped_crop = meta
                    display_attrs, buf_count = track_states[t_id].update(q_score, probs, data_manager.SCHEMA_KEYS, ped_crop)
                    
                    # --- TRUE AI REID MATCHING (TEMPORAL POOLING) ---
                    if track_states[t_id].locked_attrs and t_id not in track_to_global:
                        # 1. Extract all 10 buffered crops
                        crops = [item[2] for item in track_states[t_id].buffer]
                        
                        # 2. Preprocess and stack them into a single batch (Shape: 10 x 3 x 256 x 128)
                        reid_tensors = [preprocess_crop(c) for c in crops]
                        batch_tensor = np.stack(reid_tensors, axis=0) 
                        
                        # 3. Fix for hardcoded ONNX batch sizes (Pad to 16 if demanded)
                        expected_batch = reid_session.get_inputs()[0].shape[0]
                        if isinstance(expected_batch, int) and expected_batch > len(crops):
                            padding_needed = expected_batch - len(crops)
                            # Duplicate the last image to fill the empty slots
                            padding = np.repeat(batch_tensor[-1:], padding_needed, axis=0)
                            batch_tensor = np.concatenate([batch_tensor, padding], axis=0)
                        
                        # 4. Run all crops through OSNet simultaneously!
                        raw_outputs = reid_session.run(None, {reid_input_name: batch_tensor})[0]
                        
                        # 5. Average the valid 10 fingerprints to delete occlusion noise
                        valid_fingerprints = raw_outputs[:len(crops)]
                        avg_fingerprint = np.mean(valid_fingerprints, axis=0).flatten()
                        
                        # 6. Normalize the vector so cosine similarity math stays accurate
                        avg_fingerprint = avg_fingerprint / (np.linalg.norm(avg_fingerprint) + 1e-10)
                        
                        # Match against memory gallery
                        best_sim, best_id = 0.0, None
                        for g_id, g_emb in reid_gallery.items():
                            sim = compute_similarity(avg_fingerprint, g_emb)
                            if sim > best_sim:
                                best_sim, best_id = sim, g_id
                                
                        if best_sim > 0.80:
                            track_to_global[t_id] = best_id
                            reid_gallery[best_id] = 0.8 * reid_gallery[best_id] + 0.2 * avg_fingerprint
                        else:
                            track_to_global[t_id] = next_global_id
                            reid_gallery[next_global_id] = avg_fingerprint
                            next_global_id += 1
                    # ------------------------------------------------
                    
                    display_id = f"G-{track_to_global[t_id]}" if t_id in track_to_global else f"Scan-{t_id}"
                    draw_pedestrian_data(frame, display_id, bbox, display_attrs, buf_count, 10)

            # --- STALE TRACK CLEANUP & EXTRACTION GATEWAY ---
            stale_ids = set(track_states.keys()) - current_ids_in_frame
            for stale_id in stale_ids:
                missing_tracks[stale_id] += 1
                if missing_tracks[stale_id] > 30:
                    
                    state = track_states[stale_id]
                    duration = round(state.last_seen_time - state.entry_time, 1)
                    ent = state.entry_point or (0,0)
                    ext = state.last_point or (0,0)
                    
                    def get_zone(x, y, frame_w, frame_h):
                        distances = {"Left": x, "Right": frame_w - x, "Top": y, "Bottom": frame_h - y}
                        return min(distances, key=distances.get)
                            
                    frame_h, frame_w = frame.shape[:2]
                    primary = state.locked_attrs[1] if state.locked_attrs and len(state.locked_attrs) > 1 else "Pedestrian"
                    
                    final_log_id = track_to_global.get(stale_id, f"Unregistered-{stale_id}")
                    
                    data_manager.add_log_event({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "id": final_log_id,
                        "inference": primary,
                        "entrance": f"{get_zone(ent[0], ent[1], frame_w, frame_h)} {ent}",
                        "exit": f"{get_zone(ext[0], ext[1], frame_w, frame_h)} {ext}",
                        "duration": f"{duration}s"
                    })
                    
                    cfg = data_manager.engine_config
                    if cfg["extract_enabled"] and cfg["current_samples"] < cfg["max_samples"]:
                        if np.random.randint(1, 101) <= cfg["extraction_rate"]:
                            if data_manager.export_track_data(final_log_id, track_states[stale_id]):
                                cfg["current_samples"] += 1
                                
                    del track_states[stale_id]
                    del missing_tracks[stale_id]
                    
            for active_id in current_ids_in_frame: missing_tracks.pop(active_id, None)

            ret_enc, buffer = cv2.imencode('.jpg', frame)
            if ret_enc: data_manager.update_frame(buffer.tobytes())
                
    except Exception as e:
        logging.error(f"Fatal inference error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    args = parse_args()
    threading.Thread(target=run_inference_engine, args=(args,), daemon=True).start()
    data_manager.start_web_server(args.port)