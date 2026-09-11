import os
import sys
import pickle
import json
import cv2
import numpy as np
import glob
import time
import logging
import threading
import shutil
from datetime import datetime
from flask import Flask, Response, render_template, request, jsonify, send_from_directory

# --- SCHEMA & CONFIG ---
SCHEMA_KEYS = [
    "View_Front", "View_Back", "View_Side", "Female", "Male",
    "Age_Child", "Age_Adult", "Age_Senior", "Bald", "Short_Hair", "Long_Hair",
    "Backpack", "Hat", "Glasses", "Muffler_Scarf", "ShoulderBag", 
    "HandBag", "PlasticBag", "CarryingOther", "ShortSleeve", "LongSleeve", 
    "Tshirt", "Jacket", "LongCoat", "Logo", "Plaid", "Stripe",
    "Trousers", "Jeans", "Shorts", "Skirt_or_Dress",
    "Boots", "Sneakers", "LeatherShoes", "Sandals"
]

EXPORT_DIR = "/app/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

engine_config = {
    "extract_enabled": False,
    "max_samples": 500,
    "extraction_rate": 100,
    "retention_hours": 168, # Default to 7 days
    "current_samples": len(glob.glob(os.path.join(EXPORT_DIR, "track_*")))
}

# --- EVENT LOGGER ---
event_logs = []
MAX_LOGS = 50 

def add_log_event(event_dict):
    event_logs.insert(0, event_dict)
    if len(event_logs) > MAX_LOGS:
        event_logs.pop()

# --- FLASK SETUP & STATE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

output_frame = None
lock = threading.Lock()
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# --- 🔒 SECURITY: API AUTHENTICATION ---
# Set a default secure token in .env, or generate a random one on startup
API_TOKEN = os.getenv("API_TOKEN", "default_secure_token_change_me")

@app.before_request
def verify_token():
    # 1. Define routes that DO NOT need the token (UI, Streams, and Images)
    exempt_exact = [
        '/', 
        '/review', 
        '/video_feed',
        '/api/config',  # Let the UI read the config
        '/api/logs'     # Let the UI read the event logs
    ]
    
    # Let the request pass through if it's UI or an exported image
    if request.path in exempt_exact or request.path.startswith('/exports/'):
        return None  
        
    # 2. Check for the token in the headers for all API endpoints
    token = request.headers.get('X-API-Token')
    
    if token != API_TOKEN:
        return jsonify({
            "error": "Unauthorized Access",
            "message": "A valid X-API-Token header is required."
        }), 401
# ---------------------------------------

def update_frame(frame_bytes):
    global output_frame, lock
    with lock:
        output_frame = frame_bytes

# --- DATA PARSING & EXPORT ---
def parse_attributes(agg_probs, schema_keys=SCHEMA_KEYS):
    detected = []
    if agg_probs is None or len(agg_probs) < len(schema_keys):
        return detected
        
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

def export_track_data(track_id, state):
    if not hasattr(state, 'best_crops') or not state.best_crops:
        return False
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    track_dir = os.path.join(EXPORT_DIR, f"track_{track_id}_{timestamp}")
    os.makedirs(track_dir, exist_ok=True)
    
    all_probs = [data['probs'] for data in state.best_crops.values()]
    agg_probs = np.mean(all_probs, axis=0)
    
    auto_labels = parse_attributes(agg_probs, SCHEMA_KEYS)
    raw_scores = {SCHEMA_KEYS[i]: round(float(agg_probs[i]), 4) for i in range(len(SCHEMA_KEYS))}
    
    for view, data in state.best_crops.items():
        cv2.imwrite(os.path.join(track_dir, f"{view.lower()}.jpg"), data['crop'])
        
    metadata = {
        "track_id": track_id,
        "timestamp": timestamp,
        "auto_labels": auto_labels,
        "raw_scores": raw_scores,
        "reviewed": False
    }
    
    with open(os.path.join(track_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    logging.info(f"💾 Exported training material for Track {track_id}")
    return True

# --- ROUTES ---

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/review')
def review(): 
    return render_template('review.html', schema_keys=json.dumps(SCHEMA_KEYS))

@app.route('/exports/<path:filename>')
def serve_exports(filename): 
    return send_from_directory(EXPORT_DIR, filename)

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
def video_feed(): 
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/config', methods=['POST', 'GET'])
def config_route():
    global engine_config  
    
    if request.method == 'POST':
        data = request.json
        if 'extract_enabled' in data: engine_config['extract_enabled'] = bool(data['extract_enabled'])
        if 'extraction_rate' in data: engine_config['extraction_rate'] = int(data['extraction_rate'])
        if 'max_samples' in data: engine_config['max_samples'] = int(data['max_samples'])
        if 'retention_hours' in data: engine_config['retention_hours'] = int(data['retention_hours'])
        if 'record_events' in data: engine_config['record_events'] = bool(data['record_events'])
        if 'sleep_mode' in data: engine_config['sleep_mode'] = bool(data['sleep_mode'])
    
    engine_config["current_samples"] = len(glob.glob(os.path.join(EXPORT_DIR, "track_*")))
    return jsonify(engine_config)

@app.route('/api/tag', methods=['POST'])
def tag_identity():
    data = request.json
    global_id = int(data.get('id'))
    label = data.get('label')
    
    gallery_path = "exports/reid_gallery.pkl"
    if os.path.exists(gallery_path):
        with open(gallery_path, "rb") as f:
            gallery = pickle.load(f)
        
        if global_id in gallery:
            gallery[global_id]['label'] = label
            with open(gallery_path, "wb") as f:
                pickle.dump(gallery, f)
            return jsonify({"status": "success", "message": f"Successfully tagged ID G-{global_id} as '{label}'"})
        
    return jsonify({"status": "error", "message": "ID not found in memory vault."}), 404

@app.route('/api/logs', methods=['GET'])
def api_logs():
    return jsonify(event_logs)

@app.route('/api/review/next', methods=['GET'])
def api_review_next():
    track_dirs = glob.glob(os.path.join(EXPORT_DIR, "track_*"))
    for d in track_dirs:
        meta_path = os.path.join(d, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                data = json.load(f)
            if not data.get("reviewed", False):
                images = {}
                for view in ["front.jpg", "back.jpg", "side.jpg"]:
                    if os.path.exists(os.path.join(d, view)):
                        images[view.split('.')[0]] = f"/exports/{os.path.basename(d)}/{view}"
                data['images'] = images
                data['folder_name'] = os.path.basename(d)
                return jsonify(data)
    return jsonify({"message": "All caught up! No pending data."})

@app.route('/api/review/save', methods=['POST'])
def api_review_save():
    req = request.json
    meta_path = os.path.join(EXPORT_DIR, req["folder_name"], "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f: data = json.load(f)
        data["auto_labels"] = req["auto_labels"]
        data["reviewed"] = True
        with open(meta_path, "w") as f: json.dump(data, f, indent=4)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/api/review/delete', methods=['POST'])
def api_review_delete():
    req = request.json
    dir_path = os.path.join(EXPORT_DIR, req["folder_name"])
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

def start_web_server(port):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)