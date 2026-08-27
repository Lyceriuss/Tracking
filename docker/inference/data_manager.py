import os
import sys
import json
import cv2
import numpy as np
import glob
import time
import logging
import threading
from datetime import datetime
from flask import Flask, Response, render_template_string, request, jsonify

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
    "current_samples": len(glob.glob(os.path.join(EXPORT_DIR, "track_*")))
}

# --- FLASK SETUP & STATE ---
app = Flask(__name__)
output_frame = None
lock = threading.Lock()
logging.getLogger('werkzeug').setLevel(logging.ERROR)

def update_frame(frame_bytes):
    global output_frame, lock
    with lock:
        output_frame = frame_bytes

# --- DATA PARSING & EXPORT ---
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

# --- WEB ROUTES & HTML ---
HTML_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>PAR Control Center</title>
<style>
    body { font-family: 'Segoe UI', sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; display: flex; flex-direction: column; align-items: center;}
    header { width: 100%; background-color: #1e1e1e; padding: 20px; border-bottom: 2px solid #00ffcc; text-align: center; box-sizing: border-box; }
    h1 { margin: 0; color: #00ffcc; letter-spacing: 1.5px; }
    .layout { display: flex; max-width: 1400px; width: 100%; gap: 20px; margin-top: 20px; padding: 0 20px; box-sizing: border-box; }
    .video-container { flex: 2; border: 4px solid #333; border-radius: 12px; overflow: hidden; background: #000; min-width: 640px; }
    .video-container img { width: 100%; height: auto; display: block; }
    .controls { flex: 1; background: #1e1e1e; padding: 20px; border-radius: 12px; border: 1px solid #333; min-width: 300px; }
    .control-group { margin-bottom: 20px; }
    label { display: block; margin-bottom: 8px; color: #aaa; font-size: 0.9em; }
    input[type="number"], input[type="range"] { width: 100%; background: #2a2a2a; border: 1px solid #444; color: #fff; padding: 8px; border-radius: 4px; box-sizing: border-box;}
    button { width: 100%; padding: 12px; font-size: 1.1em; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; transition: 0.3s; }
    .btn-off { background: #ff4444; color: white; }
    .btn-on { background: #00ffcc; color: #000; }
    .stats { margin-top: 30px; padding-top: 20px; border-top: 1px solid #333; }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 1.1em; }
</style></head><body>
<header><h1>PAR DATA EXTRACTOR & CONTROL</h1></header>
<div class="layout">
    <div class="video-container">
        <img src="{{ url_for('video_feed') }}" alt="Live Stream">
    </div>
    <div class="controls">
        <div class="control-group">
            <button id="toggleExtract" class="btn-off" onclick="toggleExtraction()">Extraction: OFF</button>
        </div>
        <div class="control-group">
            <label>Extraction Rate: <span id="rateVal">100</span>%</label>
            <input type="range" id="rateInput" min="1" max="100" value="100" onchange="updateConfig()">
        </div>
        <div class="control-group">
            <label>Max Samples (Cap)</label>
            <input type="number" id="maxInput" value="500" onchange="updateConfig()">
        </div>
        <div class="stats">
            <div class="stat-row"><span>Samples Collected:</span> <strong id="currentSamples" style="color:#00ffcc">0</strong></div>
        </div>
    </div>
</div>
<script>
    let isExtracting = false;
    function fetchConfig() {
        fetch('/api/config').then(r => r.json()).then(data => {
            isExtracting = data.extract_enabled;
            document.getElementById('rateInput').value = data.extraction_rate;
            document.getElementById('rateVal').innerText = data.extraction_rate;
            document.getElementById('maxInput').value = data.max_samples;
            document.getElementById('currentSamples').innerText = data.current_samples;
            const btn = document.getElementById('toggleExtract');
            btn.className = isExtracting ? 'btn-on' : 'btn-off';
            btn.innerText = isExtracting ? 'Extraction: ACTIVE' : 'Extraction: OFF';
        });
    }
    function updateConfig() {
        fetch('/api/config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                extract_enabled: !isExtracting, 
                extraction_rate: document.getElementById('rateInput').value,
                max_samples: document.getElementById('maxInput').value
            })
        }).then(fetchConfig);
    }
    function toggleExtraction() { isExtracting = !isExtracting; updateConfig(); }
    setInterval(fetchConfig, 2000);
    fetchConfig();
</script>
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

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        data = request.json
        if 'extract_enabled' in data: engine_config['extract_enabled'] = data['extract_enabled']
        if 'max_samples' in data: engine_config['max_samples'] = int(data['max_samples'])
        if 'extraction_rate' in data: engine_config['extraction_rate'] = int(data['extraction_rate'])
    return jsonify(engine_config)

def start_web_server(port):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)