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
from flask import Flask, Response, render_template_string, request, jsonify, send_from_directory

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

# --- WEB ROUTES & HTML ---
COMMON_CSS = """
    body { font-family: 'Segoe UI', sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; display: flex; flex-direction: column; align-items: center;}
    header { width: 100%; background-color: #1e1e1e; padding: 15px 20px; border-bottom: 2px solid #00ffcc; text-align: center; box-sizing: border-box; }
    h1 { margin: 0 0 10px 0; color: #00ffcc; letter-spacing: 1.5px; font-size: 1.5em; }
    nav a { color: #aaa; text-decoration: none; font-weight: bold; margin: 0 15px; padding: 5px 10px; border-radius: 4px; transition: 0.3s; }
    nav a:hover, nav a.active { background: #333; color: #00ffcc; }
"""

HTML_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>PAR Control Center</title>
<style>
""" + COMMON_CSS + """
    .layout { display: flex; max-width: 1400px; width: 100%; gap: 20px; margin-top: 20px; padding: 0 20px; box-sizing: border-box; }
    .video-container { flex: 2; border: 4px solid #333; border-radius: 12px; overflow: hidden; background: #000; min-width: 640px; }
    .video-container img { width: 100%; height: auto; display: block; }
    .controls { flex: 1; background: #1e1e1e; padding: 20px; border-radius: 12px; border: 1px solid #333; min-width: 300px; }
    .control-group { margin-bottom: 20px; }
    label { display: block; margin-bottom: 8px; color: #aaa; font-size: 0.9em; }
    input[type="number"], input[type="range"], select { width: 100%; background: #2a2a2a; border: 1px solid #444; color: #fff; padding: 8px; border-radius: 4px; box-sizing: border-box;}
    button { width: 100%; padding: 12px; font-size: 1.1em; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; transition: 0.3s; }
    .btn-off { background: #ff4444; color: white; }
    .btn-on { background: #00ffcc; color: #000; }
    .stats { margin-top: 30px; padding-top: 20px; border-top: 1px solid #333; }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 1.1em; }
    
    .log-panel { max-width: 1400px; width: 100%; margin: 20px auto; background: #1e1e1e; padding: 20px; border-radius: 12px; border: 1px solid #333; box-sizing: border-box; }
    .log-panel h3 { margin: 0 0 15px 0; color: #00ffcc; font-size: 1.2em; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .table-container { max-height: 250px; overflow-y: auto; }
    .log-table { width: 100%; text-align: left; border-collapse: collapse; font-size: 0.95em; }
    .log-table th { padding: 10px; color: #888; border-bottom: 2px solid #333; position: sticky; top: 0; background: #1e1e1e; }
    .log-table td { padding: 12px 10px; border-bottom: 1px solid #2a2a2a; color: #ccc; }
    .log-table tr:hover td { background: #252525; }
    .badge { background: #333; color: #00ffcc; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }
</style></head><body>
<header>
    <h1>PAR AI STUDIO</h1>
    <nav><a href="/" class="active">Live Engine</a><a href="/review">Data Review</a></nav>
</header>
<div class="layout">
    <div class="video-container"><img src="/video_feed" alt="Live Stream"></div>
    <div class="controls">
        <div class="control-group"><button id="toggleExtract" class="btn-off" onclick="toggleExtraction()">Extraction: OFF</button></div>
        <div class="control-group"><label>Extraction Rate: <span id="rateVal">100</span>%</label><input type="range" id="rateInput" min="1" max="100" value="100" onchange="updateConfig()"></div>
        <div class="control-group"><label>Max Samples (Cap)</label><input type="number" id="maxInput" value="500" onchange="updateConfig()"></div>
        
        <!-- NEW MEMORY RETENTION DROPDOWN -->
        <div class="control-group">
            <label>Memory Retention (ReID):</label>
            <select id="retentionSelect" onchange="updateConfig()">
                <option value="12">12 Hours</option>
                <option value="168">7 Days</option>
                <option value="720">30 Days</option>
                <option value="0">Forever (No Purge)</option>
            </select>
        </div>
        
        <div class="stats"><div class="stat-row"><span>Samples Collected:</span> <strong id="currentSamples" style="color:#00ffcc">0</strong></div></div>
    </div>
</div>

<div class="log-panel">
    <h3>Live Event Log</h3>
    <div class="table-container">
        <table class="log-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Track ID</th>
                    <th>Primary Attribute</th>
                    <th>Entrance</th>
                    <th>Exit</th>
                    <th>Duration</th>
                </tr>
            </thead>
            <tbody id="logBody">
                <!-- JS Populated -->
            </tbody>
        </table>
    </div>
</div>

<script>
    let isExtracting = false;
    function fetchState() {
        fetch('/api/config').then(r => r.json()).then(data => {
            isExtracting = data.extract_enabled;
            
            // Only update input fields if the user IS NOT currently clicking/typing in them
            if (document.activeElement.id !== 'rateInput') {
                document.getElementById('rateInput').value = data.extraction_rate;
            }
            if (document.activeElement.id !== 'maxInput') {
                document.getElementById('maxInput').value = data.max_samples;
            }
            if (document.activeElement.id !== 'retentionSelect' && data.retention_hours !== undefined) {
                document.getElementById('retentionSelect').value = data.retention_hours;
            }
            
            // Always update display text
            document.getElementById('rateVal').innerText = data.extraction_rate;
            document.getElementById('currentSamples').innerText = data.current_samples;
            
            const btn = document.getElementById('toggleExtract');
            btn.className = isExtracting ? 'btn-on' : 'btn-off';
            btn.innerText = isExtracting ? 'Extraction: ACTIVE' : 'Extraction: OFF';
        });
        
        fetch('/api/logs').then(r => r.json()).then(data => {
            document.getElementById('logBody').innerHTML = data.map(log => `
                <tr>
                    <td>${log.timestamp}</td>
                    <td><span class="badge">${log.id}</span></td>
                    <td>${log.inference}</td>
                    <td>${log.entrance}</td>
                    <td>${log.exit}</td>
                    <td>${log.duration}</td>
                </tr>
            `).join('');
        });
    }
    
    function updateConfig() {
        fetch('/api/config', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                extract_enabled: isExtracting, 
                extraction_rate: document.getElementById('rateInput').value, 
                max_samples: document.getElementById('maxInput').value,
                retention_hours: document.getElementById('retentionSelect').value
            })
        }).then(fetchState);
    }
    
    function toggleExtraction() { isExtracting = !isExtracting; updateConfig(); }
    setInterval(fetchState, 2000);
    fetchState();
</script>
</body></html>
"""

REVIEW_HTML_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>PAR Data Review</title>
<style>
""" + COMMON_CSS + """
    .layout { display: flex; max-width: 1400px; width: 100%; gap: 20px; margin-top: 20px; padding: 0 20px; box-sizing: border-box; align-items: flex-start; }
    .images-panel { flex: 2; background: #1e1e1e; padding: 20px; border-radius: 12px; border: 1px solid #333; text-align: center; }
    .image-row { display: flex; justify-content: center; gap: 10px; margin-top: 15px; }
    .image-row img { max-height: 400px; border-radius: 8px; border: 2px solid #444; }
    .editor-panel { flex: 1; background: #1e1e1e; padding: 20px; border-radius: 12px; border: 1px solid #333; min-width: 300px; }
    h3 { margin-top: 0; color: #fff; border-bottom: 1px solid #444; padding-bottom: 10px; }
    .hints { background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 0.9em; }
    .hints span { color: #00ffcc; font-weight: bold; }
    
    /* NEW TAGGING STYLES */
    .tag-section { background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #444; }
    .tag-inputs { display: flex; gap: 10px; margin-top: 10px; }
    .tag-inputs input[type="text"] { background: #1e1e1e; border: 1px solid #444; color: #fff; padding: 10px; border-radius: 4px; box-sizing: border-box; }
    #tag-id { width: 120px; text-align: center; background: #333; cursor: not-allowed; }
    
    .checkbox-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; max-height: 400px; overflow-y: auto; }
    .checkbox-grid label { display: flex; align-items: center; cursor: pointer; color: #ccc; }
    .checkbox-grid input { margin-right: 8px; width: 16px; height: 16px; }
    .actions { display: flex; gap: 10px; }
    button { flex: 1; padding: 12px; font-size: 1.1em; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; transition: 0.3s; }
    .btn-save { background: #00ffcc; color: #000; }
    .btn-save:hover { background: #00ccaa; }
    .btn-del { background: #ff4444; color: #fff; }
    .btn-del:hover { background: #cc0000; }
    .empty-state { padding: 50px; text-align: center; color: #888; font-size: 1.2em; width: 100%; }
</style></head><body>
<header>
    <h1>PAR AI STUDIO</h1>
    <nav><a href="/">Live Engine</a><a href="/review" class="active">Data Review</a></nav>
</header>
<div id="app" class="layout">
    <div class="empty-state">Loading next sample...</div>
</div>

<script>
    let currentData = null;
    const schemaKeys = {{ schema_keys | safe }};
    
    function loadNext() {
        fetch('/api/review/next').then(r => r.json()).then(data => {
            if(data.message) {
                document.getElementById('app').innerHTML = `<div class="empty-state">🎉 ${data.message}</div>`;
                return;
            }
            currentData = data;
            renderUI();
        });
    }

    function renderUI() {
        const sortedScores = Object.entries(currentData.raw_scores).sort((a,b) => b[1] - a[1]).slice(0,3);
        let hintsHtml = sortedScores.map(([attr, score]) => `<div>${attr}: <span>${(score*100).toFixed(1)}%</span></div>`).join('');
        
        let checkboxesHtml = schemaKeys.map(key => {
            const isChecked = currentData.auto_labels.includes(key) ? 'checked' : '';
            return `<label><input type="checkbox" id="chk_${key}" value="${key}" ${isChecked}> ${key}</label>`;
        }).join('');

        let imagesHtml = '';
        if(currentData.images.front) imagesHtml += `<img src="${currentData.images.front}" title="Front">`;
        if(currentData.images.side) imagesHtml += `<img src="${currentData.images.side}" title="Side">`;
        if(currentData.images.back) imagesHtml += `<img src="${currentData.images.back}" title="Back">`;

        document.getElementById('app').innerHTML = `
            <div class="images-panel">
                <h3>Folder: ${currentData.folder_name}</h3>
                <div class="image-row">${imagesHtml}</div>
            </div>
            <div class="editor-panel">
                <!-- NEW TAGGING UI -->
                <div class="tag-section">
                    <h3 style="margin:0; border:none; padding:0; color:#00ffcc; font-size:1.1em;">Tag Identity</h3>
                    <label style="margin-top:5px;">Assign a custom name to this person across all cameras.</label>
                    <div class="tag-inputs">
                        <input type="text" id="tag-id" value="${currentData.track_id}" title="Global ID (Locked)" readonly>
                        <input type="text" id="tag-label" placeholder="e.g. John (IT)">
                        <button class="btn-save" style="padding: 10px;" onclick="tagIdentity()">Tag Person</button>
                    </div>
                </div>
                
                <h3>Top 3 Confidences</h3>
                <div class="hints">${hintsHtml}</div>
                <h3>Adjust Labels</h3>
                <div class="checkbox-grid">${checkboxesHtml}</div>
                <div class="actions">
                    <button class="btn-del" onclick="deleteSample()">Trash (Del)</button>
                    <button class="btn-save" onclick="saveSample()">Approve & Next (Enter)</button>
                </div>
            </div>
        `;
    }

    function tagIdentity() {
        const rawId = document.getElementById("tag-id").value;
        const label = document.getElementById("tag-label").value;
        
        // Extract purely the numerical ID from strings like "G-8" or "Minerva (#3)"
        const match = String(rawId).match(/\d+/);
        const id = match ? match[0] : null;
        
        if(!id) {
            alert("Could not extract a numerical ID from: " + rawId);
            return;
        }

        if(!label) {
            alert("Please enter a custom name before saving.");
            return;
        }
        
        fetch('/api/tag', {
            method: 'POST', 
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ id: id, label: label })
        })
        .then(r => r.json())
        .then(data => {
            alert(data.message);
            document.getElementById("tag-label").value = ""; // Clear input after tagging
        });
    }

    function saveSample() {
        if(!currentData) return;
        const selected = schemaKeys.filter(key => document.getElementById(`chk_${key}`).checked);
        fetch('/api/review/save', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ folder_name: currentData.folder_name, auto_labels: selected })
        }).then(loadNext);
    }

    function deleteSample() {
        if(!currentData) return;
        fetch('/api/review/delete', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ folder_name: currentData.folder_name })
        }).then(loadNext);
    }

    document.addEventListener('keydown', function(e) {
        if(e.key === 'Enter') saveSample();
        if(e.key === 'Delete') deleteSample();
    });

    loadNext();
</script>
</body></html>
"""

# --- ROUTES ---

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/review')
def review(): return render_template_string(REVIEW_HTML_TEMPLATE, schema_keys=json.dumps(SCHEMA_KEYS))

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
def video_feed(): return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/config', methods=['POST', 'GET'])
def config_route():
    if request.method == 'POST':
        data = request.json
        # Strictly cast all incoming web data to correct Python types
        if 'extract_enabled' in data: engine_config['extract_enabled'] = bool(data['extract_enabled'])
        if 'extraction_rate' in data: engine_config['extraction_rate'] = int(data['extraction_rate'])
        if 'max_samples' in data: engine_config['max_samples'] = int(data['max_samples'])
        if 'retention_hours' in data: engine_config['retention_hours'] = int(data['retention_hours'])
    
    # Always send back current config
    engine_config['current_samples'] = len(glob.glob(os.path.join(EXPORT_DIR, "track_*")))
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