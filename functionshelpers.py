import cv2
import torch
import numpy as np
import torchvision.transforms as T
from ultralytics import YOLO
from collections import defaultdict
import time
from collections import defaultdict, deque

# 1. Import your custom baseline architecture
from baseline_model import BestenSingle

# --- CONFIGURATION & SCHEMA ---
SCHEMA_KEYS = [
    "View_Front", "View_Back", "View_Side",
    "Female", "Male",
    "Age_Child", "Age_Adult", "Age_Senior",
    "Bald", "Short_Hair", "Long_Hair",
    "Backpack", "Hat", "Glasses", "Muffler_Scarf", "ShoulderBag", 
    "HandBag", "PlasticBag", "CarryingOther",
    "ShortSleeve", "LongSleeve", "Tshirt", "Jacket", "LongCoat", 
    "Logo", "Plaid", "Stripe",
    "Trousers", "Jeans", "Shorts", "Skirt_or_Dress",
    "Boots", "Sneakers", "LeatherShoes", "Sandals"
]

def load_tracking_models(checkpoint_path, device, num_classes=36):
    """Initializes and returns the YOLO tracker, Attribute Model, and image transforms."""
    print("Loading Attribute Model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    layer_config = checkpoint.get('layer_config', [(4, 64, 2), (4, 128, 2), (4, 256, 2), (4, 512, 2), (1, 1024, 2)])
    
    attr_model = BestenSingle(num_classes=num_classes, layer_config=layer_config).to(device)
    attr_model.load_state_dict(checkpoint['model_state_dict'])
    attr_model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading YOLO Tracker...")
    yolo_model = YOLO("yolo11n.pt") 
    
    return yolo_model, attr_model, transform

def parse_attributes(agg_probs, schema_keys=SCHEMA_KEYS):
  """Translates raw model probabilities for all 36 schema keys into structured,

  physically consistent pedestrian attributes using priority ladders and tuned
  thresholds.
  """
  detected = []

  # =========================================================
  # 1. VIEWPOINT (Indices 0, 1, 2) - Mutually Exclusive Argmax
  # =========================================================
  view_indices = [0, 1, 2]
  view_names = ["Front", "Back", "Side"]
  best_view_idx = np.argmax([agg_probs[i] for i in view_indices])
  detected.append(f"View: {view_names[best_view_idx]}")

  # =========================================================
  # 2. GENDER (Indices 3, 4) - Mutually Exclusive Argmax
  # =========================================================
  female_p, male_p = agg_probs[3], agg_probs[4]
  detected.append("Female" if female_p > male_p else "Male")

  # =========================================================
  # 3. AGE (Indices 5, 6, 7) - Priority Ladder (Rare Over Generic)
  # =========================================================
  # Child and Senior are rarer, so evaluate them first with lower thresholds
  if agg_probs[5] > 0.40:
    detected.append("Age_Child")
  elif agg_probs[7] > 0.40:
    detected.append("Age_Senior")
  else:
    detected.append("Age_Adult")

  # =========================================================
  # 4. HAIR STYLE (Indices 8, 9, 10) - Priority Ladder
  # =========================================================
  if agg_probs[8] > 0.25:
    detected.append("Bald")
  elif agg_probs[10] > 0.45:
    detected.append("Long_Hair")
  elif agg_probs[9] > 0.45:
    detected.append("Short_Hair")

  # =========================================================
  # 5. UPPER BODY & PATTERNS (Indices 19 - 26) - Specificity Ladder
  # =========================================================
  # Outerwear -> Tops -> Patterns -> Generic Sleeves
  if agg_probs[23] > 0.40:
    detected.append("LongCoat")
  elif agg_probs[22] > 0.20:
    detected.append("Jacket")
  elif agg_probs[21] > 0.20:
    detected.append("Tshirt")
  elif agg_probs[24] > 0.20:
    detected.append("Logo")
  elif agg_probs[25] > 0.50:
    detected.append("Plaid")
  elif agg_probs[26] > 0.50:
    detected.append("Stripe")
  elif agg_probs[20] > agg_probs[19] and agg_probs[20] > 0.50:
    detected.append("LongSleeve")
  elif agg_probs[19] > 0.80:
    detected.append("ShortSleeve")

  # =========================================================
  # 6. LOWER BODY (Indices 27, 28, 29, 30) - Specificity Ladder
  # =========================================================
  # Distinct cuts (Skirt/Dress/Shorts) take precedence over generic pants
  if agg_probs[30] > 0.30:
    detected.append("Skirt_or_Dress")
  elif agg_probs[29] > 0.30:
    detected.append("Shorts")
  elif agg_probs[28] > 0.30:
    detected.append("Jeans")
  elif agg_probs[27] > 0.50:
    detected.append("Trousers")

  # =========================================================
  # 7. FOOTWEAR (Indices 31, 32, 33, 34) - Mutually Exclusive Argmax
  # =========================================================
  shoe_indices = [31, 32, 33, 34]  # Boots, Sneakers, LeatherShoes, Sandals
  best_shoe_idx = shoe_indices[np.argmax([agg_probs[i] for i in shoe_indices])]
  if agg_probs[best_shoe_idx] > 0.45:
    detected.append(schema_keys[best_shoe_idx])

  # =========================================================
  # 8. BAGS & CARRYING (Indices 11, 15, 16, 17, 18) - High Threshold (70%+)
  # =========================================================
  # Back vs Shoulder Bag
  if agg_probs[11] > 0.70 and agg_probs[11] > agg_probs[15]:
    detected.append("Backpack")
  elif agg_probs[15] > 0.70:
    detected.append("ShoulderBag")

  # Hand-held items (HandBag, PlasticBag, CarryingOther)
  hand_indices = [16, 17, 18]
  best_hand_idx = hand_indices[np.argmax([agg_probs[i] for i in hand_indices])]
  if agg_probs[best_hand_idx] > 0.70:
    detected.append(schema_keys[best_hand_idx])

  # =========================================================
  # 9. ACCESSORIES (Indices 12, 13, 14) - Independent Binary Checks
  # =========================================================
  if agg_probs[12] > 0.50:
    detected.append("Hat")
  if agg_probs[13] > 0.15:
    detected.append("Glasses")
  if agg_probs[14] > 0.65:
    detected.append("Muffler_Scarf")

  return detected

# --- NEW: STATE MANAGEMENT CLASS ---
class PedestrianState:
    def __init__(self, top_k=10, fps=30):
        self.top_k = top_k
        self.buffer = []
        self.locked_attrs = []
        self.is_collecting = True
        self.active_view = None
        self.view_history = deque(maxlen=3 * fps) # 3 seconds rolling window

    def update(self, q_score, probs, schema_keys):
        # Determine instantaneous view for this frame
        views = ["Front", "Back", "Side"]
        current_view = views[np.argmax(probs[0:3])]
        self.view_history.append(current_view)

        # 1. Check for angle shifts if we are already locked
        if not self.is_collecting:
            view_counts = defaultdict(int)
            for v in self.view_history:
                view_counts[v] += 1
            
            # If a NEW angle appears at least 5 times in the last 3 seconds
            for v, count in view_counts.items():
                if v != self.active_view and count >= 5:
                    self.is_collecting = True
                    self.buffer = [] # Start fresh buffer for the new angle
                    self.view_history.clear()
                    break
        
        # 2. Manage the optimal inference window
        if self.is_collecting:
            self.buffer.append((q_score, probs))
            self.buffer.sort(key=lambda x: x[0], reverse=True) # Sort by quality score
            if len(self.buffer) > self.top_k:
                self.buffer = self.buffer[:self.top_k]
            
            # If we achieved the optimal window lock
            if len(self.buffer) == self.top_k:
                agg_probs = np.mean([p for _, p in self.buffer], axis=0)
                self.locked_attrs = parse_attributes(agg_probs, schema_keys)
                self.active_view = views[np.argmax(agg_probs[0:3])]
                self.is_collecting = False
                self.view_history.clear() # Reset to prevent immediate re-trigger

        # 3. Determine what to display
        if self.locked_attrs:
            # If we have locked attributes, display them (even if currently re-collecting a new angle)
            return self.locked_attrs, len(self.buffer)
        else:
            # First time seeing the person, show live averages until locked
            agg_probs = np.mean([p for _, p in self.buffer], axis=0)
            return parse_attributes(agg_probs, schema_keys), len(self.buffer)

def draw_pedestrian_data(frame, track_id, bbox, display_attrs, buf_count, top_k):
    """Renders the bounding box and text overlay onto the video frame."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(frame, f"ID: {track_id} (Opt: {buf_count}/{top_k})", 
                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    
    y_text = y1 + 15
    for attr in display_attrs:
        cv2.putText(frame, attr, (x2 + 6, y_text), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        y_text += 20

def process_video_stream(video_source, output_video, yolo_model, attr_model, transform, device, 
                         start_frame=400, max_frames=150, top_k_optimal=10):
    """Orchestrates the tracking, crop extraction, batch inference, and video writing."""
    
    # Initialize the state manager for all tracks
    track_states = defaultdict(lambda: PedestrianState(top_k=top_k_optimal, fps=30))
    missing_tracks = defaultdict(int) # Tracks how many frames an ID is absent
    video_writer = None 
    
    results = yolo_model.track(source=video_source, tracker="bytetrack.yaml", classes=[0], conf=0.4, stream=True)

    print(f"Skipping first {start_frame} frames. Fast forwarding...")
    start_time = time.time()

    for frame_idx, r in enumerate(results):
        if frame_idx < start_frame:
            continue
        if max_frames is not None and frame_idx >= start_frame + max_frames:
            break
            
        frame = r.orig_img.copy()
        
        # Initialize video writer on the first processed frame
        if video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (w, h))
            print("Fast forward complete. Starting batch processing...")

        current_ids_in_frame = set()
        batch_tensors = []
        batch_metadata = [] 
        
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy() 
            ids = r.boxes.id.cpu().numpy()     
            
            # --- PREPARE CROP BATCH ---
            for box, track_id in zip(boxes, ids):
                track_id = int(track_id)
                current_ids_in_frame.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                bbox_height, bbox_width = y2 - y1, x2 - x1
                quality_score = bbox_height * bbox_width  
                
                pedestrian_crop = frame[y1:y2, x1:x2]
                
                if pedestrian_crop.size > 0 and bbox_height > 25:
                    crop_rgb = cv2.cvtColor(pedestrian_crop, cv2.COLOR_BGR2RGB)
                    batch_tensors.append(transform(crop_rgb))
                    batch_metadata.append((track_id, (x1, y1, x2, y2), quality_score))
            
            # --- BATCH INFERENCE & STATE UPDATE ---
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(device)
                
                with torch.no_grad():
                    logits = attr_model(batch_tensor)
                    batch_probs = torch.sigmoid(logits).cpu().numpy()
                
                for meta, probs in zip(batch_metadata, batch_probs):
                    t_id, bbox, q_score = meta
                    
                    # Update State Manager & get attributes to display
                    display_attrs, buf_count = track_states[t_id].update(q_score, probs, SCHEMA_KEYS)
                    draw_pedestrian_data(frame, t_id, bbox, display_attrs, buf_count, top_k_optimal)

        # --- CLEANUP (With 30-Frame Grace Period) ---
        stale_ids = set(track_states.keys()) - current_ids_in_frame
        for stale_id in stale_ids:
            missing_tracks[stale_id] += 1
            if missing_tracks[stale_id] > 30: # Delete only if missing for > 1 second
                del track_states[stale_id]
                del missing_tracks[stale_id]
                
        # Reset the missing counter for anyone currently in frame
        for active_id in current_ids_in_frame:
            if active_id in missing_tracks:
                del missing_tracks[active_id]

        if video_writer is not None:
            video_writer.write(frame)

    if video_writer is not None:
        video_writer.release()

    elapsed = time.time() - start_time
    print(f"E2E Stream Complete! Processed in {elapsed:.2f} seconds.")
    return elapsed