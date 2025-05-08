from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import os
import time

# Import DeepSORT modules
from deep_sort_realtime.deepsort_tracker import DeepSort

# Function to check if centroid is inside bbox
def is_point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# Function to calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

# Load models (lazy loading)
_models = {}

def get_models():
    if not _models:
        # Load models on first use
        _models['general'] = YOLO('models/yolo11n.pt')
        _models['helmet'] = YOLO('models/best-pp.pt')
        
        # Set confidence threshold for general model
        _models['general'].conf = 0.7
    
    return _models

# Video processing function
def process_video(video_path, output_path):
    # Get models
    models = get_models()
    general_model = models['general']
    helmet_model = models['helmet']
    
    # Initialize DeepSORT trackers
    person_tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100
    )
    
    motorcycle_tracker = DeepSort(
        max_age=30, 
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Track helmet status for each person
    helmet_status = defaultdict(lambda: "Not Wearing Helmet")
    
    # For global tracking of bike statistics
    global_bike_stats = defaultdict(lambda: {'total_persons': 0, 'helmets_worn': 0})
    
    # For tracking person-bike associations
    person_bike_map = {}  # Maps person_id to bike_id
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:  # Print status every 10 frames
            print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
        
        # Get detections from both models
        general_results = general_model(frame)[0]
        helmet_results = helmet_model(frame)[0]
        
        # Process general model results (persons and motorcycles)
        person_detections = []
        motorcycle_detections = []
        
        for det in general_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if cls == 0:  # person class
                person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
            elif cls == 3:  # motorcycle class
                motorcycle_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'motorcycle'))
        
        # Track persons using DeepSORT
        person_tracks = person_tracker.update_tracks(person_detections, frame=frame)
        tracked_persons = []
        
        for track in person_tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            tracked_persons.append({
                'id': track_id,
                'bbox': [x1, y1, x2, y2],
                'conf': track.get_det_conf() if track.get_det_conf() is not None else 1.0
            })
        
        # Track motorcycles using DeepSORT
        motorcycle_tracks = motorcycle_tracker.update_tracks(motorcycle_detections, frame=frame)
        tracked_motorcycles = []
        
        for track in motorcycle_tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            tracked_motorcycles.append({
                'id': track_id,
                'bbox': [x1, y1, x2, y2],
                'conf': track.get_det_conf() if track.get_det_conf() is not None else 1.0
            })
        
        # Process helmet model results (class 2 is helmet)
        helmets = []
        for det in helmet_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 2:  # Helmet class
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                helmets.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'centroid': (center_x, center_y),
                    'conf': conf
                })
        
        # Associate persons with motorcycles
        person_motorcycle_pairs = []
        current_person_bike_map = {}
        
        for person in tracked_persons:
            person_id = person['id']
            for motorcycle in tracked_motorcycles:
                if calculate_iou(person['bbox'], motorcycle['bbox']) > 0.1:  # Threshold
                    bike_id = motorcycle['id']
                    person_motorcycle_pairs.append({
                        'person': person,
                        'vehicle': motorcycle
                    })
                    current_person_bike_map[person_id] = bike_id
                    
                    # Update global bike stats - count unique persons per bike
                    if person_id not in person_bike_map or person_bike_map.get(person_id) != bike_id:
                        global_bike_stats[bike_id]['total_persons'] += 1
        
        # Update person-bike mapping for next frame
        person_bike_map = current_person_bike_map
        
        # Associate helmets with person-motorcycle pairs
        for pair in person_motorcycle_pairs:
            person = pair['person']
            vehicle = pair['vehicle']
            person_id = person['id']
            bike_id = vehicle['id']
            
            # Check if this person already has "Wearing Helmet" status
            if helmet_status[person_id] == "Wearing Helmet":
                # If they've newly joined this bike and are wearing a helmet, count them
                if global_bike_stats[bike_id]['helmets_worn'] == 0:
                    global_bike_stats[bike_id]['helmets_worn'] += 1
                continue  # Maintain the status
                
            for helmet in helmets:
                person_box_contains_helmet = is_point_in_box(helmet['centroid'], person['bbox'])
                vehicle_box_contains_helmet = is_point_in_box(helmet['centroid'], vehicle['bbox'])
                
                if person_box_contains_helmet:
                    if not vehicle_box_contains_helmet:
                        if helmet_status[person_id] != "Wearing Helmet":
                            helmet_status[person_id] = "Wearing Helmet"
                            # Update global bike stats for helmets
                            global_bike_stats[bike_id]['helmets_worn'] += 1
                        break
                    else:
                        helmet_status[person_id] = "Helmet in Bike"
        
        # Calculate current frame bike stats (for display purposes)
        current_frame_bike_stats = defaultdict(lambda: {'total_persons': 0, 'helmets_worn': 0})
        
        for pair in person_motorcycle_pairs:
            bike_id = pair['vehicle']['id']
            person_id = pair['person']['id']
            
            current_frame_bike_stats[bike_id]['total_persons'] += 1
            if helmet_status[person_id] == "Wearing Helmet":
                current_frame_bike_stats[bike_id]['helmets_worn'] += 1
        
        # Create a copy of the frame for annotation
        display_frame = frame.copy()
        
        # Annotate the frame
        for pair in person_motorcycle_pairs:
            person = pair['person']
            vehicle = pair['vehicle']
            person_id = person['id']
            bike_id = vehicle['id']
            
            # Draw motorcycle bbox
            cv2.rectangle(display_frame, 
                          (vehicle['bbox'][0], vehicle['bbox'][1]), 
                          (vehicle['bbox'][2], vehicle['bbox'][3]), 
                          (0, 255, 0), 2)
            
            # Add bike ID and stats
            bike_text = f"Bike ID: {bike_id}"
            cv2.putText(display_frame, bike_text, 
                        (vehicle['bbox'][0], vehicle['bbox'][1] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            stats_text = f"Persons: {current_frame_bike_stats[bike_id]['total_persons']}, Helmets: {current_frame_bike_stats[bike_id]['helmets_worn']}"
            cv2.putText(display_frame, stats_text, 
                        (vehicle['bbox'][0], vehicle['bbox'][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw person bbox
            cv2.rectangle(display_frame, 
                          (person['bbox'][0], person['bbox'][1]), 
                          (person['bbox'][2], person['bbox'][3]), 
                          (255, 0, 0), 2)
            
            # Add status text
            status = helmet_status[person_id]
            color = (0, 255, 0) if status == "Wearing Helmet" else (0, 0, 255)
            cv2.putText(display_frame, f"Person {person_id}: {status}", 
                        (person['bbox'][0], person['bbox'][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw helmet bboxes
        for helmet in helmets:
            cv2.rectangle(display_frame, 
                          (helmet['bbox'][0], helmet['bbox'][1]), 
                          (helmet['bbox'][2], helmet['bbox'][3]), 
                          (0, 165, 255), 2)
            cv2.putText(display_frame, "Helmet", 
                        (helmet['bbox'][0], helmet['bbox'][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Write the frame to output video
        out.write(display_frame)
        
    # Release resources
    cap.release()
    out.release()
    
    # Convert helmet_status defaultdict to dict for JSON serialization
    helmet_status_dict = {str(k): v for k, v in helmet_status.items()}
    bike_stats_dict = {str(k): v for k, v in global_bike_stats.items()}
    
    return helmet_status_dict, bike_stats_dict