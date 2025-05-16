from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import os

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

# Load models
general_model = YOLO('yolo11n.pt')
helmet_model = YOLO('best-pp.pt')

# Set confidence threshold for general model
general_model.conf = 0.7

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

# Video processing function
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Track helmet status for each person
    # Keys will be person IDs
    helmet_status = defaultdict(lambda: "Not Wearing Helmet")
    
    # For global tracking of bike statistics
    global_bike_stats = defaultdict(lambda: {'total_persons': 0, 'helmets_worn': 0})
    
    # For tracking person-bike associations
    person_bike_map = {}  # Maps person_id to bike_id
    
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}")
        
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
        # Reset current frame's person-bike associations
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
        
        # Display the annotated frame
        cv2.imshow('Helmet Detection', display_frame)
        
        # Write the frame to output video
        out.write(display_frame)
        
        # Break if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return helmet_status, global_bike_stats

# Usage
if __name__ == "__main__":
    video_path = "1.mp4"
    output_path = "annotated_output.mp4"
    final_status, bike_stats = process_video(video_path, output_path)
    
    print("\nFinal helmet status for each person:")
    for person_id, status in final_status.items():
        print(f"Person {person_id}: {status}")
    
    print("\nFinal statistics for each bike:")
    for bike_id, stats in bike_stats.items():
        print(f"Bike {bike_id}: {stats['total_persons']} persons, {stats['helmets_worn']} wearing helmets")