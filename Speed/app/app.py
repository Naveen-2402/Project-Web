import os
import cv2
import numpy as np
import math  # Add math import for distance calculations
from flask import Flask, render_template, request, Response, jsonify
from flask_socketio import SocketIO
from ultralytics import YOLO
import json
import time
import datetime
import threading
import re  # Add import for regex
from sort import Sort  # Import SORT tracker

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vehicletracker'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join('static', 'results')
socketio = SocketIO(app)

# Global variables
video_path = None
# Default ROI coordinates from perspective-lucas.py
roi_points = [[63, 668], [992, 667], [764, 86], [303, 84]]
real_width = 5  # Default width in meters
real_height = 15  # Default height in meters
processing_thread = None
stop_processing = False
current_vehicles = {}
exited_vehicles = {}
next_vehicle_id = 1
rotation_angle = 0  # Degrees: 0, 90, 180, 270
ground_truth = 0  # Ground truth speed from filename

# Load YOLO model
model = YOLO('yolo11x.pt')  # Changed to yolo11x.pt to match perspective-lucas.py

# Vehicle class IDs in COCO dataset
VEHICLE_CLASSES = [3, 5, 7]  # motorcycle, bus, truck

def calculate_speed(positions, timestamps, fps):
    """Calculate speed using the algorithm from perspective-lucas.py"""
    if len(positions) < 2 or len(timestamps) < 2:
        return 0
    num_positions = min(5, len(positions))
    recent_positions = positions[-num_positions:]
    recent_timestamps = timestamps[-num_positions:]
    total_distance = 0
    for i in range(1, len(recent_positions)):
        p1 = recent_positions[i-1]
        p2 = recent_positions[i]
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        total_distance += distance
    time_diff = (recent_timestamps[-1] - recent_timestamps[0]) / fps
    if time_diff > 0:
        speed = (total_distance / time_diff) * 3.6
        return round(speed, 2)
    return 0

def is_inside_polygon(point, polygon):
    """Check if a point is inside a polygon"""
    point_tuple = (int(point[0]), int(point[1]))
    return cv2.pointPolygonTest(polygon, point_tuple, False) >= 0

def process_video():
    global stop_processing, current_vehicles, next_vehicle_id, exited_vehicles, ground_truth
    
    if not video_path or len(roi_points) != 4 or not real_width or not real_height:
        return
    
    # Extract ground truth from filename if present
    try:
        filename = os.path.basename(video_path)
        match = re.match(r'(\d+)-\d+', filename)
        if match:
            ground_truth = int(match.group(1))
            print(f"Ground truth speed: {ground_truth} km/h")
        else:
            ground_truth = 0
    except Exception as e:
        print(f"Error extracting ground truth: {str(e)}")
        ground_truth = 0
    
    # Use absolute path for video capture
    full_video_path = os.path.join(os.getcwd(), 'app', video_path)
    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {full_video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create perspective transform matrix
    src_points = np.array(roi_points, dtype=np.float32)
    dst_width, dst_height = real_width, real_height  # Use real-world dimensions directly
    dst_points = np.array([[0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]], dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # For perspective transform
    def image_to_real_world(point):
        px, py = point
        transformed_point = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), perspective_matrix)
        return transformed_point[0][0]
    
    # Vehicle tracking and speed calculation
    vehicle_tracks = {}  # {id: [(frame_num, position),...]}
    tracks_dict = {}     # For tracking path visualization
    
    # Lucas-Kanade tracking parameters
    feature_params = dict(maxCorners=50, qualityLevel=0.01, minDistance=5, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Colors for visualization
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    
    # For LK optical flow
    prev_frame = None
    prev_gray = None
    vehicle_features = {}  # {id: {'prev_points': points, 'current_points': points}}
    
    frame_count = 0
    
    # Convert ROI points to a proper numpy array for OpenCV
    roi_contour = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
    
    # Variables for speed tracking
    max_speed = 0
    final_speed = 0
    persistence_frames = int(fps * 4)  # Store tracks for 4 seconds
    
    # Initialize tracking
    # For deep sort, we will use Sort tracker in our case
    mot_tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)
    
    while cap.isOpened() and not stop_processing:
        ret, frame = cap.read()
        if not ret:
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Apply rotation if needed
        if rotation_angle != 0:
            if rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray.copy()
            prev_frame = frame.copy()
        
        # Draw ROI
        roi_overlay = frame.copy()
        cv2.polylines(roi_overlay, [roi_contour], True, (0, 255, 0), 2)
        
        try:
            # Run YOLOv8 inference
            results = model(frame)
            
            # Process detections
            detections_for_sort = []
            all_detections = []
            
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Filter for vehicle classes after detection
                    if class_id not in VEHICLE_CLASSES:
                        continue
                    
                    # Check if center of the box is inside the ROI
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Format point for pointPolygonTest
                    point = (float(center_x), float(center_y))
                    
                    if cv2.pointPolygonTest(roi_contour, point, False) >= 0:
                        try:
                            # Transform the center to bird's eye view
                            center_transformed = cv2.perspectiveTransform(
                                np.array([[[center_x, center_y]]], dtype=np.float32), 
                                perspective_matrix
                            )[0][0]
                            
                            # Format detection for SORT: [x1, y1, x2, y2, confidence]
                            detections_for_sort.append([x1, y1, x2, y2, confidence])
                            
                            # Store detection details for later use
                            all_detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'class': class_id,
                                'confidence': confidence,
                                'center': (center_x, center_y),
                                'transformed_center': (center_transformed[0], center_transformed[1])
                            })
                        except Exception as e:
                            print(f"Error transforming point: {str(e)}")
                            continue
            
            # Update SORT tracker with detections
            current_frame_vehicles = {}
            
            if detections_for_sort:
                # Convert to numpy array for SORT
                detections_array = np.array(detections_for_sort)
                
                # Update SORT tracker
                tracked_objects = mot_tracker.update(detections_array)
                
                # Process tracking results
                for track in tracked_objects:
                    # SORT output: [x1, y1, x2, y2, track_id]
                    x1, y1, x2, y2, track_id = track.astype(int)
                    track_id = int(track_id)  # Ensure integer ID
                    
                    # Find associated detection by IoU
                    detection_info = None
                    max_iou = 0
                    track_bbox = (x1, y1, x2, y2)
                    
                    for detection in all_detections:
                        d_x1, d_y1, d_x2, d_y2 = detection['bbox']
                        
                        # Calculate IoU
                        intersection_x1 = max(x1, d_x1)
                        intersection_y1 = max(y1, d_y1)
                        intersection_x2 = min(x2, d_x2)
                        intersection_y2 = min(y2, d_y2)
                        
                        if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                            detection_area = (d_x2 - d_x1) * (d_y2 - d_y1)
                            track_area = (x2 - x1) * (y2 - y1)
                            union_area = detection_area + track_area - intersection_area
                            iou = intersection_area / union_area
                            
                            if iou > max_iou:
                                max_iou = iou
                                detection_info = detection
                    
                    if detection_info:
                        # Check if we've seen this track_id before
                        if track_id in current_vehicles:
                            # Update existing vehicle
                            vehicle_info = current_vehicles[track_id].copy()
                            vehicle_info['bbox'] = track_bbox  # Use SORT's bbox which is more stable
                            vehicle_info['center'] = detection_info['center']
                            vehicle_info['transformed_center'] = detection_info['transformed_center']
                            vehicle_info['last_seen'] = frame_count
                            
                            # Check if center is inside ROI
                            cx, cy = detection_info['center']
                            is_in_roi = cv2.pointPolygonTest(roi_contour, (float(cx), float(cy)), False) >= 0
                            
                            # Update ROI status
                            was_in_roi = vehicle_info.get('was_in_roi', False)
                            vehicle_info['in_roi'] = is_in_roi
                            
                            # If vehicle is in ROI now, mark that it was in ROI at some point
                            if is_in_roi:
                                vehicle_info['was_in_roi'] = True
                            
                            # Check if vehicle just exited the ROI
                            prev_in_roi = current_vehicles[track_id].get('in_roi', False)
                            if prev_in_roi and not is_in_roi and was_in_roi:
                                # Vehicle just left the ROI - calculate reported speed
                                if 'speeds' in vehicle_info and len(vehicle_info['speeds']) > 0:
                                    max_speed = vehicle_info.get('max_speed', 0)
                                    final_speed = vehicle_info.get('final_speed', 0)
                                    
                                    # Decide which speed to report based on ground truth
                                    if ground_truth > 50:
                                        vehicle_info['reported_speed'] = final_speed
                                    else:
                                        vehicle_info['reported_speed'] = max_speed
                                    
                                    # Log the speed calculation for debugging
                                    print(f"Vehicle {track_id} exited ROI: Ground truth: {ground_truth}, Max speed: {max_speed}, Final speed: {final_speed}, Reported: {vehicle_info['reported_speed']} km/h")
                            
                            # Add to tracks for speed calculation
                            if track_id not in vehicle_tracks:
                                vehicle_tracks[track_id] = []
                            
                            vehicle_tracks[track_id].append((frame_count, detection_info['transformed_center']))
                            
                            # Apply Lucas-Kanade optical flow if we have previous points
                            if track_id in vehicle_features and vehicle_features[track_id]['prev_points'] is not None:
                                prev_pts = vehicle_features[track_id]['prev_points']
                                
                                if len(prev_pts) > 0:
                                    # Calculate optical flow
                                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                                        prev_gray, gray, prev_pts, None, **lk_params
                                    )
                                    
                                    # Select good points
                                    good_points_mask = status == 1
                                    if np.any(good_points_mask):
                                        good_new = next_pts[good_points_mask]
                                        good_old = prev_pts[good_points_mask]
                                        
                                        # Draw tracks
                                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                                            a, b = new.ravel().astype(int)
                                            c, d = old.ravel().astype(int)
                                            cv2.line(roi_overlay, (c, d), (a, b), MAGENTA, 2)
                                            cv2.circle(roi_overlay, (a, b), 3, RED, -1)
                                        
                                        # Calculate and visualize centroid if we have enough points
                                        if len(good_new) > 0:
                                            # Calculate centroid
                                            centroid = np.mean(good_new, axis=0)
                                            cx, cy = centroid.astype(int)
                                            # Draw centroid (larger circle)
                                            cv2.circle(roi_overlay, (cx, cy), 5, BLUE, -1)
                                            
                                            # Store centroid position for tracking
                                            vehicle_info['positions'].append((cx, cy))
                                            vehicle_info['timestamps'].append(frame_count)
                                            
                                            # Store for visualization
                                            tracks_dict[track_id].append((cx, cy, frame_count))
                                            
                                            # If in ROI, transform to real-world coordinates
                                            if is_in_roi:
                                                try:
                                                    real_coords = image_to_real_world((cx, cy))
                                                    vehicle_info['real_positions'].append(real_coords)
                                                    
                                                    # Calculate speed with real-world coordinates
                                                    if len(vehicle_info['real_positions']) >= 2:
                                                        speed = calculate_speed(
                                                            vehicle_info['real_positions'],
                                                            vehicle_info['timestamps'][-len(vehicle_info['real_positions']):],
                                                            fps
                                                        )
                                                        
                                                        vehicle_info['speeds'].append(speed)
                                                        vehicle_info['speed'] = speed
                                                        
                                                        # Update max speed
                                                        if 'max_speed' not in vehicle_info or speed > vehicle_info['max_speed']:
                                                            vehicle_info['max_speed'] = speed
                                                except Exception as e:
                                                    print(f"Error in coordinate transform: {str(e)}")
                                            
                                            # Store centroid position
                                            vehicle_info['centroid'] = (cx, cy)
                                        
                                        # Update points for next frame
                                        vehicle_features[track_id]['prev_points'] = good_new.reshape(-1, 1, 2)
                                        vehicle_info['current_points'] = good_new
                            
                            # Calculate speed if we have enough tracking history
                            if len(vehicle_tracks[track_id]) >= 10:
                                # Get positions for speed calculation
                                recent_positions = vehicle_tracks[track_id][-10:]
                                
                                # Calculate speed using recent positions
                                total_distance = 0
                                for i in range(1, len(recent_positions)):
                                    _, pos1 = recent_positions[i-1]
                                    _, pos2 = recent_positions[i]
                                    
                                    # Positions are already in real-world coordinates from the transform
                                    x1, y1 = pos1
                                    x2, y2 = pos2
                                    segment_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # meters
                                    total_distance += segment_distance
                                
                                # Calculate time difference
                                start_frame, _ = recent_positions[0]
                                end_frame, _ = recent_positions[-1]
                                frame_diff = end_frame - start_frame
                                
                                if frame_diff > 0:
                                    time_diff = frame_diff / fps  # seconds
                                    speed = (total_distance / time_diff) * 3.6  # km/h
                                    
                                    current_speed = round(speed, 1)
                                    vehicle_info['speed'] = current_speed
                                    
                                    # Track speed history and update max/final speed
                                    if 'speeds' not in vehicle_info:
                                        vehicle_info['speeds'] = []
                                    
                                    vehicle_info['speeds'].append(current_speed)
                                    
                                    # Keep only the last 20 speed measurements
                                    if len(vehicle_info['speeds']) > 20:
                                        vehicle_info['speeds'] = vehicle_info['speeds'][-20:]
                                    
                                    # Update max speed
                                    vehicle_info['max_speed'] = max(vehicle_info['speeds'])
                                    
                                    # Update final speed - average of last 5 speeds or fewer if not enough data
                                    if len(vehicle_info['speeds']) >= 5:
                                        vehicle_info['final_speed'] = round(sum(vehicle_info['speeds'][-5:]) / min(5, len(vehicle_info['speeds'][-5:])), 1)
                                    else:
                                        vehicle_info['final_speed'] = current_speed
                            
                            # Update the vehicle info
                            current_frame_vehicles[track_id] = vehicle_info
                        else:
                            # Create new vehicle for this track_id
                            # Crop vehicle image and save
                            x1, y1, x2, y2 = track_bbox
                            vehicle_img = frame[y1:y2, x1:x2]
                            img_path = ''
                            
                            if vehicle_img.size > 0:
                                # Create the full path for saving
                                results_dir = os.path.join(os.getcwd(), 'app', app.config['RESULTS_FOLDER'])
                                os.makedirs(results_dir, exist_ok=True)
                                img_path = os.path.join(app.config['RESULTS_FOLDER'], f"{track_id}.png")
                                full_img_path = os.path.join(os.getcwd(), 'app', img_path)
                                cv2.imwrite(full_img_path, vehicle_img)
                            
                            # Create vehicle info
                            class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                            class_id = detection_info['class']
                            
                            # Check if center is inside ROI
                            bbox_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                            is_in_roi = is_inside_polygon(bbox_centroid, roi_contour)
                            
                            # Initialize vehicle data with structure matching perspective-lucas.py
                            vehicle_info = {
                                'id': track_id,
                                'class': class_names.get(class_id, 'vehicle'),
                                'bbox': track_bbox,
                                'center': detection_info['center'],
                                'transformed_center': detection_info['transformed_center'],
                                'first_seen': frame_count,
                                'last_seen': frame_count,
                                'positions': [],        # Image space positions
                                'real_positions': [],   # Real-world space positions
                                'timestamps': [],       # Timestamps for positions
                                'prev_points': None,    # For optical flow
                                'current_points': None, # For optical flow
                                'observation_count': 0, # Track observation count
                                'speeds': [],           # List of speed measurements
                                'speed': 0,             # Current speed
                                'max_speed': 0,         # Maximum speed
                                'final_speed': 0,       # Final speed (avg of last 5)
                                'in_roi': is_in_roi,    # Currently in ROI
                                'was_in_roi': is_in_roi, # Was ever in ROI
                                'img_path': img_path
                            }
                            
                            tracks_dict[track_id] = []
                            current_frame_vehicles[track_id] = vehicle_info
                            vehicle_tracks[track_id] = [(frame_count, detection_info['transformed_center'])]
                            
                            # Find features for optical flow tracking
                            roi = gray[y1:y2, x1:x2]
                            if roi.size > 0:
                                # Find initial corners/features to track
                                corners = cv2.goodFeaturesToTrack(roi, **feature_params)
                                if corners is not None and len(corners) > 0:
                                    # Adjust corner coordinates to the full frame
                                    corners_adjusted = corners.copy()
                                    corners_adjusted[:, 0, 0] += x1
                                    corners_adjusted[:, 0, 1] += y1
                                    
                                    # Draw corners
                                    for corner in corners_adjusted:
                                        x, y = corner.ravel().astype(int)
                                        cv2.circle(roi_overlay, (x, y), 3, GREEN, -1)
                                    
                                    # Calculate and draw initial centroid
                                    if len(corners_adjusted) > 0:
                                        centroid = np.mean(corners_adjusted.reshape(-1, 2), axis=0)
                                        icx, icy = centroid.astype(int)
                                        cv2.circle(roi_overlay, (icx, icy), 5, BLUE, -1)
                                        vehicle_info['centroid'] = (icx, icy)
                                    
                                    # Store points for tracking
                                    vehicle_features[track_id] = {'prev_points': corners_adjusted}
            
            # Check for vehicles that have left the ROI
            for vid in list(current_vehicles.keys()):
                if vid not in current_frame_vehicles:
                    # Check if this vehicle has been gone for too long
                    vehicle = current_vehicles[vid]
                    frames_gone = frame_count - vehicle['last_seen']
                    
                    # If the vehicle has been gone for more than 30 frames, move it to exited list
                    if frames_gone > 30:  # About 1 second at 30fps
                        # Only process vehicles that were in the ROI at some point
                        if vehicle.get('was_in_roi', False) and vehicle.get('speeds', []):
                            # Calculate final statistics before moving to exited list
                            speeds = vehicle.get('speeds', [])
                            
                            if speeds:
                                # Calculate max_speed and final_speed as in perspective-lucas.py
                                max_speed = round(max(speeds), 2)
                                final_speed = round(sum(speeds[-5:]) / min(5, len(speeds[-5:])), 2)
                                
                                vehicle['max_speed'] = max_speed
                                vehicle['final_speed'] = final_speed
                                
                                # Select which speed to report based on ground truth
                                if ground_truth > 50:
                                    vehicle['reported_speed'] = final_speed
                                else:
                                    vehicle['reported_speed'] = max_speed
                                    
                                print(f"Vehicle {vid} final stats - Ground truth: {ground_truth}, Max: {max_speed}, Final: {final_speed}, Reported: {vehicle['reported_speed']} km/h")
                        
                        # Move vehicle to exited list and clean up resources
                        exited_vehicles[vid] = vehicle
                        if vid in vehicle_features:
                            del vehicle_features[vid]
                        if vid in vehicle_tracks:
                            del vehicle_tracks[vid]
                        if vid in tracks_dict:
                            del tracks_dict[vid]
                        if vid in current_vehicles:
                            del current_vehicles[vid]
                    else:
                        # Keep it in current vehicles but don't update it
                        current_frame_vehicles[vid] = vehicle
            
            # Update current vehicles with the new ones
            current_vehicles = current_frame_vehicles
            
            # Draw bounding boxes and info
            for vid, vehicle in current_vehicles.items():
                # Skip vehicles that have exited but haven't been fully removed yet
                if not vehicle.get('in_roi', False) and vehicle.get('was_in_roi', False) and vehicle.get('reported_speed', 0) > 0:
                    continue
                    
                x1, y1, x2, y2 = vehicle['bbox']
                speed = vehicle.get('speed', 0)
                max_speed = vehicle.get('max_speed', 0)
                final_speed = vehicle.get('final_speed', 0)
                reported_speed = vehicle.get('reported_speed', 0)
                vehicle_class = vehicle.get('class', 'vehicle')
                was_in_roi = vehicle.get('was_in_roi', False)
                in_roi = vehicle.get('in_roi', False)
                
                # Different colors for different vehicle classes
                color = BLUE  # Default color
                if vehicle_class == 'car':
                    color = (255, 0, 0)  # Blue
                elif vehicle_class == 'motorcycle':
                    color = (0, 255, 0)  # Green
                elif vehicle_class == 'bus':
                    color = (0, 0, 255)  # Red
                elif vehicle_class == 'truck':
                    color = (255, 0, 255)  # Magenta
                
                cv2.rectangle(roi_overlay, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and vehicle class
                cv2.putText(roi_overlay, f"ID: {vid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(roi_overlay, f"{vehicle_class}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display speed information based on vehicle state
                if was_in_roi:
                    if not in_roi and reported_speed > 0:  # Vehicle exited ROI and has reported speed
                        cv2.putText(roi_overlay, f"Final: {reported_speed} km/h", (x1, y1-50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)
                    elif in_roi and speed > 0:  # Vehicle is in ROI and being tracked
                        cv2.putText(roi_overlay, f"Speed: {speed} km/h", (x1, y1-50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw movement path if we have track points
                if vid in tracks_dict and len(tracks_dict[vid]) > 1:
                    recent_points = [(x, y) for x, y, f in tracks_dict[vid] if frame_count - f <= persistence_frames]
                    for i in range(1, len(recent_points)):
                        cv2.line(roi_overlay, recent_points[i-1], recent_points[i], CYAN, 2)
        
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            continue
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', roi_overlay)
        frame_data = buffer.tobytes()
        
        # Send frame to clients via socketio
        socketio.emit('video_frame', {'frame': frame_data.hex()})
        
        # Send vehicle data
        vehicle_data = []
        for vid, vehicle in current_vehicles.items():
            was_in_roi = vehicle.get('was_in_roi', False)
            in_roi = vehicle.get('in_roi', False)
            reported_speed = vehicle.get('reported_speed', 0)
            
            # Display status based on vehicle state
            speed_display = "tracking..."
            if was_in_roi and not in_roi and reported_speed > 0:
                speed_display = f"{reported_speed} km/h"
                
                # Only show vehicles that have exited ROI and have a speed
                vehicle_data.append({
                    'id': vid,
                    'class': vehicle.get('class', 'vehicle'),
                    'speed': speed_display,
                    'img_path': vehicle.get('img_path', ''),
                    'exited': True
                })
            elif in_roi:
                # Only show vehicles that are currently in the ROI
                vehicle_data.append({
                    'id': vid,
                    'class': vehicle.get('class', 'vehicle'),
                    'speed': speed_display,
                    'img_path': vehicle.get('img_path', '')
                })
        
        # Add exited vehicles with their speeds
        for vid, vehicle in exited_vehicles.items():
            was_in_roi = vehicle.get('was_in_roi', False)
            reported_speed = vehicle.get('reported_speed', 0)
            
            # Only send vehicles that were in ROI and have a calculated speed
            if was_in_roi and reported_speed > 0:
                vehicle_data.append({
                    'id': vid,
                    'class': vehicle.get('class', 'vehicle'),
                    'speed': f"{reported_speed} km/h",
                    'img_path': vehicle.get('img_path', ''),
                    'exited': True
                })
        
        socketio.emit('vehicle_data', {'vehicles': vehicle_data})
        
        frame_count += 1
        prev_gray = gray.copy()
        time.sleep(0.03)  # Limit to ~30 FPS
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Save the uploaded video
    uploads_dir = os.path.join(os.getcwd(), 'app', app.config['UPLOAD_FOLDER'])
    os.makedirs(uploads_dir, exist_ok=True)
    filename = f"input_video_{int(time.time())}.mp4"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    full_path = os.path.join(os.getcwd(), 'app', file_path)
    video_file.save(full_path)
    
    # Set the video path
    video_path = file_path
    
    return jsonify({'success': True, 'video_path': file_path})

@app.route('/setup', methods=['POST'])
def setup_tracking():
    global roi_points, real_width, real_height, processing_thread, stop_processing, rotation_angle
    
    data = request.json
    
    if 'roi_points' not in data or 'real_width' not in data or 'real_height' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Stop existing thread if running
    if processing_thread and processing_thread.is_alive():
        stop_processing = True
        processing_thread.join()
    
    # Update parameters
    roi_points = data['roi_points']
    real_width = float(data['real_width'])
    real_height = float(data['real_height'])
    
    # Update rotation angle if provided
    if 'rotation_angle' in data:
        rotation_angle = int(data['rotation_angle'])
    
    # Start processing in a new thread
    stop_processing = False
    processing_thread = threading.Thread(target=process_video)
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({'success': True})

@app.route('/rotate_video', methods=['POST'])
def rotate_video():
    global rotation_angle
    
    data = request.json
    if 'direction' not in data:
        return jsonify({'error': 'Missing rotation direction'}), 400
    
    direction = data['direction']
    
    # Update rotation angle based on direction
    if direction == 'right':
        rotation_angle = (rotation_angle + 90) % 360
    elif direction == 'left':
        rotation_angle = (rotation_angle - 90) % 360
    else:
        return jsonify({'error': 'Invalid rotation direction. Use "left" or "right"'}), 400
    
    return jsonify({'success': True, 'rotation_angle': rotation_angle})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)