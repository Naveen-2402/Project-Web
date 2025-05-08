import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math

model = YOLO('yolo11x.pt')
tracker = DeepSort(max_age=1, n_init=3, nms_max_overlap=0.3)

roi_coords = np.array([
    [63, 668],
    [992, 667],
    [764, 86],
    [303, 84]
], dtype=np.int32)

real_world_length = 15
real_world_width = 5

cap = cv2.VideoCapture("35-1-r.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rotated_width, rotated_height = orig_height, orig_width
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("perspective.mp4", fourcc, fps, (rotated_width, rotated_height))

feature_params = dict(maxCorners=50, qualityLevel=0.01, minDistance=5, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

frame_count = 0
vehicle_data = {}
tracks_dict = {}

vehicle_classes = [3, 5, 7]

# Class names dictionary (YOLO classes)
class_names = {
    2: 'car',
    3: 'motocycle',
    5: 'bus',
    7: 'truck'
}

CYAN = (255, 255, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)

persistence_frames = int(fps * 4)
observation_frames = 10

dst_points = np.array([
    [0, 0],
    [real_world_width, 0],
    [real_world_width, real_world_length],
    [0, real_world_length]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(roi_coords.astype(np.float32), dst_points)

def image_to_real_world(point):
    px, py = point
    transformed_point = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), M)
    return transformed_point[0][0]

def is_inside_polygon(point, polygon):
    point_tuple = (int(point[0]), int(point[1]))
    return cv2.pointPolygonTest(polygon, point_tuple, False) >= 0

def calculate_speed(positions, timestamps):
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

prev_gray = None
roi_polygon = np.array(roi_coords).reshape((-1, 1, 2))
debug_mode = True
final_speed = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        prev_gray = gray.copy()
    cv2.polylines(frame, [roi_coords], True, GREEN, 2)
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        if len(box.xyxy) > 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if debug_mode:
                print(f"Box: {x1}, {y1}, {x2}, {y2}, class: {cls}, conf: {conf}")
            if cls in vehicle_classes:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, 'vehicle'))
                # Display class name on the frame
                class_name = class_names.get(cls, f'class_{cls}')
                cv2.putText(frame, class_name, (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    tracks = tracker.update_tracks(detections, frame=frame)
    if debug_mode:
        print(f"Frame {frame_count}: {len(tracks)} tracks")
    for track_id, points in tracks_dict.items():
        if len(points) > 1:
            recent_points = [(x, y) for x, y, f in points if frame_count - f <= persistence_frames]
            for i in range(1, len(recent_points)):
                cv2.line(frame, recent_points[i-1], recent_points[i], CYAN, 2)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        if debug_mode:
            print(f"Track ID: {track_id}, LTRB: {ltrb}")
        if ltrb is None:
            continue
        try:
            l, t, r, b = ltrb
            x1, y1, x2, y2 = int(l), int(t), int(r), int(b)
            bbox_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if track_id not in vehicle_data:
                vehicle_data[track_id] = {
                    "positions": [],
                    "real_positions": [],
                    "timestamps": [],
                    "prev_points": None,
                    "current_points": None,
                    "observation_count": 0,
                    "speeds": [],
                    "in_roi": False,
                    "was_in_roi": False
                }
                tracks_dict[track_id] = []
            is_in_roi = is_inside_polygon(bbox_centroid, roi_polygon)
            vehicle_data[track_id]["in_roi"] = is_in_roi
            if is_in_roi:
                vehicle_data[track_id]["was_in_roi"] = True
            if y1 < y2 and x1 < x2 and y1 >= 0 and x1 >= 0 and y2 < frame.shape[0] and x2 < frame.shape[1]:
                roi_gray = gray[y1:y2, x1:x2]
                if vehicle_data[track_id]["prev_points"] is not None and len(vehicle_data[track_id]["prev_points"]) > 0:
                    try:
                        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray, gray,
                            vehicle_data[track_id]["prev_points"],
                            None, **lk_params
                        )
                        good_points_mask = status == 1
                        if np.any(good_points_mask):
                            good_new = next_points[good_points_mask]
                            good_old = vehicle_data[track_id]["prev_points"][good_points_mask]
                            if len(good_new) > 0:
                                vehicle_data[track_id]["current_points"] = good_new
                                centroid = np.mean(good_new, axis=0)
                                cx, cy = centroid.astype(int)
                                vehicle_data[track_id]["positions"].append((cx, cy))
                                vehicle_data[track_id]["timestamps"].append(frame_count)
                                tracks_dict[track_id].append((cx, cy, frame_count))
                                for i, (new, old) in enumerate(zip(good_new, good_old)):
                                    a, b = new.ravel().astype(int)
                                    cv2.circle(frame, (a, b), 2, GREEN, -1)
                                cv2.circle(frame, (cx, cy), 5, RED, -1)
                                if is_in_roi:
                                    real_coords = image_to_real_world((cx, cy))
                                    vehicle_data[track_id]["real_positions"].append(real_coords)
                                    if len(vehicle_data[track_id]["real_positions"]) >= 2:
                                        speed = calculate_speed(
                                            vehicle_data[track_id]["real_positions"],
                                            vehicle_data[track_id]["timestamps"][-len(vehicle_data[track_id]["real_positions"]):]
                                        )
                                        vehicle_data[track_id]["speeds"].append(speed)
                    except Exception as e:
                        if debug_mode:
                            print(f"Error in optical flow: {e}")
                if vehicle_data[track_id]["prev_points"] is None or \
                   (vehicle_data[track_id]["current_points"] is not None and len(vehicle_data[track_id]["current_points"]) < 10):
                    if roi_gray.size > 0:
                        try:
                            corners = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
                            if corners is not None and len(corners) > 0:
                                frame_corners = np.array([[[x1 + float(pt[0][0]), y1 + float(pt[0][1])]] for pt in corners], dtype=np.float32)
                                vehicle_data[track_id]["prev_points"] = frame_corners
                                vehicle_data[track_id]["observation_count"] += 1
                                for pt in frame_corners:
                                    a, b = pt[0].astype(int)
                                    cv2.circle(frame, (a, b), 2, MAGENTA, -1)
                                if len(frame_corners) > 0:
                                    init_centroid = np.mean(frame_corners[:, 0, :], axis=0)
                                    icx, icy = init_centroid.astype(int)
                                    vehicle_data[track_id]["positions"].append((icx, icy))
                                    vehicle_data[track_id]["timestamps"].append(frame_count)
                                    tracks_dict[track_id].append((icx, icy, frame_count))
                                    cv2.circle(frame, (icx, icy), 5, BLUE, -1)
                                    if is_in_roi:
                                        real_coords = image_to_real_world((icx, icy))
                                        vehicle_data[track_id]["real_positions"].append(real_coords)
                        except Exception as e:
                            if debug_mode:
                                print(f"Error finding features: {e}")
                if vehicle_data[track_id]["current_points"] is not None and len(vehicle_data[track_id]["current_points"]) > 0:
                    vehicle_data[track_id]["prev_points"] = vehicle_data[track_id]["current_points"].reshape(-1, 1, 2)
                if not is_in_roi and vehicle_data[track_id]["was_in_roi"] and vehicle_data[track_id]["speeds"]:
                    final_speed = round(sum(vehicle_data[track_id]["speeds"][-5:]) / min(5, len(vehicle_data[track_id]["speeds"][-5:])), 2)
                    cv2.putText(frame, f"Final: {final_speed} km/h", (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)
        except Exception as e:
            if debug_mode:
                print(f"Error processing track {track_id}: {e}")
            continue
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    out.write(frame)
    cv2.imshow("Vehicle Speed Detection with Perspective Transform", frame)
    prev_gray = gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

for track_id, data in vehicle_data.items():
    speeds = data["speeds"]
    timestamps = data["timestamps"]
    if speeds and data["was_in_roi"]:
        max_speed = round(max(speeds), 2)
        frames_tracked = len(timestamps)

print("\nSummary:")
valid_speeds = []
for track_id, data in vehicle_data.items():
    if data["speeds"] and data["was_in_roi"]:
        valid_speeds.extend(data["speeds"])

if valid_speeds:
    max_speed = round(max(valid_speeds), 2)
    print(f"Total vehicles tracked: {len([v for v in vehicle_data.values() if v['was_in_roi']])}")
    print(f"Vehicles with speed data: {len([v for v in vehicle_data.values() if v['speeds'] and v['was_in_roi']])}")
    print(f"Speed: {max_speed} km/h")
    print(f"Final Speed: {final_speed} km/h")
else:
    print("No speed data collected.")