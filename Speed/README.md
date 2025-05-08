# Vehicle Speed Detection Dashboard

A Flask-based web dashboard for vehicle detection, tracking, and speed estimation using YOLOv8.

## Features

- Upload video for processing
- Rotate video left or right as needed (90°, 180°, 270°)
- Define Region of Interest (ROI) using two methods:
  - Visual drag-and-drop of corner points
  - Manual input of exact coordinates
- Enter real-world dimensions for accurate speed calculation 
- Real-time vehicle detection and tracking
- Speed calculation for detected vehicles
- Vehicle image capture and saving
- Display of vehicle data in a table format

## Requirements

- Python 3.8+
- Flask
- OpenCV
- YOLOv8
- Flask-SocketIO

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have YOLOv8x model (`yolov8x.pt`) in the project root directory

## Usage

### On Windows:
1. Run the application using the provided batch file:
   ```
   run.bat
   ```

### On any platform:
1. Run the application:
   ```
   python run.py
   ```
2. Open your browser and navigate to `http://localhost:5000`
3. Upload a video file
4. If needed, use the rotation buttons to rotate the video to the correct orientation
5. Define the Region of Interest (ROI) using either:
   - Visual method: Drag the four red corner points to the desired positions
   - Manual method: Click the "Manual Coordinates" tab and enter precise coordinates
6. Enter the real-world dimensions (width and height in meters) of the ROI
7. Click "Start Tracking"
8. View the results page to see real-time detection and tracking

## How It Works

- Uses YOLOv8 for vehicle detection
- Supports video rotation for better alignment of traffic direction
- Implements a simple IoU-based tracker to follow vehicles across frames
- Uses perspective transformation to calculate real-world distances
- Calculates speed based on distance traveled and frame rate
- Captures images of detected vehicles
- Sends video frames and detection data to the client in real-time using WebSockets

## Troubleshooting

If you encounter issues with file paths, verify that:
1. The YOLOv8x model file is in the root directory
2. The required folders exist (app/static/uploads, app/static/results)
3. The application has permission to write to these directories 