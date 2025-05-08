from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import time
import threading
import cv2
from werkzeug.utils import secure_filename
from utils.detector import process_video

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

# Create necessary folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Global variables for tracking processing status
processing_status = {}
# Global variable to store the latest processed frame for streaming
current_processed_frames = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_thread(input_path, output_path, task_id):
    """Process video in a separate thread and update status"""
    processing_status[task_id]['status'] = 'processing'
    try:
        # Initialize the streaming frame for this task
        current_processed_frames[task_id] = None
        
        helmet_status, bike_stats = process_video(input_path, output_path, 
                                                 frame_callback=lambda frame: update_current_frame(task_id, frame))
        processing_status[task_id]['status'] = 'completed'
        processing_status[task_id]['helmet_status'] = helmet_status
        processing_status[task_id]['bike_stats'] = bike_stats
    except Exception as e:
        processing_status[task_id]['status'] = 'failed'
        processing_status[task_id]['error'] = str(e)
    finally:
        # Clean up the frame when done
        if task_id in current_processed_frames:
            del current_processed_frames[task_id]

def update_current_frame(task_id, frame):
    """Update the current frame for streaming"""
    current_processed_frames[task_id] = frame

def generate_frames(task_id):
    """Generator function for video streaming"""
    while task_id in processing_status and processing_status[task_id]['status'] == 'processing':
        if task_id in current_processed_frames and current_processed_frames[task_id] is not None:
            frame = current_processed_frames[task_id]
            # Convert frame to JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.04)  # ~25 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f"processed_{int(time.time())}_{filename}"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        
        # Generate a task ID
        task_id = str(int(time.time()))
        processing_status[task_id] = {
            'status': 'queued',
            'input_file': filename,
            'output_file': output_filename,
        }
        
        # Start processing in a separate thread
        thread = threading.Thread(
            target=process_video_thread,
            args=(input_path, output_path, task_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Video uploaded and processing started'
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(processing_status[task_id])

@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/video-stream/<task_id>')
def video_stream(task_id):
    """Stream the video processing in real-time"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    return Response(generate_frames(task_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result/<task_id>')
def result(task_id):
    if task_id not in processing_status:
        return render_template('index.html', error='Task not found')
    
    return render_template('result.html', task_id=task_id)

if __name__ == '__main__':
    app.run(debug=True)