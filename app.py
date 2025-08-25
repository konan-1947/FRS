from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import numpy as np
import base64
import io
from mtcnn import MTCNN
import tensorflow as tf
from face_recognition_system import FaceRecognitionSystem
from config import config
import os
import threading
import time
import re
import unicodedata

app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_CONFIG', 'development')
app.config.from_object(config[config_name])
config[config_name].init_app(app)

# Global variables
detector = None
face_recognition_sys = None
camera = None
detection_active = False
current_frame = None
detection_results = []

def initialize_models():
    """Initialize MTCNN detector and face recognition system"""
    global detector, face_recognition_sys
    try:
        # Initialize MTCNN with default configuration for better compatibility
        print("Initializing MTCNN detector...")
        detector = MTCNN()
        
        # Test detector with a small dummy image to check compatibility
        test_image = np.ones((48, 48, 3), dtype=np.uint8) * 128
        test_result = detector.detect_faces(test_image)
        print(f"MTCNN test completed: {len(test_result) if test_result else 0} faces detected")
        
        print("Initializing face recognition system...")
        face_recognition_sys = FaceRecognitionSystem()
        
        print("✅ Models initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        print("💡 Try: pip install --upgrade mtcnn tensorflow")
        return False

def visualize_faces(image, detection_results):
    """Enhanced face visualization with recognition status"""
    vis_image = image.copy()
    
    if not detection_results:
        cv2.putText(vis_image, "No faces detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return vis_image

    for i, face in enumerate(detection_results):
        try:
            x, y, width, height = face['box']
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, image.shape[1] - x)
            height = min(height, image.shape[0] - y)
            
            # Check if this is an authorized person
            is_authorized = face_recognition_sys.is_authorized_person(image, face)
            
            # Color coding: Green for authorized, Red for unauthorized
            box_color = (0, 255, 0) if is_authorized else (0, 0, 255)
            status_text = "AUTHORIZED" if is_authorized else "UNAUTHORIZED"
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + width, y + height), box_color, 2)

            # Draw confidence score and status
            confidence = face['confidence']
            cv2.putText(vis_image, f'Face {i+1}: {confidence:.3f}',
                        (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        box_color, 2)
            
            cv2.putText(vis_image, status_text,
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        box_color, 2)

            # Draw keypoints if available
            if 'keypoints' in face:
                keypoints = face['keypoints']
                colors = {
                    'left_eye': (255, 0, 0),
                    'right_eye': (255, 0, 0),
                    'nose': (0, 255, 255),
                    'mouth_left': (0, 0, 255),
                    'mouth_right': (0, 0, 255)
                }
                
                for key, (kx, ky) in keypoints.items():
                    color = colors.get(key, (255, 255, 255))
                    cv2.circle(vis_image, (int(kx), int(ky)), 3, color, -1)
                    
        except Exception as e:
            print(f"Error visualizing face {i}: {e}")
            continue

    return vis_image

def capture_frames():
    """Background thread function to capture and process frames"""
    global camera, current_frame, detection_results, detection_active
    
    frame_count = 0
    last_detection_results = []
    
    while detection_active:
        if camera is not None and camera.isOpened():
            ret, frame = camera.read()
            if ret:
                current_frame = frame.copy()
                frame_count += 1
                
                # Process frame for face detection (every nth frame for performance)
                if frame_count % app.config['PROCESS_EVERY_N_FRAMES'] == 0:
                    try:
                        # Validate frame
                        if frame is None or frame.size == 0:
                            print("Invalid frame detected, skipping...")
                            detection_results = last_detection_results
                            continue
                        
                        height, width = frame.shape[:2]
                        
                        # Ensure minimum frame size
                        if height < 24 or width < 24:
                            print(f"Frame too small ({width}x{height}), skipping...")
                            detection_results = last_detection_results
                            continue
                        
                        # Resize frame for optimal processing
                        target_width = min(width, app.config['FRAME_RESIZE_MAX_WIDTH'])
                        if width > target_width:
                            scale = target_width / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            
                            # Ensure dimensions are even numbers and minimum size
                            new_width = max(48, new_width - (new_width % 2))
                            new_height = max(48, new_height - (new_height % 2))
                            
                            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        else:
                            frame_resized = frame.copy()
                            scale = 1.0
                        
                        # Additional validation after resize
                        if frame_resized.shape[0] < 48 or frame_resized.shape[1] < 48:
                            print("Resized frame too small, skipping...")
                            detection_results = last_detection_results
                            continue
                        
                        # Convert BGR to RGB for MTCNN
                        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        
                        # Ensure the frame is contiguous in memory
                        rgb_frame = np.ascontiguousarray(rgb_frame)
                        
                        # Detect faces with timeout protection
                        result = detector.detect_faces(rgb_frame)
                        
                        # Validate detection results
                        if result is None:
                            result = []
                        
                        # Scale back coordinates if frame was resized
                        if scale != 1.0 and len(result) > 0:
                            for face in result:
                                if 'box' in face and len(face['box']) >= 4:
                                    face['box'][0] = int(face['box'][0] / scale)
                                    face['box'][1] = int(face['box'][1] / scale)
                                    face['box'][2] = int(face['box'][2] / scale)
                                    face['box'][3] = int(face['box'][3] / scale)
                                    
                                    if 'keypoints' in face:
                                        for key, (x, y) in face['keypoints'].items():
                                            face['keypoints'][key] = (int(x / scale), int(y / scale))
                        
                        detection_results = result
                        last_detection_results = result.copy() if result else []
                        
                    except Exception as e:
                        print(f"Error during face detection: {e}")
                        # Use last known good results instead of empty list
                        detection_results = last_detection_results
                else:
                    # Use previous results for non-processing frames
                    detection_results = last_detection_results
        
        time.sleep(0.033)  # ~30 FPS

def generate_frames():
    """Generate frames for video streaming"""
    global current_frame, detection_results
    
    while detection_active:
        if current_frame is not None:
            # Create visualization
            vis_frame = visualize_faces(current_frame, detection_results)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', vis_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/start_detection')
def start_detection():
    """Start face detection"""
    global camera, detection_active
    
    if not detection_active:
        try:
            camera = cv2.VideoCapture(app.config['CAMERA_INDEX'])
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, app.config['CAMERA_WIDTH'])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, app.config['CAMERA_HEIGHT'])
            camera.set(cv2.CAP_PROP_FPS, app.config['CAMERA_FPS'])
            
            if camera.isOpened():
                detection_active = True
                # Start background thread for frame capture
                threading.Thread(target=capture_frames, daemon=True).start()
                return jsonify({"status": "success", "message": "Detection started"})
            else:
                return jsonify({"status": "error", "message": "Could not open camera"})
                
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error starting detection: {e}"})
    
    return jsonify({"status": "info", "message": "Detection already active"})

@app.route('/stop_detection')
def stop_detection():
    """Stop face detection"""
    global camera, detection_active
    
    detection_active = False
    
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_status')
def detection_status():
    """Get current detection status"""
    global detection_results
    
    status = {
        "active": detection_active,
        "faces_detected": len(detection_results),
        "faces": []
    }
    
    for i, face in enumerate(detection_results):
        face_info = {
            "id": i + 1,
            "confidence": face['confidence'],
            "box": face['box']
        }
        
        # Check authorization status
        if current_frame is not None:
            is_authorized = face_recognition_sys.is_authorized_person(current_frame, face)
            face_info["authorized"] = is_authorized
        
        status["faces"].append(face_info)
    
    return jsonify(status)

@app.route('/add_user', methods=['POST'])
def add_user():
    """Add a new authorized user"""
    try:
        name = request.form.get('name')
        if not name:
            return jsonify({"status": "error", "message": "Name is required"})
        
        # Sanitize name for filename (remove/replace problematic characters)
        import re
        safe_name = re.sub(r'[^\w\s-]', '', name.strip())
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        
        # If safe_name is empty after sanitization, use a default
        if not safe_name:
            safe_name = f"user_{int(time.time())}"
        
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image file uploaded"})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No image file selected"})
        
        # Create data directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save uploaded file with safe filename
        timestamp = int(time.time())
        filename = f"{safe_name}_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure the filepath uses forward slashes for cross-platform compatibility
        filepath = filepath.replace('\\', '/')
        
        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")
        except Exception as save_error:
            print(f"Error saving file: {save_error}")
            return jsonify({"status": "error", "message": "Failed to save uploaded file"})
        
        # Verify file was saved and can be read
        if not os.path.exists(filepath):
            return jsonify({"status": "error", "message": "File was not saved properly"})
        
        # Add user to recognition system
        success = face_recognition_sys.add_authorized_user(name, filepath)
        
        if success:
            return jsonify({"status": "success", "message": f"User {name} added successfully"})
        else:
            # Clean up the file if face recognition failed
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({"status": "error", "message": "Failed to process face in image"})
            
    except Exception as e:
        print(f"Error in add_user: {e}")
        return jsonify({"status": "error", "message": f"Error adding user: {str(e)}"})

if __name__ == '__main__':
    # Initialize models
    if initialize_models():
        print("="*50)
        print("🚀 Face Detection Flask Server")
        print("="*50)
        print(f"🌐 Server URL: http://{app.config['HOST']}:{app.config['PORT']}")
        print(f"📷 Camera: Index {app.config['CAMERA_INDEX']} ({app.config['CAMERA_WIDTH']}x{app.config['CAMERA_HEIGHT']})")
        print(f"🔍 Detection: MTCNN + OpenCV")
        print(f"💾 Data directory: {app.config['UPLOAD_FOLDER']}")
        print("="*50)
        print("💡 Open your browser and go to the server URL to use the system")
        print("⏹️  Press Ctrl+C to stop the server")
        print("="*50)
        
        app.run(
            debug=app.config['DEBUG'],
            host=app.config['HOST'],
            port=app.config['PORT'],
            threaded=app.config['THREADED']
        )
    else:
        print("❌ Failed to initialize models. Please check your installation.")
        print("💡 Make sure you have installed all requirements: pip install -r requirements.txt")