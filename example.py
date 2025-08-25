from mtcnn import MTCNN
import cv2
import numpy as np
import tensorflow as tf

# Disable TensorFlow warnings and set memory growth
tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Tạo detector with default configuration
try:
    detector = MTCNN()
    print("MTCNN detector initialized successfully")
except Exception as e:
    print(f"Error initializing MTCNN: {e}")
    exit()


def visualize_faces(image, detection_results):
    vis_image = image.copy()
    
    if not detection_results:
        # Add "No faces detected" text
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
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Draw confidence score
            confidence = face['confidence']
            cv2.putText(vis_image, f'Face {i+1}: {confidence:.3f}',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

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


# Mở webcam
cap = cv2.VideoCapture(0)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Camera opened successfully. Press 'q' to quit.")
frame_count = 0

while True:
    # Đọc từng frame
    ret, frame = cap.read()
    frame_count += 1

    # Nếu không đọc được frame thì thoát
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Process every 3rd frame for better performance
    if frame_count % 3 == 0:
        # Preprocess frame for better detection
        # Resize frame if too large
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640.0 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame.copy()
            scale = 1.0
        
        # Convert BGR to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Detect faces with error handling
        try:
            result = detector.detect_faces(rgb_frame)
            
            # Scale back coordinates if frame was resized
            if scale != 1.0:
                for face in result:
                    face['box'][0] = int(face['box'][0] / scale)
                    face['box'][1] = int(face['box'][1] / scale)
                    face['box'][2] = int(face['box'][2] / scale)
                    face['box'][3] = int(face['box'][3] / scale)
                    
                    if 'keypoints' in face:
                        for key, (x, y) in face['keypoints'].items():
                            face['keypoints'][key] = (int(x / scale), int(y / scale))
            
            if result:
                print(f"Detected {len(result)} face(s)")
            else:
                print("No faces detected")
                
        except Exception as e:
            print(f"Error during face detection: {e}")
            result = []
    else:
        # Use previous result for non-processing frames
        if 'result' not in locals():
            result = []

    # Visualize
    if result:
        vis_image = visualize_faces(frame, result)
    else:
        vis_image = frame  # Hiển thị frame gốc nếu không có khuôn mặt

    cv2.imshow("Detected Faces (OpenCV)", vis_image)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()

