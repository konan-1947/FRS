#!/usr/bin/env python3
"""
Test script to verify MTCNN functionality and debug tensor shape issues
"""

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import time

# Configure TensorFlow to reduce verbosity
tf.get_logger().setLevel('ERROR')

# Configure GPU memory growth if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ GPU memory growth configured")
    except RuntimeError as e:
        print(f"⚠️ GPU memory growth setting failed: {e}")

def test_mtcnn_initialization():
    """Test MTCNN initialization"""
    print("🔄 Testing MTCNN initialization...")
    try:
        detector = MTCNN()
        print("✅ MTCNN initialized successfully")
        return detector
    except Exception as e:
        print(f"❌ MTCNN initialization failed: {e}")
        return None

def test_dummy_detection(detector):
    """Test detection with dummy images of various sizes"""
    print("\n🔄 Testing detection with dummy images...")
    
    test_sizes = [
        (48, 48),   # Minimum size
        (64, 64),   # Small
        (128, 128), # Medium
        (240, 320), # Common webcam ratio
        (480, 640), # Larger
    ]
    
    for width, height in test_sizes:
        try:
            # Create a dummy image with some variation
            dummy_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add a simple face-like pattern
            center_x, center_y = width // 2, height // 2
            cv2.circle(dummy_image, (center_x, center_y), min(width, height) // 6, (200, 180, 160), -1)
            
            print(f"  Testing {width}x{height} image...", end=" ")
            
            start_time = time.time()
            result = detector.detect_faces(dummy_image)
            end_time = time.time()
            
            print(f"✅ OK ({len(result)} faces, {(end_time - start_time)*1000:.1f}ms)")
            
        except Exception as e:
            print(f"❌ Failed: {e}")

def test_webcam_detection(detector):
    """Test detection with webcam frames"""
    print("\n🔄 Testing webcam detection...")
    
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print("📷 Webcam opened, testing 10 frames...")
        
        successful_detections = 0
        total_frames = 10
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Could not read frame {i+1}")
                continue
            
            try:
                # Resize frame to safe size
                height, width = frame.shape[:2]
                if width > 480:
                    scale = 480 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    # Ensure even dimensions and minimum size
                    new_width = max(48, new_width - (new_width % 2))
                    new_height = max(48, new_height - (new_height % 2))
                    
                    frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                rgb_frame = np.ascontiguousarray(rgb_frame)
                
                # Detect faces
                start_time = time.time()
                result = detector.detect_faces(rgb_frame)
                end_time = time.time()
                
                print(f"  Frame {i+1}: {len(result)} faces detected ({(end_time - start_time)*1000:.1f}ms)")
                successful_detections += 1
                
            except Exception as e:
                print(f"  Frame {i+1}: ❌ Detection failed: {e}")
        
        cap.release()
        
        success_rate = (successful_detections / total_frames) * 100
        print(f"\n📊 Detection success rate: {success_rate:.1f}% ({successful_detections}/{total_frames})")
        
        if success_rate >= 80:
            print("✅ Webcam detection test passed!")
        else:
            print("⚠️ Webcam detection test had issues")
            
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")

def main():
    """Main test function"""
    print("🧪 MTCNN Functionality Test")
    print("=" * 40)
    
    # Test initialization
    detector = test_mtcnn_initialization()
    if detector is None:
        print("\n❌ Cannot proceed with tests - MTCNN initialization failed")
        return
    
    # Test with dummy images
    test_dummy_detection(detector)
    
    # Test with webcam
    test_webcam_detection(detector)
    
    print("\n🏁 Test completed!")
    print("\n💡 If all tests passed, the Flask app should work correctly.")
    print("💡 If tests failed, check your MTCNN and TensorFlow installation.")

if __name__ == "__main__":
    main()