import cv2
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognitionSystem:
    def __init__(self):
        """Initialize face recognition system using OpenCV"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.authorized_faces = []
        self.authorized_names = []
        self.encodings_file = 'data/face_encodings.pkl'
        self.threshold = 0.8  # Similarity threshold for recognition
        
        # Load existing encodings if available
        self.load_encodings()
    
    def extract_face_features(self, face_img):
        """
        Extract simple features from face image using OpenCV
        
        Args:
            face_img: Face image (cropped)
            
        Returns:
            Feature vector or None if failed
        """
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_img, (100, 100))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better features
            gray = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Flatten to create feature vector
            features = gray.flatten().astype(np.float32)
            
            # Normalize features
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting face features: {e}")
            return None
    
    def add_authorized_user(self, name, image_path):
        """
        Add a new authorized user to the system
        
        Args:
            name: User's name
            image_path: Path to user's photo
            
        Returns:
            Boolean indicating success
        """
        try:
            print(f"Attempting to load image: {image_path}")
            
            # Handle Unicode file paths by reading as binary and decoding
            try:
                # Method 1: Try direct imread first
                image = cv2.imread(image_path)
                
                # Method 2: If direct imread fails, try reading as binary
                if image is None:
                    import numpy as np
                    # Read file as binary and decode
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                    
                    # Convert binary data to numpy array
                    nparr = np.frombuffer(image_data, np.uint8)
                    # Decode image
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    print(f"Could not load image: {image_path}")
                    print("Possible issues: file corruption, unsupported format, or encoding problems")
                    return False
                    
            except Exception as load_error:
                print(f"Error loading image {image_path}: {load_error}")
                return False
            
            print(f"Image loaded successfully: {image.shape}")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using Haar cascade
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            if len(faces) == 0:
                print(f"No face found in image: {image_path}")
                return False
            
            if len(faces) > 1:
                print(f"Multiple faces found in image: {image_path}. Using the largest one.")
            
            # Use the largest face detected
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            print(f"Face detected at: x={x}, y={y}, w={w}, h={h}")
            
            # Extract face region with some padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0:
                print(f"Extracted face image is empty")
                return False
            
            # Extract features
            features = self.extract_face_features(face_img)
            if features is None:
                print(f"Could not extract features from face in: {image_path}")
                return False
            
            # Add to authorized list
            self.authorized_faces.append(features)
            self.authorized_names.append(name)
            
            # Save encodings
            self.save_encodings()
            
            print(f"✅ Added authorized user: {name}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding authorized user {name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_authorized_person(self, frame, face_info):
        """
        Check if detected face belongs to an authorized person
        
        Args:
            frame: Input image frame
            face_info: Face detection result from MTCNN
            
        Returns:
            Boolean indicating if person is authorized
        """
        try:
            if len(self.authorized_faces) == 0:
                return False  # No authorized users registered
            
            # Extract face region
            x, y, width, height = face_info['box']
            
            # Add padding and ensure bounds
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + width + padding)
            y2 = min(frame.shape[0], y + height + padding)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return False
            
            # Extract features from detected face
            features = self.extract_face_features(face_img)
            if features is None:
                return False
            
            # Compare with authorized faces using cosine similarity
            similarities = []
            for auth_features in self.authorized_faces:
                try:
                    similarity = cosine_similarity([features], [auth_features])[0][0]
                    similarities.append(similarity)
                except:
                    similarities.append(0.0)
            
            # Check if any similarity is above threshold
            if len(similarities) > 0:
                max_similarity = np.max(similarities)
                
                if max_similarity > self.threshold:
                    best_match_index = np.argmax(similarities)
                    recognized_name = self.authorized_names[best_match_index]
                    print(f"Authorized person detected: {recognized_name} (similarity: {max_similarity:.3f})")
                    return True
                else:
                    print(f"Unauthorized person detected (max similarity: {max_similarity:.3f})")
                    return False
            else:
                return False
                
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return False
    
    def recognize_person(self, frame, face_info):
        """
        Recognize the person in the detected face
        
        Args:
            frame: Input image frame
            face_info: Face detection result from MTCNN
            
        Returns:
            Tuple of (name, confidence) or (None, 0) if not recognized
        """
        try:
            if len(self.authorized_faces) == 0:
                return None, 0
            
            # Extract face region
            x, y, width, height = face_info['box']
            
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + width + padding)
            y2 = min(frame.shape[0], y + height + padding)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None, 0
            
            # Extract features
            features = self.extract_face_features(face_img)
            if features is None:
                return None, 0
            
            # Compare with authorized faces
            similarities = []
            for auth_features in self.authorized_faces:
                try:
                    similarity = cosine_similarity([features], [auth_features])[0][0]
                    similarities.append(similarity)
                except:
                    similarities.append(0.0)
            
            # Find best match
            if len(similarities) > 0:
                max_similarity = np.max(similarities)
                best_match_index = np.argmax(similarities)
                
                if max_similarity > self.threshold:
                    name = self.authorized_names[best_match_index]
                    return name, max_similarity
                else:
                    return None, 0
            else:
                return None, 0
                
        except Exception as e:
            print(f"Error in person recognition: {e}")
            return None, 0
    
    def save_encodings(self):
        """Save face encodings to file"""
        try:
            os.makedirs('data', exist_ok=True)
            data = {
                'faces': self.authorized_faces,
                'names': self.authorized_names
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print("Face encodings saved successfully")
        except Exception as e:
            print(f"Error saving encodings: {e}")
    
    def load_encodings(self):
        """Load face encodings from file"""
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                self.authorized_faces = data.get('faces', [])
                self.authorized_names = data.get('names', [])
                print(f"Loaded {len(self.authorized_faces)} authorized users")
            else:
                print("No existing encodings file found")
        except Exception as e:
            print(f"Error loading encodings: {e}")
            self.authorized_faces = []
            self.authorized_names = []
    
    def remove_authorized_user(self, name):
        """Remove an authorized user from the system"""
        try:
            if name in self.authorized_names:
                index = self.authorized_names.index(name)
                del self.authorized_names[index]
                del self.authorized_faces[index]
                self.save_encodings()
                print(f"Removed authorized user: {name}")
                return True
            else:
                print(f"User not found: {name}")
                return False
        except Exception as e:
            print(f"Error removing user {name}: {e}")
            return False
    
    def get_authorized_users(self):
        """Get list of authorized users"""
        return self.authorized_names.copy()
    
    def clear_all_users(self):
        """Clear all authorized users"""
        self.authorized_faces = []
        self.authorized_names = []
        self.save_encodings()
        print("All authorized users cleared")