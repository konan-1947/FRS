# modules/ai_core.py
from deepface import DeepFace
from deepface.modules.verification import find_cosine_distance
import numpy as np

MODEL_NAME = 'ArcFace'
DETECTOR_BACKEND = 'mediapipe'


def generate_embedding(image_path):
    """Tạo embedding cho một ảnh duy nhất."""
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )
        return embedding_objs[0]['embedding']
    except Exception as e:
        print(f"Lỗi khi tạo embedding cho {image_path}: {e}")
        return None


def find_identity(target_embedding, db, threshold=0.68):
    """Tìm danh tính bằng cách so sánh embedding."""
    target_embedding = np.array(target_embedding)
    min_dist = float('inf')
    matched_id = None

    for person_id, person_data in db.items():
        known_embedding = np.array(person_data['embedding'])
        dist = find_cosine_distance(target_embedding, known_embedding)

        if dist < min_dist:
            min_dist = dist
            if dist < threshold:
                matched_id = person_id

    return matched_id


def extract_faces(frame):
    """Trích xuất tất cả khuôn mặt từ một frame."""
    try:
        return DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
    except:
        return []
