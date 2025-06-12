# enroll.py
import os
import json
from modules import database_manager
from modules import ai_core


def run_enrollment():
    identities_path = "data/identities.json"
    known_faces_dir = "data/known_faces"

    try:
        with open(identities_path, 'r', encoding='utf-8') as f:
            identities = json.load(f)
    except FileNotFoundError:
        print(
            f"Lỗi: Không tìm thấy file '{identities_path}'. Vui lòng tạo file này trước.")
        return

    embeddings_data = {}
    for person in identities:
        person_id = person['id']
        image_file = person['image_file']
        image_path = os.path.join(known_faces_dir, image_file)

        if not os.path.exists(image_path):
            print(
                f"Cảnh báo: Không tìm thấy file ảnh '{image_path}' cho {person['name']}. Bỏ qua.")
            continue

        print(f"Đang xử lý {person['name']}...")
        embedding = ai_core.generate_embedding(image_path)
        if embedding:
            embeddings_data[person_id] = embedding

    database_manager.save_embeddings(embeddings_data)


if __name__ == "__main__":
    run_enrollment()
