# modules/database_manager.py
import json
import os
import cv2

IDENTITIES_FILE = "data/identities.json"
EMBEDDINGS_FILE = "data/embeddings.json"


def load_full_database():
    """Hợp nhất thông tin từ identities.json và embeddings.json vào một cấu trúc duy nhất."""
    try:
        with open(IDENTITIES_FILE, 'r', encoding='utf-8') as f:
            identities = json.load(f)
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp cơ sở dữ liệu. {e}")
        print("Vui lòng chạy 'enroll.py' trước.")
        return None

    # Tạo một dictionary để truy cập nhanh bằng ID
    full_database = {}
    for person in identities:
        person_id = person['id']
        if person_id in embeddings:
            person['embedding'] = embeddings[person_id]
            # Thêm đường dẫn đầy đủ đến ảnh
            person['image_path'] = os.path.join(
                'data', 'known_faces', person['image_file'])
            full_database[person_id] = person

    print(f"Đã tải thành công cơ sở dữ liệu với {len(full_database)} người.")
    return full_database


def save_embeddings(embeddings_data):
    """Lưu dữ liệu embeddings vào file."""
    with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=4)
    print(f"Đã lưu thành công embeddings vào '{EMBEDDINGS_FILE}'")


def register_new_person(person_id, name, extra_info, cropped_face_image, embedding):
    """Đăng ký một người mới vào cơ sở dữ liệu."""

    # --- 1. Lưu ảnh khuôn mặt đã được cắt ---
    image_filename = f"{person_id}.jpg"
    image_save_path = os.path.join('data', 'known_faces', image_filename)
    # Lưu ảnh, chất lượng JPEG là 95
    cv2.imwrite(image_save_path, cropped_face_image)
    print(f"Đã lưu ảnh cho {name} tại {image_save_path}")

    # --- 2. Cập nhật file identities.json ---
    identities_path = "data/identities.json"
    try:
        with open(identities_path, 'r+', encoding='utf-8') as f:
            identities = json.load(f)
            new_person_info = {
                "id": person_id,
                "name": name,
                "image_file": image_filename,
                "extra_info": extra_info
            }
            identities.append(new_person_info)
            # Di chuyển con trỏ về đầu file để ghi đè
            f.seek(0)
            json.dump(identities, f, indent=4, ensure_ascii=False)
            f.truncate()
    except FileNotFoundError:
        # Nếu file chưa tồn tại, tạo mới
        with open(identities_path, 'w', encoding='utf-8') as f:
            json.dump([new_person_info], f, indent=4, ensure_ascii=False)

    # --- 3. Cập nhật file embeddings.json ---
    embeddings_path = "data/embeddings.json"
    try:
        with open(embeddings_path, 'r+', encoding='utf-8') as f:
            embeddings = json.load(f)
            embeddings[person_id] = embedding
            f.seek(0)
            json.dump(embeddings, f, indent=4)
            f.truncate()
    except FileNotFoundError:
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            json.dump({person_id: embedding}, f, indent=4)

    print(f"Đã đăng ký thành công {name} vào cơ sở dữ liệu.")
    # Trả về thông tin đầy đủ của người mới để cập nhật vào cache
    new_person_info['embedding'] = embedding
    new_person_info['image_path'] = image_save_path
    return new_person_info
