# main.py
import cv2
import numpy as np
from modules import database_manager, ai_core, ui_manager

# --- Trạng thái toàn cục của ứng dụng ---
full_db = None
faces_in_current_frame = []
selected_face_obj = None


def mouse_callback(event, x, y, flags, param):
    """Hàm xử lý click, cập nhật trạng thái selected_face_obj."""
    global selected_face_obj
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_face_obj = None
        for face in faces_in_current_frame:
            fx, fy, fw, fh = face['box']
            if fx <= x <= fx + fw and fy <= y <= fy + fh:
                selected_face_obj = face
                break


def handle_registration():
    """Xử lý logic đăng ký người mới."""
    global selected_face_obj, full_db

    if selected_face_obj and selected_face_obj['identity'] == "Unknown":
        print("Bắt đầu quá trình đăng ký...")
        user_input = ui_manager.get_user_input_dialog()

        if user_input:
            new_person_data = database_manager.register_new_person(
                person_id=user_input['id'],
                name=user_input['name'],
                extra_info=user_input['extra_info'],
                cropped_face_image=selected_face_obj['raw_face'],
                embedding=selected_face_obj['embedding']
            )
            full_db[user_input['id']] = new_person_data
            print("Đã cập nhật CSDL trong bộ nhớ.")
            selected_face_obj = None


def main():
    global full_db, faces_in_current_frame, selected_face_obj

    full_db = database_manager.load_full_database()
    if not full_db:
        return

    cap = cv2.VideoCapture(0)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = ui_manager.UIManager("Face Recognition Pro", cam_w, cam_h)
    ui.set_mouse_callback(mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces_in_current_frame.clear()

        try:
            extracted_faces = ai_core.extract_faces(frame.copy())
            for face_obj in extracted_faces:
                facial_area = face_obj['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                cropped_face = face_obj['face']

                current_embedding = ai_core.DeepFace.represent(
                    img_path=cropped_face, model_name=ai_core.MODEL_NAME, enforce_detection=False, detector_backend='skip')[0]['embedding']
                matched_id = ai_core.find_identity(current_embedding, full_db)

                identity_name = full_db[matched_id]['name'] if matched_id else "Unknown"

                faces_in_current_frame.append({
                    'box': (x, y, w, h),
                    'id': matched_id,
                    'identity': identity_name,
                    'raw_face': (cropped_face * 255).astype(np.uint8),
                    'embedding': current_embedding
                })
        except Exception as e:
            # Bỏ qua nếu không có khuôn mặt hoặc có lỗi từ deepface
            pass

        canvas = ui.create_canvas()
        ui.draw_camera_feed(canvas, frame)

        selected_id_to_draw = selected_face_obj['id'] if selected_face_obj and selected_face_obj.get(
            'id') else None
        ui.draw_face_boxes(canvas, faces_in_current_frame, selected_id_to_draw)

        selected_person_info = full_db.get(
            selected_id_to_draw) if selected_id_to_draw else None
        is_unknown_selected = selected_face_obj and selected_face_obj['identity'] == 'Unknown'
        ui.draw_info_panel(canvas, selected_person_info, is_unknown_selected)

        ui.show(canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            handle_registration()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
