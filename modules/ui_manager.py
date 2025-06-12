# modules/ui_manager.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog


class UIManager:
    def __init__(self, window_name, cam_w, cam_h):
        self.window_name = window_name
        self.cam_w = cam_w
        self.cam_h = cam_h

        self.info_panel_width = 400
        self.canvas_w = self.cam_w + self.info_panel_width
        self.canvas_h = self.cam_h

        self.BG_COLOR = (80, 80, 80)
        self.TEXT_COLOR = (255, 255, 255)
        self.KNOWN_COLOR = (0, 255, 0)
        self.UNKNOWN_COLOR = (0, 0, 255)
        self.PROMPT_COLOR = (255, 255, 0)
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.namedWindow(self.window_name)

    def set_mouse_callback(self, callback_func):
        cv2.setMouseCallback(self.window_name, callback_func)

    def create_canvas(self):
        return np.full((self.canvas_h, self.canvas_w, 3), self.BG_COLOR, dtype=np.uint8)

    def draw_camera_feed(self, canvas, frame):
        canvas[0:self.cam_h, 0:self.cam_w] = frame

    def draw_face_boxes(self, canvas, faces_data, selected_id):
        """
        Vẽ tất cả các bounding box và tên.

        Args:
            canvas (np.ndarray): Canvas để vẽ lên.
            faces_data (list): Danh sách các dictionary chứa thông tin khuôn mặt.
            selected_id (str or None): ID của người đang được chọn.
        """
        for face in faces_data:
            x, y, w, h = face['box']
            identity = face['identity']

            is_known = identity != "Unknown"
            is_selected = face.get('id') is not None and face.get(
                'id') == selected_id

            color = self.KNOWN_COLOR if is_known else self.UNKNOWN_COLOR
            thickness = 4 if is_selected else 2

            cv2.rectangle(canvas, (x, y), (x+w, y+h), color, thickness)

            (text_w, text_h), _ = cv2.getTextSize(identity, self.FONT, 0.7, 2)
            cv2.rectangle(canvas, (x, y - text_h - 15),
                          (x + text_w + 10, y), color, -1)
            cv2.putText(canvas, identity, (x + 5, y - 10),
                        self.FONT, 0.7, (0, 0, 0), 2)

    def draw_info_panel(self, canvas, person_info, is_unknown_selected=False):
        info_x_start = self.cam_w + 20

        if person_info:
            try:
                profile_img = cv2.imread(person_info['image_path'])
                profile_img = cv2.resize(profile_img, (200, 200))
                canvas[40:240, info_x_start:info_x_start+200] = profile_img
            except Exception:
                cv2.rectangle(canvas, (info_x_start, 40),
                              (info_x_start+200, 240), (0, 0, 0), -1)
                cv2.putText(canvas, "Image Error", (info_x_start+40,
                            140), self.FONT, 0.7, self.TEXT_COLOR, 1)

            cv2.putText(canvas, "Name:", (info_x_start, 290),
                        self.FONT, 0.8, self.TEXT_COLOR, 2)
            cv2.putText(
                canvas, person_info['name'], (info_x_start, 320), self.FONT, 1, self.KNOWN_COLOR, 2)
            cv2.putText(canvas, "Extra Info:", (info_x_start, 370),
                        self.FONT, 0.8, self.TEXT_COLOR, 2)
            cv2.putText(canvas, person_info['extra_info'],
                        (info_x_start, 400), self.FONT, 0.8, self.TEXT_COLOR, 2)

        elif is_unknown_selected:
            cv2.putText(canvas, "Unknown Person", (info_x_start, 50),
                        self.FONT, 1, self.PROMPT_COLOR, 2)
            cv2.putText(canvas, "Press 's' to register",
                        (info_x_start, 90), self.FONT, 0.9, self.PROMPT_COLOR, 2)
        else:
            cv2.putText(canvas, "Click on a face", (info_x_start,
                        50), self.FONT, 0.9, self.TEXT_COLOR, 2)
            cv2.putText(canvas, "to see details.", (info_x_start,
                        80), self.FONT, 0.9, self.TEXT_COLOR, 2)

    def show(self, canvas):
        cv2.imshow(self.window_name, canvas)


def get_user_input_dialog():
    root = tk.Tk()
    root.withdraw()

    person_id = simpledialog.askstring(
        "Register - Step 1/3", "Enter a unique ID (e.g., person_003):", parent=root)
    if not person_id:
        print("Registration cancelled: ID is required.")
        return None

    name = simpledialog.askstring(
        "Register - Step 2/3", "Enter name:", parent=root)
    if not name:
        print("Registration cancelled: Name is required.")
        return None

    extra_info = simpledialog.askstring(
        "Register - Step 3/3", "Enter extra info (optional):", parent=root)
    if extra_info is None:
        print("Registration cancelled by user.")
        return None

    return {"id": person_id, "name": name, "extra_info": extra_info}
