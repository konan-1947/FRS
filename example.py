from mtcnn import MTCNN
import cv2
import numpy as np

# Tạo detector
detector = MTCNN(device="CPU:0")

# Đọc ảnh bằng cv2 (chuẩn BGR, uint8) -> an toàn nhất
image = cv2.imread("image.png")

# Nếu vẫn muốn dùng load_image (RGB float32 [0,1]) thì phải convert lại:
# image = load_image("image.png")
# image = (image * 255).astype(np.uint8)      # về uint8
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # sang BGR

# Detect faces
result = detector.detect_faces(image)
print(result)


def visualize_faces(image, detection_results):
    vis_image = image.copy()

    for face in detection_results:
        x, y, width, height = face['box']
        cv2.rectangle(vis_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        confidence = face['confidence']
        cv2.putText(vis_image, f'Conf: {confidence:.3f}',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

        if 'keypoints' in face:
            keypoints = face['keypoints']
            for (kx, ky) in keypoints.values():
                cv2.circle(vis_image, (int(kx), int(ky)), 3, (0, 0, 255), -1)

    return vis_image


# Visualize
if result:
    vis_image = visualize_faces(image, result)
    cv2.namedWindow("Detected Faces (OpenCV)", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Detected Faces (OpenCV)", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No faces detected")
