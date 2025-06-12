# Hệ thống Nhận diện Khuôn mặt Thời gian thực

Đây là một dự án ứng dụng nhận diện khuôn mặt trong thời gian thực bằng Python. Ứng dụng sử dụng camera để phát hiện các khuôn mặt, so sánh với một cơ sở dữ liệu có sẵn để xác định danh tính, và cho phép đăng ký người mới trực tiếp từ giao diện.

  <!-- Thay thế link này bằng ảnh chụp màn hình thực tế của ứng dụng bạn -->

## Tính năng chính

- **Nhận diện thời gian thực**: Sử dụng webcam để liên tục phát hiện và nhận diện khuôn mặt.
- **Cơ sở dữ liệu cục bộ**: Quản lý thông tin người dùng qua các tệp JSON và thư mục ảnh cục bộ.
- **Giao diện tương tác**:
    - Hiển thị tên và hộp bao quanh các khuôn mặt được nhận diện.
    - Cho phép click vào một khuôn mặt để xem thông tin chi tiết (tên, ảnh gốc, thông tin phụ).
- **Đăng ký người mới**: Dễ dàng thêm người mới vào cơ sở dữ liệu bằng cách chọn một khuôn mặt "Unknown" và nhấn phím 's' để nhập thông tin qua cửa sổ pop-up.
- **Cấu trúc module hóa**: Mã nguồn được chia thành các module riêng biệt (AI, UI, Database) giúp dễ dàng bảo trì và mở rộng.

## Công nghệ sử dụng

- **Python 3.10+**
- **OpenCV**: Để xử lý hình ảnh, video và tạo giao diện chính.
- **DeepFace**: Một thư viện mạnh mẽ để xử lý các tác vụ liên quan đến khuôn mặt, bao gồm:
    - **Phát hiện khuôn mặt**: Sử dụng backend `mediapipe`.
    - **Trích xuất đặc trưng (Embedding)**: Sử dụng mô hình `ArcFace`.
- **Tkinter**: Để tạo các cửa sổ pop-up nhập liệu.
- **NumPy**: Để thực hiện các phép toán trên mảng và hình ảnh.

## Cấu trúc thư mục

Dự án được tổ chức theo cấu trúc module hóa để đảm bảo sự rõ ràng và dễ quản lý:

```
face_recognition_pro/
├── data/
│   ├── known_faces/         # Chứa ảnh gốc của những người đã biết
│   ├── identities.json      # (Do bạn điền) Chứa thông tin danh tính
│   └── embeddings.json      # (Do AI tạo) Chứa đặc trưng khuôn mặt
│
├── modules/
│   ├── ai_core.py           # Logic xử lý AI (DeepFace)
│   ├── database_manager.py  # Quản lý đọc/ghi dữ liệu
│   └── ui_manager.py        # Quản lý giao diện (OpenCV, Tkinter)
│
├── enroll.py                # Script chạy 1 lần để tạo CSDL embeddings
├── main.py                  # Điểm khởi đầu của ứng dụng
├── requirements.txt         # Danh sách các thư viện cần thiết
└── README.md                # Tài liệu hướng dẫn này
```

## Hướng dẫn cài đặt và sử dụng

### Bước 1: Clone dự án

Clone kho lưu trữ này về máy tính của bạn:
```bash
git clone https://your-repo-url.com/face_recognition_pro.git
cd face_recognition_pro
```

### Bước 2: Tạo và kích hoạt Môi trường ảo

Rất khuyến khích sử dụng một môi trường ảo để tránh xung đột thư viện.
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết

Cài đặt tất cả các gói phụ thuộc từ file `requirements.txt`.
```bash
pip install -r requirements.txt
```
*(Lưu ý: Quá trình cài đặt `deepface` và `tensorflow` có thể mất một chút thời gian).*

### Bước 4: Chuẩn bị Cơ sở dữ liệu

Đây là bước quan trọng nhất trước khi chạy ứng dụng.

1.  **Thêm ảnh gốc**: Đặt các file ảnh (định dạng `.jpg` hoặc `.png`) của những người bạn muốn nhận diện vào thư mục `data/known_faces/`.

2.  **Cập nhật thông tin danh tính**: Mở file `data/identities.json` và điền thông tin cho từng người theo mẫu sau. Đảm bảo `id` là duy nhất và `image_file` khớp chính xác với tên file ảnh bạn đã thêm.

    ```json
    [
      {
        "id": "person_001",
        "name": "Elon Musk",
        "image_file": "elon_musk.jpg",
        "extra_info": "CEO of Tesla, SpaceX"
      },
      {
        "id": "person_002",
        "name": "Bill Gates",
        "image_file": "bill_gates.png",
        "extra_info": "Co-founder of Microsoft"
      }
    ]
    ```

### Bước 5: "Huấn luyện" - Tạo CSDL Embeddings

Chạy script `enroll.py` để phân tích các ảnh và tạo ra file `embeddings.json`. **Bạn cần chạy lại script này mỗi khi thêm/xóa người trong `identities.json`**.
```bash
python enroll.py
```

### Bước 6: Chạy ứng dụng chính

Sau khi đã chuẩn bị xong dữ liệu, hãy khởi động ứng dụng:
```bash
python main.py
```
- Một cửa sổ sẽ hiện lên hiển thị hình ảnh từ webcam của bạn.
- Nhấn phím **'q'** để thoát ứng dụng.
- Nhấn phím **'s'** để lưu thông tin khi một khuôn mặt "Unknown" đang được chọn.

