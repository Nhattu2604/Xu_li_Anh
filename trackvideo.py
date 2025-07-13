import numpy as np
import cv2
import time
import argparse

# Xử lý đối số video tùy chọn với argparse
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# Xác định ngưỡng màu cho xanh (có thể điều chỉnh nếu cần)
BlueLower = np.array([100, 67, 0], dtype="uint8")  # Điều chỉnh dựa trên ánh sáng
BlueUpper = np.array([255, 128, 50], dtype="uint8") 

# Thử mở tệp video bằng cv2.VideoCapture()
try:
  camera = cv2.VideoCapture("e:/py/videos/mausac.mp4")
except FileNotFoundError:
  print("Lỗi: Tệp video không được tìm thấy!")
  exit()

# Vòng lặp xử lý chính
while True:
    # Đọc một khung hình từ video
    (grabbed, frame) = camera.read()
    # Kiểm tra xem khung hình có được lấy thành công hay không
    if not grabbed:
        break
    # Áp dụng ngưỡng màu cho pixel xanh và tạo thành hình ảnh nhị phân 
    blue = cv2.inRange(frame, BlueLower, BlueUpper)
    # Áp dụng Gaussian blur để giảm nhiễu để làm mờ hình ảnh bằng cách làm mờ các biên cạnh và chi tiết nhỏ
    #kernel có kích thước là (3, 3)
    blue = cv2.GaussianBlur(blue, (3, 3), 0)
    # Tìm đường viền trong có ngưỡng
 #opencv mới chỉ trả về 2 giá trị contours: Một danh sách các đường viền tìm được.
                                  #hierarchy: Thông tin về cấu trúc phân cấp của các đường viền.,
    #thay vi (_,cnt,_) ta chỉ (cnt,_)
    (cnts,_) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Tìm đường viền lớn nhất và vẽ hình chữ nhật màu xanh xung quanh nó (nếu có)
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
        cv2.drawContours(frame, [rect], -1, (0, 255, 0),5)
        #cv2.drawContours() được sử dụng để vẽ hình chữ nhật bao quanh contour lên khung hình gốc (frame). 
        # Đối số thứ hai là danh sách chứa các điểm đỉnh của hình chữ nhật, trong trường hợp này chỉ có một hình chữ nhật nên được đưa vào dưới dạng một danh sách con [rect].
        # Đối số thứ ba à chỉ số contour muốn vẽ. Nếu đặt giá trị này là -1, tất cả các contours trong danh sách được vẽ. Trong trường hợp này, sẽ vẽ tất cả các contours trong danh sách.
        # Đối số thứ tư (0, 255, 0) đại diện cho màu của đường viền (màu xanh lá cây trong trường hợp này), 
        # và đối số cuối cùng là độ dày của đường viền.
    # Hiển thị khung hình gốc và hình ảnh nhị phân
    cv2.imshow("Tracking", frame)
    cv2.imshow("Binary", blue)
    # Xử lý đầu vào của người dùng để thoát
    time.sleep(0.010)
      # Thêm độ trễ nhẹ giữa các khung hình (tùy chọn)
    # nhấn q để thoát ra màn hình chính
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Giải phóng tài nguyên camera và đóng tất cả các cửa sổ đang mở
camera.release()
cv2.destroyAllWindows()