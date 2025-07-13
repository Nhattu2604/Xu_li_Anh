import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

def calculate_histogram(image):
    histogram = np.zeros(256, dtype=int)  # Mảng lưu trữ giá trị histogram

    # Chuyển đổi ảnh sang dạng mảng numpy
    image_array = np.array(image)

    # Duyệt qua từng pixel trong ảnh
    for row in image_array:
        for pixel in row:
            gray_value = int(pixel[0] * 0.3 + pixel[1] * 0.59 + pixel[2] * 0.11)  # Chuyển đổi sang giá trị xám

            histogram[gray_value] += 1  # Tăng giá trị tương ứng trong histogram

    return histogram

def draw_histogram(histogram):
    plt.bar(range(256), histogram, color='gray')
    plt.xlabel('Gray Level')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

# Đường dẫn đến ảnh
image_path = 'image/trex.png'

# Đọc ảnh
image = Image.open(image_path)

# Tính toán histogram
histogram = calculate_histogram(image)

# Vẽ histogram
draw_histogram(histogram)
import cv2 
import matplotlib.pyplot as plt 