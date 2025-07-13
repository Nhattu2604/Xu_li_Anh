import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib

# plt.rcParams.update({"text.usetex":True})
# matplotlib.use('pgf')

# Đọc ảnh
img = cv2.imread('test_pattern.tif',0)
m,n = img.shape[:2]

# Thêm nhiễu Gaussian vào ảnh img
gia_tri_TB = 10
phuong_sai = 25
noise = np.random.normal(loc=gia_tri_TB,scale=phuong_sai,size=(m,n))
Gau_noisy_img = img + noise

# Thêm nhiễu Rayleigh vào ảnh img
phuong_sai = 40
noise = np.random.rayleigh(scale=phuong_sai,size=(m,n))
Ray_noisy_img = img + noise

# Thêm nhiễu Erlang (Gammar) vào ảnh img
K = 2.0
phuong_sai = 18
noise = np.random.gamma(shape=K,scale=phuong_sai,size=(m,n))
Gam_noisy_img = img + noise

# Thêm nhiễu hàm mũ vào ảnh img
phuong_sai = 26
noise = np.random.exponential(scale=phuong_sai,size=(m,n))
Exp_noisy_img = img + noise

# Thêm nhiễu Uniform vào ảnh img
a,b = 10,100
noise = np.random.uniform(low=a,high=b,size=(m,n))
Uni_noisy_img = img + noise

# Thêm nhiễu muối tiêu (add salt and pepper) vào ảnh img
number_black = int(m*n*0.05)  # định nghĩa số điểm đen
number_white = int(m*n*0.05)  # định nghĩa số điểm trắng
# Lấy giá trị nguyên ngẫu nhiên trong đoạn 0..m
# Giá trị này sẽ biểu diễn tọa độ điểm đen theo hàng
m_blacks = np.random.randint(0,m,number_black)
# Lấy giá trị nguyên ngẫu nhiên trong đoạn 0..n
# Giá trị này sẽ biểu diễn tọa độ điểm đen theo cột
n_blacks = np.random.randint(0,n,number_black)
# Lấy giá trị nguyên ngẫu nhiên trong đoạn 0..m
# Giá trị này sẽ biểu diễn tọa độ điểm trắng theo hàng
m_whites = np.random.randint(0,m,number_white)
# Lấy giá trị nguyên ngẫu nhiên trong đoạn 0..n
# Giá trị này sẽ biểu diễn tọa độ điểm trắng theo cột
n_whites = np.random.randint(0,n,number_white)

SP_noisy_img = np.copy(img) # Sao chép ảnh img để tạo ảnh SP_noisy_img
# Thiết lập mức xám = 0 (điểm đen) cho điểm ảnh có tọa độ (m_blacks,n_blacks)
SP_noisy_img[m_blacks,n_blacks] = 0
# Thiết lập mức xám = 255 (điểm trắng) cho điểm ảnh có tọa độ (m_whites,n_whites)
SP_noisy_img[m_whites,n_whites] = 255

# 1. Hiển thị ảnh gốc, ảnh nhiễu và histogram
# Tạo cửa số 1 để hiển thị ảnh cho nhiễu Gaussian
fig1 = plt.figure(figsize=(11, 9))  # Tạo vùng vẽ tỷ lệ 16:9
#Tạo 9 vùng vẽ con
(ax1, ax2), (ax3, ax4) = fig1.subplots(2, 2)
# Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')
# Hiển thị histogram ảnh gốc
ax2.hist(img.flatten(),bins=256)
ax2.set_title('Histogram')
# Hiển thị ảnh nhiễu Gaussian
ax3.imshow(Gau_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu Gaussian')
ax3.axis('off')
# Hiển thị histogram ảnh nhiễu gaussian
ax4.hist(Gau_noisy_img.flatten(),bins=256)
ax4.set_title('Hitogram')
plt.savefig("gaussian.pdf",bbox_inches='tight')
plt.show()

# 2. Tạo cửa số 2 để hiển thị ảnh cho nhiễu Rayleigh
fig2 = plt.figure(figsize=(11, 9))  # Tạo vùng vẽ tỷ lệ 16:9
#Tạo 9 vùng vẽ con
(ax1, ax2), (ax3, ax4) = fig2.subplots(2, 2)
# Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')
# Hiển thị histogram ảnh gốc
ax2.hist(img.flatten(),bins=256)
ax2.set_title('Hitogram')
# Hiển thị ảnh nhiễu Rayleigh
ax3.imshow(Ray_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu Rayleigh')
ax3.axis('off')
# Hiển thị histogram ảnh nhiễu Rayleigh
ax4.hist(Ray_noisy_img.flatten(),bins=256)
ax4.set_title('Histogram')
plt.savefig("rayleigh.pdf",bbox_inches='tight')
plt.show()

# 3. Tạo cửa số 3 để hiển thị ảnh cho nhiễu Erlang (Gamma)
fig2 = plt.figure(figsize=(11, 9))  # Tạo vùng vẽ tỷ lệ 16:9
#Tạo 9 vùng vẽ con
(ax1, ax2), (ax3, ax4) = fig2.subplots(2, 2)
# Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')
# Hiển thị histogram ảnh gốc
ax2.hist(img.flatten(),bins=256)
ax2.set_title('Hitogram')
# Hiển thị ảnh nhiễu Erlang (Gammar)
ax3.imshow(Gam_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu Erlang (Gammar)')
ax3.axis('off')
# Hiển thị histogram ảnh nhiễu Erlang (Gammar)
ax4.hist(Gam_noisy_img.flatten(),bins=256)
ax4.set_title('Histogram')
plt.savefig("Erlang.pdf",bbox_inches='tight')
plt.show()

# 4. Tạo cửa số 4 để hiển thị ảnh cho nhiễu nhiễu hàm mũ
fig2 = plt.figure(figsize=(11, 9))  # Tạo vùng vẽ tỷ lệ 16:9
#Tạo 4 vùng vẽ con
(ax1, ax2), (ax3, ax4) = fig2.subplots(2, 2)
# Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')
# Hiển thị histogram ảnh gốc
ax2.hist(img.flatten(),bins=256)
ax2.set_title('Hitogram')
# Hiển thị ảnh nhiễu nhiễu hàm mũ
ax3.imshow(Exp_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu hàm mũ')
ax3.axis('off')
# Hiển thị histogram ảnh nhiễu nhiễu hàm mũ
ax4.hist(Exp_noisy_img.flatten(),bins=256)
ax4.set_title('Histogram')
plt.savefig("exp.pdf",bbox_inches='tight')
plt.show()

# 5. Tạo cửa số 5 để hiển thị ảnh cho nhiễu nhiễu Uniform
fig2 = plt.figure(figsize=(11, 9))  # Tạo vùng vẽ tỷ lệ 16:9
#Tạo 4 vùng vẽ con
(ax1, ax2), (ax3, ax4) = fig2.subplots(2, 2)
# Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')
# Hiển thị histogram ảnh gốc
ax2.hist(img.flatten(),bins=256)
ax2.set_title('Hitogram')
# Hiển thị ảnh nhiễu nhiễu Uniform
ax3.imshow(Uni_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu Uniform')
ax3.axis('off')
# Hiển thị histogram ảnh nhiễu Uniform
ax4.hist(Uni_noisy_img.flatten(),bins=256)
ax4.set_title('Histogram')
plt.savefig("Uniform.pdf",bbox_inches='tight')
plt.show()

# 6. Tạo cửa số 6 để hiển thị ảnh cho nhiễu nhiễu muối tiêu (nhiễu xung)
fig2 = plt.figure(figsize=(11, 9))  # Tạo vùng vẽ tỷ lệ 16:9
#Tạo 4 vùng vẽ con
(ax1, ax2), (ax3, ax4) = fig2.subplots(2, 2)
# Hiển thị ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')
# Hiển thị histogram ảnh gốc
ax2.hist(img.flatten(),bins=256)
ax2.set_title('Hitogram')
# Hiển thị ảnh nhiễu nhiễu muối tiêu (nhiễu xung)
ax3.imshow(SP_noisy_img, cmap='gray')
ax3.set_title('Ảnh nhiễu muối tiêu')
ax3.axis('off')
# Hiển thị histogram ảnh nhiễu muối tiêu (nhiễu xung)
ax4.hist(SP_noisy_img.flatten(),bins=256)
ax4.set_title('Histogram')
plt.savefig("salt_pepper.pdf",bbox_inches='tight')
plt.show()

