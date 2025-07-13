''' SIMPLE THEROSHOLDING '''
import numpy as np
import matplotlib.pyplot as plt
import cv2

def cat_nguong_toan_cuc(image, T, inverse=False):
    """
    Hàm cắt ngưỡng toàn cục, ngưỡng T chọn bằng tay.
    """
    # Các tham số khi gọi phương thức cv2.threshold:
    # T là ngưỡng
    # 255 là cường độ sáng cao nhất
    # thresholding method: cv2.THRESH_BINARY -> nếu lớn hơn ngưỡng thì gán bằng cường độ sáng cao nhất;
    # cv2.THRESH_BINARY_INV -> ngược lại nếu lớn hơn ngưỡng thì gán bằng 0;
    if inverse:
        _, thresh = cv2.threshold(image, T, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
    return thresh

if __name__ == '__main__':
    img = cv2.imread('image/coins.png', 0)
    
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Áp dụng Gaussian blurring với bán kính bằng 5 để loại bỏ một vài cạnh có tần số cao mà chúng ta không quan tâm
    T = 155  # Thiết lập ngưỡng cho cắt ngưỡng toàn cục
    anh_cat_nguong = cat_nguong_toan_cuc(blurred, T)
    anh_cat_nguong_inv = cat_nguong_toan_cuc(blurred, T, inverse=True)
    mask_image = cv2.bitwise_and(img, img, mask=anh_cat_nguong_inv)

    # Vẽ kết quả bằng matplotlib
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 9))
    ax1.imshow(cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB))
    ax1.set_title("Ảnh gốc")

    ax2.imshow(cv2.cvtColor(anh_cat_nguong, cv2.COLOR_GRAY2RGB))
    ax2.set_title("Ảnh cắt ngưỡng T=155, cv2.THRESH_BINARY")

    ax3.imshow(cv2.cvtColor(anh_cat_nguong_inv, cv2.COLOR_GRAY2RGB))
    ax3.set_title("Ảnh cắt ngưỡng T=155, cv2.THRESH_BINARY_INV")

    ax4.imshow(cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB))
    ax4.set_title("Mask, cv2.THRESH_BINARY_INV")

    plt.savefig("simple_thresholding.pdf", bbox_inches='tight')
    plt.show()


'''ADAPTIVE THRESHOLDING'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

def cat_nguong_thich_nghi(image, neighborhood='mean', ksize=3, int_c=4):
    """
    Hàm cắt ngưỡng thích nghi, chọn ngưỡng T tối ưu.
    """
    # Các tham số khi gọi phương thức cv2.threshold:
    # 255 là cường độ sáng cao nhất
    # thresholding method: cv2.THRESH_BINARY -> nếu lớn hơn ngưỡng thì gán bằng cường độ sáng cao nhất;
    # cv2.THRESH_BINARY_INV -> ngược lại nếu lớn hơn ngưỡng thì gán bằng 0;
    if neighborhood == 'mean':
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ksize, int_c)
    elif neighborhood == 'gaussian':
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ksize, int_c)
    else:
        return image
    return thresh

if __name__ == '__main__':
    img = cv2.imread('image/coins.png', 0)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Áp dụng Gaussian blurring với bán kính bằng 5 để loại bỏ một vài cạnh có tần số cao mà chúng ta không quan tâm

    anh_cat_nguong_mean = cat_nguong_thich_nghi(blurred, 'mean', ksize=11, int_c=4)
    anh_cat_nguong_gauss = cat_nguong_thich_nghi(blurred, 'gaussian', ksize=15, int_c=3)

    # Vẽ kết quả bằng matplotlib
    fig=plt.figure(figsize=(16,9))
    ax1,ax2,ax3=fig.subplots(1,3)
    ax1.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    ax1.set_title("Ảnh gốc")

    ax2.imshow(cv2.cvtColor(anh_cat_nguong_mean, cv2.COLOR_BGR2RGB))
    ax2.set_title("Ảnh cắt ngưỡng thích nghi - mean")

    ax3.imshow(cv2.cvtColor(anh_cat_nguong_gauss, cv2.COLOR_BGR2RGB))
    ax3.set_title("Ảnh cắt ngưỡng thích nghi - gaussian")

    plt.savefig("adaptive_thresholding.pdf", bbox_inches='tight')
    plt.show()

'''OTSU'''

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import mahotas

def otsu_threshold(image):
    """
    NOTE: Hàm chọn ngưỡng T tối ưu bằng phương pháp Otsu.
    """
    T = mahotas.thresholding.otsu(image)
    print("Otsu's threshold: {}".format(T))
    thresh = image.copy()
    thresh[thresh > T] = 255
    thresh[thresh < 255] = 0
    thresh = cv2.bitwise_not(thresh)  # We then invert our threshold by using cv2.bitwise_not. This is equivalent to applying a cv2.THRESH_BINARY_INV thresholding type
    return thresh

if __name__ == '__main__':
    img = cv2.imread('image/coins.png', 0)
        
    blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Áp dụng Gaussian blurring với bán kính bằng 5 để loại bỏ một vài cạnh có tần số cao mà chúng ta không quan tâm

    anh_cat_nguong_otsu = otsu_threshold(blurred)
    # Vẽ kết quả bằng matplotlib
    fig=plt.figure(figsize=(16,9))
    ax1,ax2=fig.subplots(1,2)
    ax1.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    ax1.set_title("Ảnh gốc")

    ax2.imshow(cv2.cvtColor(anh_cat_nguong_mean, cv2.COLOR_BGR2RGB))
    ax2.set_title("Ảnh cắt ngưỡng toàn cục, tìm T bằng otsu")

    plt.savefig("otsu_thresholding.pdf", bbox_inches='tight')
    plt.show()


'''laplacian and sobel'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

def sobel_filter(image):
    """
    Lọc sắc nét Sobel.
    """
    # Compute gradients along the X and Y axis, respectively
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    # Take the absolute value of gradient images
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    # Combine our Sobel gradient images using bitwise OR
    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    return sobelCombined

def laplacian_filter(image):
    # Compute the Laplacian of the image
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap

if __name__=='__main__':
    img = cv2.imread('image/coins.png', 0)

    sobel_img = sobel_filter(img)
    lap_img = laplacian_filter(img)
    # vẽ kết quả bằng matplotlib
    # Display the results using matplotlib
    fig=plt.figure(figsize=(16,9))
    ax1,ax2,ax3=fig.subplots(1,3)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Ảnh gốc")

    ax2.imshow(cv2.cvtColor(sobel_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Phát hiện cạnh bằng đạo hàm Sobel")

    ax3.imshow(cv2.cvtColor(lap_img, cv2.COLOR_BGR2RGB))
    ax3.set_title("Phát hiện cạnh bằng đạo hàm Laplacian")

    plt.savefig("gradient_edges.pdf", bbox_inches='tight')
    plt.show()

'''Canny edge detection'''
import numpy as np
import matplotlib.pyplot as plt 
import cv2

def canny_edge_detection(img, threshold1,threshold2):
    '''
    note: phát hiện cạnh bằng canny_edge.
    '''
    image = cv2.GaussianBlur(img,(5,5),0)
    canny = cv2.Canny(image, threshold1,threshold2)
    return canny

if __name__=='__main__':
    img=cv2.imread('image/coins.png',0)
    canny_img= canny_edge_detection(img,30,150) 
    # vẽ kết quả bằng matplotlib 
    fig=plt.figure(figsize=(16,9))
    ax1,ax2=fig.subplots(1,2) 
    ax1.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    ax1.set_title("ảnh gốc") 

    ax2.imshow(cv2.cvtColor(canny_img,cv2.COLOR_BGR2RGB))
    ax2.set_title("Phát hiện cạnh bằng canny edge detection")
    plt.savefig("canny_edges.pdf",bbox_inches='tight')
    plt.show()




