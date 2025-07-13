import matplotlib.pyplot as plt
import cv2
import numpy as np

class lowpass_filter():
    '''Lớp xử lọc mịn ảnh'''
    def __init__(self,image) -> None:
        '''Khởi tạo đối tượng với ảnh đầu vào'''
        self.img=image
    def convolution2d(self,kernel):
        #padding??????
        m, n = self.img.shape
        img_new = np.zeros([m, n],dtype="uint8")
        for i in range(1, m-1):
            for j in range(1, n-1):
                temp= self.img[i-1, j-1]    *   kernel[0, 0]\
                    +  self.img[i-1, j]     *   kernel[0, 1]\
                    +  self.img[i-1, j + 1] *   kernel[0, 2]\
                    +  self.img[i, j-1]     *   kernel[1, 0]\
                    +  self.img[i, j]       *   kernel[1, 1]\
                    +  self.img[i, j + 1]   *   kernel[1, 2]\
                    +  self.img[i + 1, j-1] *   kernel[2, 0]\
                    +  self.img[i + 1, j]   *   kernel[2, 1]\
                    +  self.img[i + 1, j+1] *   kernel[2, 2]
                img_new[i, j]= temp
        img_new = img_new.astype(np.uint8)
        return img_new

    def MeanFilter3_3(self):
        '''Phương thức lọc trung bình'''
        MeanKernel33=np.array([[1/9, 1/9, 1/9], 
                                    [1/9, 1/9, 1/9],
                                    [1/9, 1/9, 1/9]], dtype="float")
        return self.convolution2d(MeanKernel33)
    def Median33(self):
        '''Lọc trung vi'''
        m, n = self.img.shape
        img_new = np.zeros([m, n])
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = [self.img[i - 1, j - 1],
                        self.img[i - 1, j],
                        self.img[i - 1, j + 1],
                        self.img[i, j - 1],
                        self.img[i, j],
                        self.img[i, j + 1],
                        self.img[i + 1, j - 1],
                        self.img[i + 1, j],
                        self.img[i + 1, j + 1]]
                temp = sorted(temp)
                img_new[i, j] = temp[4]
        return img_new
    
    

if __name__=='__main__':
    # Đọc và hiển thị ảnh gốc
    image = cv2.imread('image/test2.tif', 0)
    fig=plt.figure(figsize=(9, 9))
    plt.show()