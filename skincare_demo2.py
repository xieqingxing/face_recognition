# --coding:utf-8 --
import cv2


start = cv2.getTickCount()
image = cv2.imread('./skin_test.jpg')
value1 = 3
value2 = 1
dx = value1 * 5
fc = value1 * 12.5
p = 0.1
image1 = cv2.bilateralFilter(image,dx, fc, fc)
image2 = cv2.subtract(image1,image)
image3 = cv2.add(image2,(128,128,128,128))

# 高斯模糊
image4 = cv2.GaussianBlur(image3,(2 * value2 - 1, 2 * value2 - 1),0,0)
image4 *= 2
image4 -= 255
image5 = cv2.add(image,image4)
image6 = cv2.addWeighted(image, p, image5, 1 - p, 0.0)
image7 = cv2.add(image6,(10,10,10,10))
end = cv2.getTickCount()
during = (end - start) / cv2.getTickFrequency()
print(during)
cv2.imwrite('test2.jpg',image6)