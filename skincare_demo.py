import cv2

def white_face(image):
    rows,cols = image.shape[:2]
    dst = image.copy()
    a = 0.7
    b = 68
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = image[i, j][c] * a + b
                if color > 255:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 255
                elif color < 0:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 0
    return dst

start = cv2.getTickCount()
image = cv2.imread('./skin_test.jpg')
image = white_face(image)
# 高斯模糊
image = cv2.GaussianBlur(image, (5,5), 0, 0)
# 滤波
image = cv2.bilateralFilter(image,30,60,15)
# 图像增强
image2 = cv2.GaussianBlur(image,(0,0),9)
res = cv2.addWeighted(image,1.5,image2,-0.5,0)
end = cv2.getTickCount()
during = (end - start) / cv2.getTickFrequency()
print(during)
cv2.imshow('test.jpg',res)
cv2.imwrite('test_4.jpg',res)
cv2.waitKey(0)



