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
    cv2.waitKey(0)
    cv2.imwrite('test_1.jpg', image)
    return dst

image = cv2.imread('./skin_test.jpg')
image = white_face(image)
# 高斯模糊
cv2.GaussianBlur(image, (9, 9), 0, 0)
# cv2.imshow('test.jpg',image)
bilateralFilterVal = 30



