import cv2


capture = cv2.VideoCapture(0)
while True:
    ret,frame = capture.read()
    # gray = cv2.cvtColor(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
       gray,
       scaleFactor = 1.15,
       minNeighbors = 5,
       minSize = (5,5)
    )
    print ("发现{0}个人脸!".format(len(faces)))
    faces_list = []
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+w),(0,255,0),2)
        faces_list.append([x,y,x+w,y+w])
    for i in faces_list:
        res_face = frame[i[1]:i[3],i[0]:i[2]]

        image = cv2.GaussianBlur(res_face, (5,5), 0, 0)
        # 滤波
        image = cv2.bilateralFilter(image,30,60,15)
        # 图像增强
        image2 = cv2.GaussianBlur(image,(0,0),9)
        res = cv2.addWeighted(image,1.5,image2,-0.5,0)
        # value1 = 3
        # value2 = 1
        # dx = value1 * 5
        # fc = value1 * 12.5
        # p = 0.1
        # image1 = cv2.bilateralFilter(res_face, dx, fc, fc)
        # image2 = cv2.subtract(image1, res_face)
        # image3 = cv2.add(image2, (128, 128, 128, 128))
        #
        # # 高斯模糊
        # image4 = cv2.GaussianBlur(image2, (2 * value2 - 1, 2 * value2 - 1), 0, 0)
        # image4 *= 2
        # image4 -= 255
        # image5 = cv2.add(res_face, image4)
        # image6 = cv2.addWeighted(res_face, p, image5, 1 - p, 0.0)
        # res = cv2.add(image6, (10, 10, 10, 10))
        frame[i[1]:i[3],i[0]:i[2]] = res

    # res_face = frame.copy()
    # image = cv2.GaussianBlur(res_face, (5,5), 0, 0)
    # # 滤波
    # image = cv2.bilateralFilter(image,30,60,15)
    # # 图像增强
    # image2 = cv2.GaussianBlur(image,(0,0),9)
    # res = cv2.addWeighted(image,1.5,image2,-0.5,0)
    # frame = res

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break