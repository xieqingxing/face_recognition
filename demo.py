# --coding:utf-8 --
from dface.core.detect import create_mtcnn_net, MtcnnDetector
import dlib
import cv2
import numpy as np
import os

def get_feature(image,db_path,facerec,sp):

    rec = dlib.rectangle(0,0,image.shape[1],image.shape[0])
    shape = sp(image, rec)  # 获取landmark
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    faceArray = np.array(face_descriptor).reshape((1, 128))
    db_names = os.listdir(db_path)
    for j in db_names:
        db_name = os.path.join(db_path,j)
        for i in os.listdir(db_name):
            db_feature = np.load(os.path.join(db_name,i))
            dist1 = np.sqrt(np.sum(np.square(np.subtract(db_feature, faceArray))))
            if dist1 < 0.55:
                return db_name.split('/')[-1]
    return 'others'

if __name__ == '__main__':

    facerec = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
    sp = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')
    # refer to your local model path 
    p_model = "./model_store/pnet_epoch.pt"
    r_model = "./model_store/rnet_epoch.pt"
    o_model = "./model_store/onet_epoch.pt"

    #use cpu version set use_cuda=False, if you want to use gpu version set use_cuda=True
    pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model, r_model_path=r_model, o_model_path=o_model, use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    # img = cv2.imread("./test.jpg")
    # b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])

    capture = cv2.VideoCapture(0)
    j = 0
    while True:
        ret, frame = capture.read()
        bboxs,landmarks = mtcnn_detector.detect_face(frame)
        for bbox in bboxs:
            bbox = [int(x) for x in bbox[:4]]
            # cv2.imwrite('./db/black_box/'+str(j)+'.jpg',frame[bbox[1]:bbox[3],bbox[0]:bbox[2]])
            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
            result = get_feature(frame[bbox[1]:bbox[3],bbox[0]:bbox[2]],'./db/',facerec,sp)
            cv2.putText(frame, result, (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # for landmark in landmarks:
        #     for i in range(0,len(landmark),2):
        #         cv2.circle(frame,(int(landmark[i]),int(landmark[i+1])),1,(0,0,255),4)
        # j += 1
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break