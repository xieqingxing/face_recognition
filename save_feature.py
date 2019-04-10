#--coding:utf-8--
#录入新的人脸

# --coding:utf-8 --
from dface.core.detect import create_mtcnn_net, MtcnnDetector
import dlib
import cv2
import numpy as np
import os
import commands
import sys


def save_feature(image,db_path,facerec,sp,j,len_feature):
    rec = dlib.rectangle(0,0,image.shape[1],image.shape[0])
    shape = sp(image, rec)  # 获取landmark
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    faceArray = np.array(face_descriptor).reshape((1, 128))
    np.save(db_path+'/'+str(len_feature+j)+'.npy',faceArray)


if __name__ == '__main__':

    db_path = './db/'
    facerec = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
    sp = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')
    # refer to your local model path
    p_model = "./model_store/pnet_epoch.pt"
    r_model = "./model_store/rnet_epoch.pt"
    o_model = "./model_store/onet_epoch.pt"

    # input name
    name = raw_input('please input your name:')
    if name == '' or name == ' ':
        print 'input error'
        exit(0)
    name = name.split(' ')[0]
    feature_path = os.path.join(db_path,name)
    if not os.path.exists(feature_path):
        output = commands.getoutput('mkdir {}'.format(feature_path))
    len_feature = len(os.listdir(feature_path))
    if len_feature > 100:
        len_feature = 80
    # exit(0)

    # model infer
    pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model, r_model_path=r_model, o_model_path=o_model, use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    # get video camera
    capture = cv2.VideoCapture(0)
    j = 1
    while j < 21:
        ret, frame = capture.read()
        bboxs,landmarks = mtcnn_detector.detect_face(frame)
        # write bbox
        for bbox in bboxs:
            bbox = [int(x) for x in bbox[:4]]
            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
            cv2.imwrite('./db/black_box/'+str(j)+'.jpg',frame[bbox[1]:bbox[3],bbox[0]:bbox[2]])
            save_feature(frame[bbox[1]:bbox[3],bbox[0]:bbox[2]],feature_path,facerec,sp,j,len_feature)
        j += 1
        # write landmark
        for landmark in landmarks:
            for i in range(0,len(landmark),2):
                cv2.circle(frame,(int(landmark[i]),int(landmark[i+1])),1,(0,0,255),4)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    print 'get feature over,you can use this system'