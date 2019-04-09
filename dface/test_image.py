import cv2
from dface.core.detect import create_mtcnn_net, MtcnnDetector
from dface.core import vision


if __name__ == '__main__':

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
    while True:
        ret, frame = capture.read()
        bboxs, landmarks = mtcnn_detector.detect_face(frame)
        # write bbox
        print len(bboxs)
        for bbox in bboxs:
            bbox = [int(x) for x in bbox[:4]]
            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
        for landmark in landmarks:
            for i in range(0,len(landmark),2):
                cv2.circle(frame,(int(landmark[i]),int(landmark[i+1])),1,(0,0,255),4)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break