import cv2
import numpy as np
from keras.models import load_model

model=load_model('./trainingDataTarget/Face_MobileV2modelv2.h5')

video=cv2.VideoCapture(0)

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

color_dict={0: (0,255,0), 1: (0,0,255)}
labels_dict={0: "With Mask", 1: "Without Mask"}

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        frame=cv2.flip(frame,1,1)
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray,
                                         scaleFactor=1.3,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE
                                        )

        for x,y,w,h in faces:
            x1,y1=x+w, y+h

            face_image = frame[y:y+h,x:x+w]
            resize_img  = cv2.resize(face_image,(224,224))
            normalized = resize_img/255.0
            reshape = np.reshape(normalized,(1,224,224,3))
            reshape = np.vstack([reshape])
            result = model.predict(reshape)
            
            label=np.argmax(result, axis=1)[0]

            label_text = labels_dict[label]
            confidence = str(round(result[0][1] * 100, 2)) + '%'

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 2)

            cv2.line(frame, (x,y), (x+30, y),(255,0,255), 6)
            cv2.line(frame, (x,y), (x, y+30),(255,0,255), 6)

            cv2.line(frame, (x1,y), (x1-30, y),(255,0,255), 6)
            cv2.line(frame, (x1,y), (x1, y+30),(255,0,255), 6)

            cv2.line(frame, (x,y1), (x+30, y1),(255,0,255), 6)
            cv2.line(frame, (x,y1), (x, y1-30),(255,0,255), 6)

            cv2.line(frame, (x1,y1), (x1-30, y1),(255,0,255), 6)
            cv2.line(frame, (x1,y1), (x1, y1-30),(255,0,255), 6)

            cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            cv2.putText(frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, confidence, (x, y+h+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()