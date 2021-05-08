import cv2
import numpy as np
from tensorflow import keras
from mtcnn.mtcnn import MTCNN
from PreProcessor import recenter


def build_model():
    model = keras.models.load_model("model_best.h5")
    return model

def detect_faces(data, face_detector):
    faces = face_detector.detectMultiScale(data, 1.05, 4, minSize=(124,124), flags=cv2.CASCADE_SCALE_IMAGE)
    ret = []
    for face in faces:
        x1, y1, width, height = face
        ret.append(recenter(x1, y1, width, height))

    return ret

def run(model, face_detector):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Our operations on the frame come here
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        list_of_faces = detect_faces(image, face_detector)
        for face in list_of_faces:
            xl, xr, yt, yb = face
            model_data = cv2.resize(image[yt:yb, xl:xr],(124, 124))
            model_data = cv2.cvtColor(model_data, cv2.IMREAD_COLOR)
            model_data = np.array(model_data, dtype="float")/255.0
            model_data = np.expand_dims(model_data, axis=0)
            model_data = model_data.reshape(1, 124, 124, 3)
            prediciton = model.predict(model_data)[0]
            print(prediciton)
            cv2.rectangle(image,(xl,yt),(xr,yb), (0,255*prediciton[0],255*prediciton[1]), 2)

        # Display the resulting frame
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = build_model()
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    run(model,face_detector)
    quit()