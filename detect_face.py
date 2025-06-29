import cv2
import numpy as np
import tensorflow as tf  

# загрузка модели
model = tf.keras.models.load_model("new_best_model//ResNet50.keras")


emotion_labels = classes = {
    0: 'Surprise',
    1: 'Fear',
    2: 'Disgust',
    3: 'Happy',
    4: 'Sad',
    5: 'Angry',
    6: 'Neutral'
}


cap = cv2.VideoCapture(0)

#  OpenCV для лица
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # вырезаем лици и подаем модели
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        #Обрабатываем
        face = rgb[y:y+h, x:x+w]                       
        face = cv2.resize(face, (224, 224))            
        face = face / 255.0                            
        face = np.expand_dims(face, axis=0)            

        predictions = model.predict(face, verbose=0)
        emotion_index = np.argmax(predictions)
        emotion_text = emotion_labels[emotion_index]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)



    cv2.imshow("Emotion Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
