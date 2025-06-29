import cv2
from ultralytics import YOLO
import numpy as np
import tensorflow as tf  

emotion_classes = {
    0: 'Surprise',
    1: 'Fear',
    2: 'Disgust',
    3: 'Happy',
    4: 'Sad',
    5: 'Angry',
    6: 'Neutral'
}

# Загрузка модели YOLO для обнаружения лица
model_detect = YOLO("best_model\\best_new.pt")
# Загрузка модели CNN для классификации эмоции
model_classification = tf.keras.models.load_model("best_model\\MyModel_tune.keras")

# Открытие веб-камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Предсказание
    results = model_detect(frame)

    # Отображение результатов
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Преобразование координат
            conf = box.conf[0]  # Доверие
            
            x1 += 10
            x2 -= 10
            y1 += 10
            y2 -= 10

            face = frame[y1:y2, x1:x2]  # получаем лицо
            
            #Нормализация данных
            face = cv2.resize(face, (48, 48)) 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = face / 255.0  
            face = np.expand_dims(face, axis=0) 
            face = np.expand_dims(face, axis=-1) 

            # Ответ модели
            predictions = model_classification.predict(face, verbose=0)
            emotion_index = np.argmax(predictions)  
            emotion_text = emotion_classes[emotion_index] 



            # Рисуем рамку
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Отображение процента обнаружения лица
            cv2.putText(frame, f"Face: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Отображение эмоции лица
            cv2.putText(frame, f"Emotion:{emotion_text}", (x1, y1 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Recognize Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
