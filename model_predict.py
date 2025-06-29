from tensorflow.keras.models import load_model

model = load_model('MyModel_73_58.keras')  # Укажи путь к сохранённому файлу

import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))  # Изменяем размер
    img = img / 255.0  # Нормализация пикселей
    img = np.expand_dims(img, axis=0)  # Добавляем размерность batch
    return img

classes = {
    0: 'Гнев',
    1: 'Страх',
    2: 'Счастливый',
    3: 'Нейтрально',
    4: 'Грустно',
    5: 'Удивление'
}

import os

def predict_test_images():
    folders = ['angry','fear', 'happy', 'neutral', 'sad', 'surprise']  # Папка с тестовыми изображениями
    path = 'test/'

    total_class = 0
    total_predict = 0
    for i in range(len(folders)):
        count_class = 0
        total_for_class = 0
        for img_name in os.listdir(path + folders[i]):
            img_path = os.path.join(path + folders[i], img_name)
            img = preprocess_image(img_path)
            pred = model.predict(img, verbose=0)
            if classes[np.argmax(pred)] == classes[i]:
                count_class += 1
            total_for_class += 1

            # if total_for_class == 100:
            #     break
        total_predict += total_for_class
        total_class += count_class

        print(f"Класс: {folders[i]}")
        print(f'Всего {total_for_class}')
        print(f'Распознано верно {count_class}')
    print(f'Процент предсказаний: {(total_class / total_predict) * 100}%')

def pred_real_images():
    for img_name in os.listdir('image'):
        img_path = os.path.join('image', img_name)
        img = preprocess_image(img_path)
        pred = model.predict(img, verbose=0)


        print(f'{img_name}: {classes[np.argmax(pred)]} Массив предсказаний: {pred}')


predict_test_images()

pred_real_images()
