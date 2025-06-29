from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Загружаем модель
model = load_model('new_best_model\VGG16.keras')


# Препроцессинг изображений
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Классы эмоций
classes = ['Удивление', 'Страх', 'Отвращение', 'Счастливый', 'Грустно', 'Злой', 'Нейтрально']


# Папки с данными
folders = ['1', '2', '3', '4', '5', '6', '7']
folder_to_class = {name: i for i, name in enumerate(folders)}

def plot_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f'Матрица предсказаний (%) ({dataset_name})')
    plt.colorbar()

    # Подписываем оси
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Заполняем значения внутри ячеек
    for i in range(len(folders)):
        for j in range(len(folders)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", 
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('Истинные классы')
    plt.xlabel('Предсказанные классы')
    plt.show()

# Функция анализа предсказаний (работает и для train, и для test)
def analyze_predictions(dataset_path, dataset_name):
    # Инициализируем матрицу ошибок
    cm = np.zeros((len(folders), len(folders)), dtype=int)
    len_folders = []
    for folder in folders:
        
        true_class = folder_to_class[folder]
        folder_path = os.path.join(dataset_path, folder)
        len_folder = os.listdir(path=folder_path)

        len_folders.append(len(len_folder))
        print('Класс ',folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = preprocess_image(img_path)

            pred = model.predict(img, verbose=0)
            predicted_class = np.argmax(pred)

            # Заполняем матрицу ошибок
            cm[true_class, predicted_class] += 1
        print('Класс распозонвание закончил ',folder)
        print(cm)

    # Выводим процент распознавания по классам
    print(f"\nТочность по классам ({dataset_name}):")
    for i in range(len(folders)):
        total = np.sum(cm[i])
        accuracy = cm[i, i] / total if total > 0 else 0
        print(f"{classes[i]}: {accuracy * 100:.2f}%")
    
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            cm[i][j] = (cm[i][j] / len_folders[i] ) * 100
    # Отображаем матрицу ошибок
    plot_confusion_matrix(cm, dataset_name)

#train папка с датасета 
analyze_predictions('train', 'Тестовая выборка')
