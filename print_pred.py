from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Загрузка модели и подготовка данных
model = load_model('best_model\MyModel_tune.keras')
classes = ['Гнев', 'Страх', 'Счастливый', 'Нейтрально', 'Грустно', 'Удивление']
folders = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
path = 'example_images'

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img


files = os.listdir(path)
num_images = len(files)
rows = (num_images + 2) // 3  


fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
if rows == 1:
    axes = axes.reshape(1, -1)  
predict_dict = {}


for i, file in enumerate(files):
    img_path = os.path.join(path, file)
    img = preprocess_image(img_path)
    

    pred = model.predict(img, verbose=0)
    predicted_class = np.argmax(pred)
    print('Предсказания: ',pred)
    emotion = classes[predicted_class]
    predict_dict[file] = emotion
    
 
    row = i // 3
    col = i % 3
    

    img_display = cv2.imread(img_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    axes[row, col].imshow(img_display)
    axes[row, col].axis('off')

    axes[row, col].set_title(f"Ответ модели: {emotion}", fontsize=20)

# Скрываем пустые subplot, если количество изображений не кратно 3
for i in range(len(files), rows * 3):
    row = i // 3
    col = i % 3
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

print("Результаты распознавания:")
print(predict_dict)