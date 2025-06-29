import os
from PIL import Image

# Путь к папке с оригинальными изображениями
source_dir = "train_test/"
# Путь к папке для сохранения изображений в новом формате
output_dir = "min_dataset"

min_files = 800

# Поддерживаемые форматы
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
dir_list = ['angry', 'fear','happy', 'neutral', 'sad', 'surprise']

def quantity_files_in_folder(folders):
    import os.path
    path = source_dir
    quantity = []
    for folder in folders:
        num_files = len([f for f in os.listdir(path+folder)
                        if os.path.isfile(os.path.join(path+folder, f))])
        # print(f'Количество файлов в папке {folder}: {num_files}')
        quantity.append(num_files)
    return quantity

quantity_folder = quantity_files_in_folder(dir_list)
print(quantity_folder)

def delete_random_files(folders, quantities):
    from random import sample
    dataset_path = source_dir
    for i in range(len(folders)):
        if quantities[i] > min_files:
            folder_path = os.path.join(dataset_path, folders[i])
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]  # Полные пути
            files_to_delete = sample(files, quantities[i] - min_files)  # Список файлов для удаления

            for file in files_to_delete:
                os.remove(file)
            print(f"Из папки {folders[i]} удалено {quantities[i] - min_files} файлов.")


delete_random_files(dir_list, quantity_folder)


import os
from PIL import Image

def convert_png_to_jpg(root_dir):
    """
    Обходит все папки в root_dir, находит файлы .png и конвертирует их в .jpg.
    После успешной конвертации удаляет исходные .png файлы.
    """
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.png'):
                png_path = os.path.join(folder, file)
                jpg_path = os.path.join(folder, os.path.splitext(file)[0] + ".jpg")

                try:
                    # Открываем PNG и конвертируем в RGB (чтобы убрать альфа-канал)
                    img = Image.open(png_path).convert("RGB")
                    img.save(jpg_path, "JPEG", quality=95)

                    # Удаляем оригинальный PNG после успешной конвертации
                    os.remove(png_path)

                    print(f"✅ {png_path} -> {jpg_path}")
                except Exception as e:
                    print(f"❌ Ошибка при обработке {png_path}: {e}")

# Пример использования
dataset_path = source_dir  # Укажите путь к вашему датасету
convert_png_to_jpg(dataset_path)
