import os

path_del = "C:\\Projects\\MO\\new_dataset\\fear_tune"
arr_fiels_del = os.listdir(path_del) 

path_file_check = "C:\\Projects\\MO\\emotion\\train\\fear"
arr_file_check =os.listdir(path_file_check)
print('Файлов всего: ', len(arr_fiels_del))
print('Оставшийся: ', len(arr_file_check))

for file in arr_fiels_del:
    if file in arr_file_check:
        os.remove(os.path.join(path_del,file))

