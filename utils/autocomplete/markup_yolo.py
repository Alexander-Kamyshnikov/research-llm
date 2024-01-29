import shutil
import numpy as np
import cv2
import random
from pdf2image import convert_from_path
from pathlib import Path
from ultralytics import YOLO

from utils.autocomplete.yolo_utils import get_bboxes
from utils.extraction.prioritet_extractor import PrioritetExtractor
import matplotlib.pyplot as plt

extractor = PrioritetExtractor()


def is_true_pdf(data: dict) -> bool:
    return all([data['kind'] == 'Письмо', data['delivery_type'] != 'МЭДО'])


def get_true_pdf_with_order(path: Path, data: dict) -> str:
    # функция возвращает путь к существующему файду пдф с максимальным ордером
    order_files = {}
    for file in data['files']:
        order_files[file['order']] = file
    while order_files:
        value = order_files.pop(max(order_files.keys()))
        if value['extension'] == '.pdf':
            if (exists_path := path / (value['id'] + '.pdf')).exists():
                return exists_path


def get_true_path_to_pdf(path: Path, with_medo=False) -> Path:
    extracted_data = extractor.process_folder(path)
    # функция возвращает путь к существующему файлу пдф с максимальным значением order,
    # если kind == "Письмо" и delivery_type != "МЭДО"
    if with_medo:
        if extracted_data['kind'] == 'Письмо' and (true_path := get_true_pdf_with_order(path, extracted_data)):
            return Path(true_path)
    else:        
        if (is_true_pdf(extracted_data)) and (true_path := get_true_pdf_with_order(path, extracted_data)):
            return Path(true_path)


def make_img_dir(path_to_img: Path, counter: int, doc_count_in_dir: int):
    if counter%doc_count_in_dir == 0:
        img_dir = str(counter) + '_' + str(counter + doc_count_in_dir)
        path_to_img_dir = path_to_img / img_dir
        Path.mkdir(path_to_img_dir, exist_ok=True)


def pdf_to_img(path_to_pdf_files: Path, path_to_img: Path, format_img: str = '.jpg', doc_count_in_dir=50, last_page=1):
    
    count_all_paths = 0
    true_counter = 0
    
    for path in path_to_pdf_files.iterdir():
        
        count_all_paths += 1
        doc = get_true_path_to_pdf(path)
    
        if doc:
            if true_counter%doc_count_in_dir == 0:
                img_dir = str(true_counter) + '_' + str(true_counter + doc_count_in_dir)
                path_to_img_dir = path_to_img / img_dir
                Path.mkdir(path_to_img_dir, exist_ok=True)

            doc_name = doc.parent.name
            #  конвертация изображения
            images = convert_from_path(doc, last_page=last_page, fmt=format_img)
            if last_page == 1:
                img = np.array(np.rot90(images[0], 0))
                # путь сохранения иображения
                # имя изображения = doc_id
                path_doc = path_to_img_dir/doc_name
                path_doc_ext = str(path_doc.with_suffix(format_img))
                # сохранение изображения
                cv2.imwrite(path_doc_ext, img)
                print(f'true_paths: {true_counter}  {path_doc_ext}', end='\r')
                true_counter += 1 
            else:
                for num, im in enumerate(images):
                    img = np.array(np.rot90(im, 0))
                    # путь сохранения иображения
                    # имя изображения = doc_id + num page
                    name = str(doc_name) + '_' + str(num)
                    path_doc = path_to_img_dir/name
                    path_doc_ext = str(path_doc.with_suffix(format_img))
                    cv2.imwrite(path_doc_ext, img)
                    print(f'true_paths: {true_counter}  {path_doc_ext}', end='\r')
                    true_counter += 1 
                    
                    if true_counter%doc_count_in_dir == 0:
                        img_dir = str(true_counter) + '_' + str(true_counter + doc_count_in_dir)
                        path_to_img_dir = path_to_img / img_dir
                        Path.mkdir(path_to_img_dir, exist_ok=True)                 
          
        if true_counter > 1000:
            print(f'всего обработано {count_all_paths} документов')
            break


def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def make_train_test_dir(path_from: Path, path_to: Path, test_size=0.2):
    """create directories with train and test folders with  images and labels folders"""
    
    if Path(path_to / 'train').is_dir():
        y = input('created directories  train and  test  already exists, delete existing directories? Y/N ')
        if y == 'Y':
            rm_tree(path_to / 'train')
            rm_tree(path_to / 'test')
        elif y != 'N':
            print('invalid input format, input Y or N')
            return
    
    Path.mkdir(path_to / 'train', exist_ok=True)
    Path.mkdir(path_to / 'test', exist_ok=True)
    Path.mkdir(path_to / 'train' / 'images', exist_ok=True)
    Path.mkdir(path_to / 'test' / 'images', exist_ok=True)
    Path.mkdir(path_to / 'train' / 'labels', exist_ok=True)
    Path.mkdir(path_to / 'test' / 'labels', exist_ok=True)
    images = list(Path(path_from / 'images').iterdir())
    labels = list(Path(path_from / 'labels').iterdir())
    labels.sort()
    images.sort()
    
    c = int(len(images) * test_size)
    while c:
        indx = random.randrange(len(images))
        shutil.copy(images.pop(indx), path_to / 'test' / 'images')
        shutil.copy(labels.pop(indx), path_to / 'test' / 'labels')
        c -= 1
    for path in images:
        shutil.copy(path, path_to / 'train' / 'images')
    for path in labels:
        shutil.copy(path, path_to / 'train' / 'labels')
        
        
def check_dates_numbers(labels: list, CLASSES):
    date_number_index = [CLASSES.index('predict_date'), CLASSES.index('predict_number')]
    # наличие классов date или number в предсказанных классах на одной странице
    for i in labels:
        if i[0] in date_number_index:
            return i
    return False


def get_dates_and_numbers(paths: list, path_save_date: Path,
                          path_save_numbers: Path,
                          model: YOLO, CLASSES: list,
                          limit = 500, disp=True):
    
    date_index = CLASSES.index('predict_date')
    number_index = CLASSES.index('predict_number')
    
    Path.mkdir(path_save_date, exist_ok=True)
    Path.mkdir(path_save_numbers, exist_ok=True)
    # функция ищет в предсказанных сущностях даты и номера и возвращает изображения таких boundig boxes
    # заложен функционал позволяющий разделять пути сохранения дат и номеров
    img = []
    count_list, date_list, number_list = [], [], []
    counter = 0  # счётчик всех документов
    count_dates = 0  # счётчик изображений дат
    count_numbers = 0  # счётчик изображений номеров

    for path in paths:
        counter += 1
        print(f'doc in process: {counter} found dates: {count_dates} found_numbers: {count_numbers}', end='\r')
        doc_id = path.parents[0].name
        doc_id = doc_id + '.jpeg'
        try:
            images = convert_from_path(path, last_page=1, fmt='.jpeg')
        except Exception:
            continue
        first_page = images[0]
        predict = model(first_page, verbose=False)
        labels = get_bboxes(predict)
        if (coords:=check_dates_numbers(labels, CLASSES)):           
            cropped = first_page.crop(coords[1:-1])
            if disp:
                display(cropped)
            if coords[0] == date_index and count_dates < limit:
                count_dates += 1
                cropped.save(path_save_date / doc_id)
            elif count_numbers < limit:
                count_numbers += 1
                cropped.save(path_save_numbers / doc_id)
        count_list.append(counter)
        date_list.append(count_dates)
        number_list.append(count_numbers)
                
        if (count_dates >= limit) and (count_numbers >= limit):
            return count_list, date_list, number_list
        
    return count_list, date_list, number_list


def plot_count_dates_numbers(count_list: list, date_list: list, number_list: list, xticks=100, y_ticks=20):
    plt.figure(figsize=(15,5))
    plt.plot(count_list, date_list,  label = "dates")
    plt.plot(count_list, number_list, label = "numbers")
    plt.xticks(list(range(0, len(count_list),xticks)))
    plt.yticks(list(range(0, max(date_list),y_ticks)))
    plt.grid(True)
    plt.xlabel('общее количество документов')
    plt.ylabel('количество документов содержащих искомый класс')
    # plt.set_xlabels()
    plt.legend()
    plt.show()
    
    
def split_images_to_folders(path_from: Path, path_to: Path, doc_count_in_dir = 50):
    
    Path.mkdir(path_to, exist_ok=True)
    counter = 0
    for path in path_from.iterdir():
        if counter%doc_count_in_dir == 0:
            Path.mkdir(path_to/ str(counter), exist_ok=True)
            path_to_img_dir = path_to/ str(counter)
        shutil.copy(path, path_to_img_dir)            
        counter += 1
    # удаление общей дирректории для изображений дат и номеров
    rm_tree(path_from)
