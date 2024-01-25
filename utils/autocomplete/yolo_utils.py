from ultralytics import YOLO
import torch
from pathlib import Path
import shutil
from IPython.display import display
from pytesseract import pytesseract
from transformers import pipeline
from pdf2image import convert_from_path, convert_from_bytes
import pandas as pd
import csv
import numpy as np
import re
import typing as tp
from typing import IO, Union, Optional
import dateutil
import datetime
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from pyjarowinkler import distance
from rapidfuzz import fuzz
from utils.extraction.prioritet_extractor import PrioritetExtractor
from utils.extraction.file_handlers import read_json

from tqdm import tqdm
tqdm.pandas()
 
    
def get_bboxes(predict) -> list:
    boxes = [result.boxes for result in predict]
    labels = []
    for box in boxes[0]:
        boxes = torch.cat((box.cls, box.xyxy[0], box.conf)).tolist()
        boxes[0] = int(boxes[0])
        labels.append(boxes)
        
    return labels

    
def get_text_by_coords(CLASSES, labels, image, model_handwrite, num_page=0):
    classes = {}
    predicts = {'predict_text':['', 0]}
    text_index = CLASSES.index('predict_text')
    join_text = ''
    for bbox in labels:
        if (predict_class := CLASSES[bbox[0]]) == 'predict_text':
            cropped = image.crop(bbox[1:-1])

            proba = bbox[-1]

            # для классов даты и номеров сначала определяется - рукописныый/машинописный. 
            # в случае рукописного текста, такому тексту присваивается значение "handwrite"
            date_number_index = [CLASSES.index('predict_date'), CLASSES.index('predict_number')]
            if bbox[0] in(date_number_index):
                if model_handwrite(cropped)[0].boxes.cls[0].item() == 0.0:
                    text = 'handwrite'
                else:
                    text = pytesseract.image_to_string(cropped, lang="rus")

            else: # определение любых символов для остальных классов
                    text = pytesseract.image_to_string(cropped, lang="rus")
                # если в документе определилось два экземпляра одного класса, то выбираем с наибольшей вероятностью
                # для предсказанного текста берём все экземпляры
            if predict_class == 'predict_text':
                predict_text = re.sub('\n', ' ',text)
            elif predict_class not in classes:
                classes[predict_class] = proba
                predict_text = re.sub('\n', ' ',text)    
            else:
                if proba > classes[predict_class]:
                    classes[predict_class] = bbox[-1]
                    predict_text = re.sub('\n', '',text)
                else:
                    continue
            # добавление найденных классов на первой странице
            # для текста объединяются все найденные экземпляры
            if num_page == 0:
                if predict_class == 'predict_text':
                    predicts[predict_class] = [predicts[predict_class][0] + ' ' + predict_text.strip(), proba]
                else:   
                    predicts[predict_class] = [predict_text.strip(), proba]
    #         добавление найденного сигнера на странице отличной от первой страницы
    #         остальные найденные классы игнорируются
            else:
                if predict_class == 'predict_signer':
                    predicts[predict_class] = [predict_text.strip(), proba]
                elif predict_class == 'predict_text':
                    predicts[predict_class] = [predicts[predict_class][0] + ' ' + predict_text.strip(), proba]
    return predicts


def check_signer(labels: list, CLASSES: list):
    signer_index = CLASSES.index('predict_signer')
    # наличие класса signer в предсказанных классах на одной странице
    for i in labels:
        if i[0] == signer_index:
            return True
        
def set_none_for_noexist_key(CLASSES, labels):
    for key in CLASSES:
        if key not in labels.keys():
            labels[key] = None

        
def show_detect_labels(df_labels: pd.DataFrame, DIR_TEST_IMAGES: str,  CLASSES: list ):
    
    # отображение найденых bboxes на изображении текста, 
    # также отображение; предсказанного класса ,вероятность класса, 
    # doc_id документа, распознанный текст, координаты bbox
    doc_id_list = df_labels.doc_id.to_list()
    image_list = []
    for doc_id in doc_id_list:
        path = Path.cwd() / DIR_TEST_IMAGES / doc_id
        original = Image.open(str(Path.cwd() / DIR_TEST_IMAGES / doc_id) + '.jpeg')
        for bbox in df_labels[df_labels['doc_id'] == doc_id]['values'].iloc[0]:
            cropped = original.crop(bbox[1:-1])
            if bbox[0] == 1:
    #             text = pytesseract.image_to_string(cropped, lang='eng', config='digits')
                text = pytesseract.image_to_string(cropped, lang='eng', config='outputbase digits')
            else:
                text = pytesseract.image_to_string(cropped, lang="rus")


            display(cropped)
            print('class: {},   confidence: {},  doc_id: {}, text: {}'.format(CLASSES[int(bbox[0])], bbox[-1], doc_id, text))
            print('coordinates: {}\n'.format(bbox[1:-1]))
            print ('-----------------------------------------------------------------------------------------------------')


def get_yolo_predicts(pdf_file: Union[Path, IO[bytes]], 
                      model_predict_classes: YOLO,
                      model_handwrite: YOLO,
                      CLASSES: list) -> list:
    
    if isinstance(pdf_file, Path):
        images = convert_from_path(pdf_file, fmt='.jpeg', last_page=10)
    elif isinstance(pdf_file, bytes):
        images = convert_from_bytes(pdf_file, fmt=".jpg", last_page=10)
    else:
        raise TypeError('unknown format of file')
        
    predicts_on_pages = []
    
    for num, page in enumerate(images):
        predict = model_predict_classes(page)
        labels = get_bboxes(predict)
        # если классов на странице не найдено, переход к следующей странице
        if not labels or len(labels) == 0:
            continue
        # определение текста на изображении по координатам
        predicts_on_pages.append(get_text_by_coords(CLASSES, labels, page, model_handwrite, num))
        # если найден сигнер (самый последний в документе класс) поиск прекращается
        if check_signer(labels, CLASSES):
            break
    return predicts_on_pages
            
def update_predicts(yolo_predicts: list, CLASSES) -> dict:
    predicts = {}
    # если текст найден более чем на одной странице - объеденим этот текст. proba - среднее арифметическое для найденных текстов.
    if len(yolo_predicts) > 1:
        text = ''
        proba = 0
        c =0
        for i in yolo_predicts:
            if i['predict_text']:
                c+=1
                text = text + ' ' + i['predict_text'][0]
                proba += i['predict_text'][1]
        if c:
            proba = proba/c
        else:
            proba = 0
        yolo_predicts[len(yolo_predicts)-1]['predict_text'] = [text, proba]
        [yolo_predicts[0].update(yolo_predicts[i+1]) for i in range(len(yolo_predicts)-1)]
        yolo_predicts = yolo_predicts[0]
    # если классы найдены только на первой странице, вернём словарь первой страницы
    elif len(yolo_predicts) == 1:
        yolo_predicts = yolo_predicts[0]
    else:
        yolo_predicts = {}
    # для ненайденых классов определим значение None
    set_none_for_noexist_key(CLASSES, yolo_predicts)
    return yolo_predicts         
            

def set_proba_for_classes(predicts: dict, CLASSES) -> dict:
    predicts_with_proba = {}
    for cl in CLASSES:
        p = cl + '_proba'
        if predicts[cl]:
            predicts_with_proba[cl] = predicts[cl][0]
            predicts_with_proba[p] = predicts[cl][1]
        else:
            predicts_with_proba[cl] = None
            predicts_with_proba[p] = 0.0
    return predicts_with_proba

def set_sender_by_comparison(predicts: dict, comparison: dict) -> dict:
    predicts['predict_sender'] = check_comparison(predicts['predict_sender'], comparison)
    return predicts

def get_sender_id(sender: str, partners: pd.DataFrame):
    
    # если сендер не определён ,возвращаются пустые спискм
    if (sender is None) or (sender == 'не определён'):
        return []
    
    # поиск всех соответствующих сендеров - полное совпадение
    n = partners[partners.partner_name == sender]
    m = partners[partners.partner_full_name == sender]
    df_sender = pd.concat([n,m])
    
    # если по полному совпадению сендер не найден, ищем по частичному совпадению
    if df_sender.empty:
        df_sender = partners.copy()
        df_sender['sim_name'] = df_sender.apply(lambda row: check_sender(row.partner_name, 
                                                                    sender, 
                                                                    return_value= True ), axis=1)
        df_sender['sim_fullname'] = df_sender.apply(lambda row: check_sender(row.partner_full_name, 
                                                                    sender, 
                                                                    return_value= True ), axis=1)

        df_sender['sim'] = df_sender.apply(lambda row: max([row.sim_name, row.sim_fullname]), axis=1)
        df_sender = df_sender[df_sender.sim >= 90]

        if not df_sender.empty:
            df_sender = df_sender[df_sender['sim'] == df_sender['sim'].max()]

    df_sender = df_sender.drop_duplicates(subset='partner_id')
    if df_sender.empty:
        return []
    if (p:= df_sender.partner_id.to_list()):
        return p
    else:
        return []

def clear_sender(sender: str) -> str:
    if sender is None or type(sender) == float:
        sender = 'не определён'
    sender = re.sub(r'[^\w\s\.]', ' ', sender.lower()).strip()
    if sender == '':
        sender = 'не определён'
    return sender    
    
def clear_signer(signer: str) -> str:
    if signer is None or type(signer) == float or signer == '':
        signer = 'не определён'
    else:
        signer = format_fio(signer)
        signer = format_fullname(signer)
    return signer

def get_signer_id(senders, fio, partners_employees):

    if not senders:
        return []    
    # поиск соответствующего сигнера во всех найденных организациях    
    if fio[0] is None:
        return []
    
    df_signer = partners_employees[partners_employees.partner_id.isin(senders)]
    # поиск сигнеров по фамилии
    df_signer_f = df_signer[df_signer.last_name == fio[0]]
    if df_signer_f.empty:
        return []
    elif len(df_signer_f) == 1:
        return df_signer_f.employee_id.to_list()
    else:
        # поиск сигнеров однофамильцев по имени
        if fio[1] is None:
            return list(set(df_signer_f.employee_id.to_list()))
        df_signer_fi = df_signer_f[df_signer_f.first_name == fio[1]]
        if df_signer_fi.empty:
            return list(set(df_signer_f.employee_id.to_list()))
        elif len(df_signer_fi) == 1:
            return df_signer_fi.employee_id.to_list()
        else:
            # поиск тёзок ФИ по их отчеству:
            if fio[1] is None:
                return list(set(df_signer_fi.employee_id.to_list()))
            df_signer_fio = df_signer_fi[df_signer_fi.middle_name == fio[2]]
            if df_signer_fio.empty:
                return list(set(df_signer_fi.employee_id.to_list()))
            else:
                return list(set(df_signer_fio.employee_id.to_list()))  
            
def get_sender(predicts: dict, comparison: dict, partners: pd.DataFrame) -> dict:
    
    predicts['predict_sender'] = clear_sender(predicts['predict_sender'])
    predicts = set_sender_by_comparison(predicts, comparison)
    predicts['predict_sender_id'] = get_sender_id(predicts['predict_sender'], partners)
    
    return predicts

def get_signer(predicts: dict, partners_employees: pd.DataFrame) -> dict:
    
    predicts['predict_signer'] = clear_signer(predicts['predict_signer'])
    fio = split_fio(predicts['predict_signer'])
    predicts['predict_signer_id'] = get_signer_id(predicts['predict_sender_id'], fio, partners_employees)
    
    return predicts
            
            
def month_translator(month: str) -> str:
    months = {'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8, 'авуста': 8,
              'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12, 'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'июн': 6,
              'июл': 7, 'авг': 8, 'ав': 8, 'сен': 9, 'окт': 10, 'ноя': 11, 'дек': 12}
    if month in months.keys():
        return months[month], month
    
def replace_str_month_to_digit(row: str):
    true_month = []
    month = list(filter(None, re.findall(r'[А-Яа-я]*', row)))
    for r in month:
        true_month.append(month_translator(r))
    true_month = list(filter(None, true_month))[0]
    digit_month = true_month[0]
    alpha_month = true_month[1]
    d = re.sub(alpha_month, digit_month, str(row))
    return '.'.join(list(filter(None, re.findall(r'[\d]*', d))))




def get_true_date(row:str) -> Optional[datetime.date]:

    if row == 'handwrite':
        return row
    elif row is None:
        return 'не определён'
    trash = '#№()@$&?!=+_*г'
    for symb in trash:
        row = row.replace(symb, '')
    row = row.strip()
    
    try:
        date = dateutil.parser.parse(row, dayfirst=True)
        return date.date()
    except (ValueError, TypeError, OverflowError):
        try:
            d = replace_str_month_to_digit(row)
            return dateutil.parser.parse(d, dayfirst=True).date()
        except:
            return 'не определён'

def get_date(predicts: dict) -> dict:
    predicts['predict_date'] = get_true_date(predicts['predict_date'])
    return predicts

def get_numbers(predicts: dict) -> dict:
    if predicts['predict_number']:
        predicts['predict_number'] = re.sub(r'[№_»]', '', predicts['predict_number'])
        predicts['predict_number'] = predicts['predict_number'].strip()
        predicts['predict_number'] = predicts['predict_number'].strip('—')
    if not predicts['predict_number']:
        predicts['predict_number'] = 'не определён'
    return predicts

def get_summary(predicts: dict, summarizer: pipeline) -> dict:
    text = predicts['predict_text']
    predicts['predict_summary'] = summarizer(text)[0]['summary_text']
    return predicts
    

def get_yolo_autocomplete(pdf_file: Union[Path,IO[bytes]], 
                                                     model_predict_classes: YOLO, 
                                                     model_handwrite: YOLO,
                                                     comparison: dict,
                                                     partners: pd.DataFrame,
                                                     partners_employees: pd.DataFrame,
                                                     CLASSES: list,
                                                     summarizer: pipeline,
                                                     get_df = False,
                                                     exists_id = ['predict_sender', 'predict_signer']):
    
    predicts = get_yolo_predicts(pdf_file, model_predict_classes, model_handwrite, CLASSES)
    predicts = update_predicts(predicts, CLASSES)
    predicts = set_proba_for_classes(predicts, CLASSES)
    
    # get sender and sender id
    predicts = get_sender(predicts, comparison, partners)
    # get signer and signer id
    predicts = get_signer(predicts, partners_employees)
    # get date on format datetime.date
    predicts = get_date(predicts)
    # get number
    predicts = get_numbers(predicts)
    # get summary text
    predicts = get_summary(predicts, summarizer)
    
    
    if isinstance(pdf_file, Path):
        doc_id = doc_id = pdf_file.parents[0].name
        pred = {}
        pred[doc_id] = predicts
        if get_df:
            predicts = pd.DataFrame.from_dict(pred, orient='index').reset_index()
            predicts.rename(columns={'index':'doc_id'}, inplace=True)
            return predicts
        else: 
            return pred
        
       
    elif isinstance(pdf_file, bytes):        
        pred = {}
        for cls in CLASSES:
            proba_cls = cls + '_proba'
            if cls in exists_id:
                cls_id = cls + '_id'
                pred[cls] = {cls: predicts[cls], proba_cls: predicts[proba_cls], cls_id:predicts[cls_id]}
            else:
                pred[cls] = {cls: predicts[cls], proba_cls: predicts[proba_cls]}
                     
    return pred


def get_metrics(df, meta):
    df_merge = merge_dfs_by_id(df, meta)
    df_merge = df_merge[df_merge.kind == 'Письмо']
    
    df_merge['is_sender'] = df_merge.apply(lambda row: row.sender_id in row.predict_sender_id, axis=1)
    df_merge['is_signer'] = df_merge.apply(lambda row: row.signer_id in row.predict_signer_id, axis=1)
    df_merge['is_date'] = df_merge.apply(lambda row: row.predict_date == row.out_reg_date, axis=1)
    df_merge['is_number'] = df_merge.apply(lambda row: row.predict_number == row.outgoing_number, axis = 1)
    df_merge['is_content'] = df_merge.apply(lambda row: row.predict_content == row.content, axis=1)
    
    metrics = {}
    metrics['date_accuracy'] = len(df_merge[df_merge['is_date'] == True]) / len(df_merge)
    metrics['number_accuracy'] = len(df_merge[df_merge['is_number'] == True]) / len(df_merge)
    metrics['content_accuracy'] = len(df_merge[df_merge['is_content'] == True]) / len(df_merge)
    metrics['signer_accuracy'] = len(df_merge[df_merge['is_signer'] == True]) / len(df_merge)
    metrics['sender_accuracy'] = len(df_merge[df_merge['is_sender'] == True]) / len(df_merge)
    
    print(f"""
    date accuracy: {metrics['date_accuracy']}
    number accuracy: {metrics['number_accuracy']}
    content accuracy: {metrics['content_accuracy']}
    signer_accuracy: {metrics['signer_accuracy']}
    sender_accuracy: {metrics['sender_accuracy']}""")
    
    return metrics           

def check_comparison(row: str, comparison: dict) -> str:
    if row in comparison.keys():
        return comparison[row]
    else:
        return row


def get_meta(path: Path) -> pd.DataFrame:
    extractor = PrioritetExtractor()
    df_meta = extractor.create_meta_df(path)
    df_meta = df_meta.reset_index()
    #df_meta = df_meta[['doc_id','sender', 'signer', 'sender_id', 'signer_id']]    
    return df_meta


def merge_dfs_by_id(df_pred: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    df_true = df_ref[df_ref.doc_id.isin(df_pred.doc_id.to_list())]
    df_pred = df_pred[df_pred.doc_id.isin(df_true.doc_id.to_list())]
    #df_true = df_true[['doc_id', 'signer', 'sender', 'sender_id', 'signer_id']]
    df_predict = df_true.merge(df_pred, on='doc_id')    
    return df_predict


def get_similarity(text: str, row_reference: str, estimator: str) -> float:
    if len(text) > 50:
        return 0
    if estimator == 'lev_distance':
        return fuzz.token_sort_ratio(text, row_reference)
    elif estimator == 'jaro_winkler':
        return distance.get_jaro_distance(text.lower(), row_reference.lower(), winkler=True, scaling=0.1) * 100


def format_fullname(fullname: str, fio_format=True) -> str:
    # форматирование ФИО в ИОФ или ИОФ в ФИО, удаление лишних знаков и пробелов
    pattern_iof = r'\b[A-Я]\.\s{0,8}[А-Я]\.\s{0,8}[A-Я][a-я]{2,34}'
    pattern_fio = r'\b[А-Я][а-я]{2,34}\s{0,8}[A-Я]\.\s{0,8}[А-Я]\.?'
    split_name = 'bad_format'
    try:
        # проверка на корректность написания ФИО (поиск ИвановИ. И.)
        pattern_mistakes = r'[а-я]{1}[А-Я]{1}'
        if re.findall(pattern_mistakes, fullname):
            x = re.findall(pattern_mistakes, fullname)[0]
            x = x[:1] + ' ' + x[1:]
            fullname = re.sub(pattern_mistakes, x, fullname)
        if fio_format:  # 'Фамилия И. О.'
            if re.findall(pattern_fio, fullname):  # ФИО
                split_name = fullname.replace('.', ' ').split()  # [Ф, И, О]
                # удаление лишних пробелов, объединение [Ф, И, О] в одну строку 'Фамилия И. О.'
                format_name = '{} {}.{}.'.format(''.join(split_name[0].split()),
                                                 ''.join(split_name[1].split()),
                                                 ''.join(split_name[2].split()))  # -> 'Фамилия И. О.'

            elif re.findall(pattern_iof, fullname):  # ИОФ
                split_name = fullname.replace('.', ' ').split()  # [И, О, Ф]
                # удаление лишних пробелов, объединение [И, О, Ф] в одну строку 'Фамилия И. О.'
                format_name = '{} {}.{}.'.format(''.join(split_name[2].split()),
                                                 ''.join(split_name[0].split()),
                                                 ''.join(split_name[1].split()))  # -> 'Фамилия И. О.'
            else:
                format_name = fullname
        else:  # 'И. О. Фамилия'
            if re.findall(pattern_iof, fullname):  # ИОФ
                split_name = fullname.replace('.', ' ').split()  # [И, О, Ф]
                # удаление лишних пробелов, объединение [И, О, Ф] в одну строку 'И.О. Фамилия'
                format_name = '{}.{}. {}'.format(''.join(split_name[0].split()),
                                                 ''.join(split_name[1].split()),
                                                 ''.join(split_name[2].split()))  # -> 'И. О. Фамилия'

            elif re.findall(pattern_fio, fullname):  # ФИО
                split_name = fullname.replace('.', ' ').split()  # [Ф, И, О]
                # удаление лишних пробелов, объединение [Ф, И, О] в одну строку 'И.О. Фамилия'
                format_name = '{}.{}. {}'.format(''.join(split_name[1].split()),
                                                 ''.join(split_name[2].split()),
                                                 ''.join(split_name[0].split()))  # -> 'И. О. Фамилия'
            else:
                format_name = fullname
        return format_name
    except Exception:
        print('fullname: {},split_name: {} - неправильный формат имени'.format(fullname, split_name))
        return fullname


def format_fio(row:str)-> str:
    
    row = re.sub(r'\d\.|\d', ' ', row) # удаление чисел и чисел с точками
     # удаление небуквенных символов, удаление лишних пробелов по бокам
    if len(re.sub(r'[^\w\s\.]', '', row).strip()) != 0:
        row = re.sub(r'[^\w\s\.]', '', row).strip()
    row = re.sub(r'[\s]+', ' ', row) # замена пробелов на один пробел
    
    
    fio = sorted(list(filter(None, re.split('\s|\.', row))), key=lambda x: len(x), reverse=True)
   

    if len(fio) == 3:
        fio[0] = fio[0].capitalize() + ' '
        if len(fio[1]) == 1:
            fio[1] = fio[1].upper()+ '.'
        if len(fio[2]) == 1:
            fio[2] = fio[2].upper()+ '.'        
        row = ''.join(fio)
   

    elif len(fio) ==2:
        fio[0] = fio[0].capitalize() + ' '
        if len(fio[1]) == 1:
            fio[1] == fio[1].upper() + '.'
        row = ''.join(fio)
        
        
    elif len(fio) == 1:
        fio[0] == fio[0].capitalize()
        row = ''.join(fio)
        
    return row
    
    
def clear_text_signer_sender(df:pd.DataFrame, signer:str='SIGNER', sender:str='SENDER') -> pd.DataFrame:
    
    df[signer] = df[signer].apply(lambda row: format_fio(str(row)))
    df[signer] = df[signer].apply(lambda row: format_fullname(row))
    df[sender] = df[sender].fillna('не определён')
    df[sender] = df[sender].apply(lambda row: re.sub(r'[^\w\s\.]', ' ', row.lower()).strip())
    df[sender] = df[sender].replace('', 'не определён')    
    return df


def replace_initials(text, initial = ['подписант','Руководитель', 'Руководитель Р.Р.', 'руководитель', 
                                      'получатель', 'фамилия', 'ФИО', 'Подписант', 'Получатель', 'Фамилия']):
    
    """ функция заменяет Фамилию по шаблону на None """
    
    initials = np.where([bool(re.match(x, text, re.IGNORECASE)) for x in initial], 1,0).sum()
    if initials == 0:
        return text
    else:
        return None
    
    
def split_fio(fio: str) -> list:
    split_fio = list(filter(None, re.split('\s|\.', fio)))
    if len(split_fio) == 3:
        return split_fio
    elif len(split_fio) == 2:
        split_fio.append(None)
        return split_fio
    elif len(split_fio) == 1:
        split_fio.extend([None, None])
        return split_fio
    elif len(split_fio) == 0:
        return [None,None,None]
    else:
        return split_fio[:3]  
    
    

def word_filter(words):
    words = words.strip().split()
    simple_words = []
    for i in words:
        i = re.sub(r'[^\w\s\.]', ' ', i).strip()
        for j in i.split():
            simple_words.append(j)
    return simple_words


def check_word(word: str, check: str) -> bool:
    try:
        if distance.get_jaro_distance(word, check, winkler=True, scaling=0.1) * 100 > 90:
            return 1
        else:
            return 0
    except distance.JaroDistanceException:
        return 0

    
def get_matrix(predict: list, target: list) ->list:
    matrix = []
    for row in target:
        cols = []
        for col in predict:
            cols.append(check_word(row,col))
        matrix.append(cols)
    matrix = np.array(matrix)
    return matrix


def tp_fp_fn(predict,target):
    predicts = word_filter(predict)
    targets = word_filter(target)
    tp = get_matrix(predicts, targets).sum()
    fp = len(predicts) - tp
    if fp < 0:
        fp = 0
    fn = len(targets) - tp
    if fn < 0:
        fn = 0
    return tp,fp,fn


def get_precision(predict,target):
    tp,fp,fn = tp_fp_fn(predict, target)
    return round(tp/(tp+fp), 3)


def get_recall(predict,target):
    tp,fp,fn = tp_fp_fn(predict, target)
    return round(tp/(tp+fn), 3)


def get_features_with_metrics_for_yolov_predicts(df: pd.DataFrame) -> pd.DataFrame:
    df['sim_signer_yolov'] = df.apply(lambda row: get_similarity(row.predict_signer, row.signer, 'jaro_winkler'),axis=1)
    df['sim_sender_yolov'] = df.apply(lambda row: distance.get_jaro_distance(row.predict_sender, 
                                                                             row.sender, 
                                                                             winkler=True, 
                                                                             scaling=0.1) * 100, axis=1)
    df['precision_sender_yolov'] = df.apply(lambda row: get_precision(row.predict_sender, row.sender),axis=1)
    df['recall_sender_yolov'] = df.apply(lambda row: get_recall(row.predict_sender, row.sender),axis=1)    
    return df

                   
def compare_sender(sender, comparison = None):
    
    for key, values in comparison.items():
        values.append(key)
        compare_score = []
        for row in values:
            compare_score.append(distance.get_jaro_distance(sender, row, winkler=True, scaling=0.1))
        if max(compare_score) >= 0.9:
            return key
    return sender


def check_sender(sender, row_reference, return_value = False):
    if (type(sender) != str) or (type(row_reference) != str):
        if return_value:
            return 0
        return False

    sender = sender.lower()
    sender = re.sub(r'[#_"\']*', '', sender)
    row_reference = re.sub(r'[#_"\']*', '', row_reference)
    
    if (sim := fuzz.token_sort_ratio(sender, row_reference)) >= 95:
            if return_value:
                return sim
            return True
    elif (sim := fuzz.partial_ratio(sender, row_reference)) >= 90:
            if return_value:
                return sim
            return True
    if return_value:
        return 0
    return False
    
    

###################################### SHOW PDF ###########################################################


import base64


class PDF(object):
    def __init__(self, pdf, size=(640,640)):
        self.pdf = pdf
        self.size = size

    def _repr_html_(self):
        return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

    def _repr_latex_(self):
        return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

    
def open_pdf(pdf_file):
    
    with open(pdf_file, "rb") as pdf_file:
            encoded_pdf = base64.b64encode(pdf_file.read())
        
    src = "data:application/pdf;base64," + str(encoded_pdf, "utf-8")
    return PDF(src)


def show_pdf(doc_id: str):
    path_to_doc = Path('/data/docs/mer/vh_full/')/doc_id
    pdf_docs = []
    for doc in path_to_doc.iterdir():
        if doc.suffix == '.pdf':
            pdf_docs.append(doc)
    if len(pdf_docs) > 1:
        inp = int(input('в папке {0} pdf, какой открыть?'.format(len(pdf_docs))))
        return open_pdf(pdf_docs[inp])        
    else:
        return open_pdf(pdf_docs[0])
