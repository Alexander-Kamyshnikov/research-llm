import re
from typing import Tuple, Any

import numpy as np
import pandas as pd
import datetime
from pathlib import Path

from rapidfuzz import fuzz
from pyjarowinkler import distance

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils.autocomplete import predictions
from utils.autocomplete import preprocessing

import typing as tp
import time
from tqdm import tqdm

tqdm.pandas()


# ############################ MARKUP SENDER SIGNER ##########################################################


def get_similarity(text: str, row_reference: str, estimator: str) -> float:
    if len(text) > 50:
        return 0
    if estimator == 'lev_distance':
        return fuzz.token_sort_ratio(text, row_reference)
    elif estimator == 'jaro_winkler':
        return distance.get_jaro_distance(text.lower(), row_reference.lower(), winkler=True, scaling=0.1) * 100


def get_similarity_partial_sender(ybot: float, text: str, row_reference: pd.DataFrame,
                                  field: str, estimator: str) -> float:
    if (ybot < 0.68) | (len(text) > 200):
        return 0
    reference = row_reference[field].iloc[0]

    if estimator == 'lev_distance':
        return fuzz.partial_ratio(text.lower(), reference.lower())
    elif estimator == 'jaro_winkler':
        return distance.get_jaro_distance(text.lower(), reference.lower(), winkler=True, scaling=0.1) * 100


def get_space_ratio(text: str) -> float:
    cyrillic_pattern = r'[а-яА-ЯёЁ]'
    cyrillic_counter = len(re.findall(cyrillic_pattern, text)) + 0.0001
    space_counter = text.count(' ')

    return space_counter / cyrillic_counter


# Функция убирает лишние пробелы в найденных фамилиях и инициалах, если таковые остались
def replace_space(text: str) -> str:
    if get_space_ratio(text) < 0.6:
        return text
    else:
        return text.replace(' ', '')


def print_score(predicts: np.array, test_data: pd.DataFrame) -> print:
    recall = recall_score(test_data.target, predicts, average=None)
    precision = precision_score(test_data.target, predicts, average=None)
    recall_macro = recall_score(test_data.target, predicts, average='macro')
    precision_macro = precision_score(test_data.target, predicts, average='macro')
    scores = {'recall': recall[1], 'precision': precision[1],
              'recall_macro': recall_macro, 'precision_macro': precision_macro}
    print('recall_other: {},\n recall_class: {},\nprecis_other: {},\n precis_class: {},\nrecall_macro: {}, precis_macro: {}'.
          format(recall[0], recall[1], precision[0], precision[1], recall_macro, precision_macro))
    return scores


def check_names(text: str, ner, return_text=None, ) -> tp.Optional[str]:
    # проверка - является ли строка фамилией с инициалами (ФИО, ИОФ)
    # или не является
    names = [text[x.start:x.stop] for x in ner(text).spans if x.type == 'PER']
    pattern = r'[А-Я]\.[А-Я]\.\s{0,3}[А-Я][а-я]{2,30}|[А-Я][а-я]{2,30}\s{0,3}[А-Я]\.[А-Я]\.'
    if len(names) == 0:
        names = re.findall(pattern, text)
    if len(names) != 0:
        return names[0]
    else:
        return return_text


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


def clear_reference(df: pd.DataFrame, df_meta: pd.DataFrame, features_for_compare: list) -> Tuple[Any, Any]:
    # чистка справочника от значений таргетной фичи отсутствующих в 
    # исследуемом датафрейме и в самом справочнике
    df_meta = df_meta.dropna(subset=features_for_compare)
    features_for_compare.append('doc_id')
    df_meta = df_meta[features_for_compare]
    df = df[df.doc_id.isin(df_meta.doc_id)]
    df_meta = df_meta[df_meta.doc_id.isin(df.doc_id)]
    return df, df_meta


# ###################################### GET META ###########################################################


def get_meta(path: Path) -> pd.DataFrame:
    df = preprocessing.extract_pdf(path)
    df = df[['doc_id', 'reg_date', 'content', 'sender', 'sender_id', 'signer', 'signer_id',
             'outgoing_number', 'out_reg_date', 'recipient']]
    df.rename(columns={'signer': 'out_signed_by', 'signer_id': 'out_signed_by_id'}, inplace=True)
    df['out_signed_by'] = df['out_signed_by'].apply(lambda row: format_fullname(row, fio_format=False))

    return df


# ############################### MARKUP SENDER SIGNER #######################################################


def check_fullname(text: str, row_reference: pd.DataFrame) -> tuple:
    # проверка соответствия исследуемой строки и имени в справочнике паттерну ИОФ,
    # в случае соответствмия возвращаем строку и сравниваемое значение без изменений для
    # дальнейшего сравнения. В случае несовпадения с паттерном ИОФ, для сравнения оставляем только фамилии
    if len(text) > 50:
        return text, row_reference
    else:
        pattern_iof = r'\b[A-Я]\.\s{0,8}[А-Я]\.\s{0,8}[A-Я][a-я]{2,34}'
        fullname_df = format_fullname(text, fio_format=False)
        fullname_reference = format_fullname(row_reference['out_signed_by'].iloc[0], fio_format=False)
        if bool(re.findall(pattern_iof, fullname_df)) & bool(re.findall(pattern_iof, fullname_reference)):
            return fullname_df, fullname_reference
        else:
            fullname_df = max(fullname_df.split(), key=len)
            fullname_reference = max(fullname_reference.split(), key=len)
            return fullname_df, fullname_reference


def markup_sender(row_df: pd.DataFrame, row_reference: pd.DataFrame,
                  estimator_sender='jaro_winkler',
                  threshold=90) -> str:
    ybot = row_df.ybot
    text = row_df.text
    if get_similarity_partial_sender(ybot, text, row_reference,
                                     'sender', estimator_sender) >= threshold:
        return 'sender'
    else:
        return 'other'


def get_df_with_marked_sender(df: pd.DataFrame,
                              df_meta: pd.DataFrame,
                              threshold=90) -> pd.DataFrame:
    # отправитель всегда на первой странице, удаление страниц кроме первой
    df_send = df[df.page_num == 1]
    # разметка df для предсказательной модели сендера
    df_send, reference = clear_reference(df_send, df_meta, ['sender'])
    df_sender = df_send.copy()
    df_sender['target'] = df_sender.progress_apply(lambda row:
                                                   markup_sender(row,
                                                                 reference[reference.doc_id == row.doc_id],
                                                                 threshold=threshold), axis=1)
    
    df_sender['target'] = df_sender['target'].astype('category')
    # чистка полученного df
    df_sender.drop_duplicates(inplace=True)
    # если только один класс other - удаляем такую страницу:
    df_sender = df_sender.groupby(['doc_id', 'page_num']).filter(lambda group: (group.target.nunique() > 1) |
                                                                               (group.target.unique()[0] != 'other'))
    del df_send
    return df_sender


def markup_signer(row_df: pd.DataFrame, row_reference: pd.DataFrame,
                  estimator_signer='lev_distance',
                  ner=None,
                  threshold=60) -> str:
    
    text = row_df.text
    text, row_reference = check_fullname(text, row_reference)
    if get_similarity(text, row_reference, estimator_signer) > threshold:
        if bool(check_names(text, ner)):
            return 'out_signed_by'
        else:
            return 'other'
    else:
        return 'other'


def get_df_with_marked_signer(df: pd.DataFrame, df_meta: pd.DataFrame,
                              ner=None,
                              threshold=60) -> pd.DataFrame:
    # разметка df для предсказательной модели сигнера
    df_sign, reference = clear_reference(df, df_meta, ['out_signed_by'])
    df_signer = df_sign.copy()
    df_signer['target'] = df_signer.progress_apply(lambda row:
                                                   markup_signer(row_df=row,
                                                                 row_reference=reference[
                                                                     reference.doc_id == row.doc_id],
                                                                 ner=ner,
                                                                 threshold=threshold), axis=1)
    df_signer['target'] = df_signer['target'].astype('category')

    # если на странице только один класс other - удаляем такую страницу:
    df_signer = df_signer.groupby(['doc_id', 'page_num']).filter(lambda group: (group.target.nunique() > 1) |
                                                                               (group.target.unique()[0] != 'other'))
    del df_sign
    return df_signer


# ################################# MARKUP DATE ############################################################

def get_df_with_marked_date(df: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    df_date, reference = clear_reference(df, df_meta, ['out_reg_date'])
    # # определим строки с датами, остальное удалим
    df_date = df_date.groupby('doc_id').progress_apply(lambda group:
                                                       predictions.search_date_df(group, one_value_for_doc=False))
    df_date.dropna(inplace=True)
    df_date = df_date.reset_index(drop=True)
    # фичи отражающие координатное расположение даты в текстовом блоке:
    df_date['start_ratio'] = df_date['start_date'] / len(df_date.text)
    df_date['end_ratio'] = df_date['end_date'] / len(df_date.text)
    # фича истинной даты
    df_date = df_date.merge(reference[['doc_id', 'out_reg_date']], on='doc_id')
    # Исключение из фичей дат - часов, минут, секунд
    # Приведение обоих фичей к типу datetime
    df_date.out_reg_date = pd.to_datetime(df_date.out_reg_date).dt.date
    df_date.date = pd.to_datetime(df_date.date, errors='coerce').dt.date
    # Исключение  всех дат ранее 2019 года
    df_date = df_date[df_date.date > datetime.date(2019, 1, 1)]

    # таргетная фича
    df_date['target'] = df_date.progress_apply(lambda row: 'date' if row.date == row.out_reg_date else 'other', axis=1)
    return df_date


# ############################# MARKUP OUTGOING NUMBER ###################################################

def get_df_with_marked_outgoing_number(df: pd.DataFrame, df_meta) -> pd.DataFrame:
    df_number, reference = clear_reference(df, df_meta, ['outgoing_number'])
    df_number = predictions.found_number_in_text(df_number)
    df_number = df_number.merge(reference, on='doc_id')
    df_number['target'] = df_number.apply(lambda row:
                                          'number' if row.found_outgoing_number == row.outgoing_number
                                          else 'other', axis=1)
    return df_number


# ################################           ML            ###################################################


def train_test_val(df: pd.DataFrame, val_data: pd.DataFrame, entitie: str) -> tuple:
    val = val_data.rename(columns={'meta_guid': 'doc_id'})
    entities = {'sender': 'Sender', 'signer': 'Signer',
                'outgoing_number': 'OutgoingNumber', 'date': 'OutgoingDate'}
    features = list(['doc_id'])
    features.append(entities[entitie])
    val = val[features].dropna()

    df_train = df[~df['doc_id'].isin(val['doc_id'])]
    train_data, test_data = train_test_split(df_train, test_size=0.15, random_state=43)
    return train_data, test_data, val


def train_model(df: pd.DataFrame, val: pd.DataFrame, entitie: str,
                features=None):
    if not features:
        features = ['page_num', 'font_size', 'font_bold', 'number_uppercase_letters',
                    'uppercase_letter_ratio', 'dot_count', 'dot_ratio', 'is_cirillic',
                    'is_latin', 'is_number', 'is_space', 'is_other', 'Xmin', 'Ymin', 'Xmax', 'Ymax',
                    'xbot', 'ybot', 'xtop', 'ytop', 'left_right', 'up_down', 'width_ratio', 'height_ratio',
                    'squareness', 'square', 'square_ratio', 'vertical', 'distance_x', 'distance_y', 'height',
                    'width', 'blocks_len', 'words_count', 'mean_word_len', 'first_page', 'blocks_on_the_left',
                    'blocks_on_the_right', 'location_on_the_page', 'location_ratio_left', 'location_ratio_right']
    train, test, val = train_test_val(df, val, entitie)
    model = RandomForestClassifier(random_state=43)
    start_time = time.time()
    model.fit(train[features], train.target)
    predicts = model.predict(test[features])
    print(entitie.upper())
    print('\n{}.sec\n'.format(time.time() - start_time))
    cm = confusion_matrix(test.target, predicts, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()
#     plot_confusion_matrix(model, test[features], test.target)
    scores = print_score(predicts, test)
    return model, scores


# ############################################## VALIDATE ################################################


def only_one_predict_sender(df: pd.DataFrame.groupby) -> str:
    predict_sender = predictions.join_predict_with_neighbours(df)
    return predict_sender


def predict_only_one_sender_for_each_doc(df_val: pd.DataFrame,
                                         model: RandomForestClassifier,
                                         features=None) -> pd.DataFrame:
    if not features:
        features = ['page_num', 'font_size', 'font_bold', 'number_uppercase_letters',
                    'uppercase_letter_ratio', 'dot_count', 'dot_ratio', 'is_cirillic',
                    'is_latin', 'is_number', 'is_space', 'is_other', 'Xmin', 'Ymin', 'Xmax', 'Ymax',
                    'xbot', 'ybot', 'xtop', 'ytop', 'left_right', 'up_down', 'width_ratio', 'height_ratio',
                    'squareness', 'square', 'square_ratio', 'vertical', 'distance_x', 'distance_y', 'height',
                    'width', 'blocks_len', 'words_count', 'mean_word_len', 'first_page', 'blocks_on_the_left',
                    'blocks_on_the_right', 'location_on_the_page', 'location_ratio_left', 'location_ratio_right']

        # вероятность принаждлежности к целевому классу для каждой строки
    df_validate = df_val.copy()
    df_validate['sender_proba'] = model.predict_proba(df_validate[features])[:, 1]
    # исключение из предиктов сендеров найденных не на первой странице
    df_validate = df_validate[df_validate.page_num == 1]
    # определение  одного сендера для каждого документа
    # при необходимости ячейки соседние с максимально вероятным предиктом объединяются
    predict_senders = df_validate.groupby('doc_id').progress_apply(lambda doc: only_one_predict_sender(doc))
    predict_senders = pd.DataFrame(predict_senders).reset_index()
    predict_senders.rename(columns={0: 'predict_sender'}, inplace=True)
    return predict_senders


def similarity_predict_with_true_sender(df: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    # фича истинного сендера для каждого документа

    predict_senders = df.merge(val, on='doc_id')
    predict_senders.rename(columns={'Sender': 'true_sender'}, inplace=True)
    predict_senders['true_sender'] = predict_senders['true_sender'].apply(lambda row:
                                                                          row.replace('\n', '').lower())
    predict_senders['predict_sender'] = predict_senders['predict_sender'].apply(lambda row:
                                                                                row.lower())
    predict_senders['similarity'] = predict_senders.progress_apply(lambda row: fuzz.partial_ratio(row.true_sender,
                                                                                                  row.predict_sender),
                                                                   axis=1)
    return predict_senders


# определение одного сендера для каждого документа
def validate_sender(df: pd.DataFrame, val: pd.DataFrame, model: RandomForestClassifier) -> pd.DataFrame:
    df_val = df[df.doc_id.isin(val.doc_id.to_list())]
    predict_senders = predict_only_one_sender_for_each_doc(df_val, model)
    predict_senders = similarity_predict_with_true_sender(predict_senders, val)
    return predict_senders


def predict_only_one_signer_for_each_doc(df_val: pd.DataFrame,
                                         model: RandomForestClassifier,
                                         features=None) -> pd.DataFrame:
    if not features:
        features = ['page_num', 'font_size', 'font_bold', 'number_uppercase_letters',
                    'uppercase_letter_ratio', 'dot_count', 'dot_ratio', 'is_cirillic',
                    'is_latin', 'is_number', 'is_space', 'is_other', 'Xmin', 'Ymin', 'Xmax', 'Ymax',
                    'xbot', 'ybot', 'xtop', 'ytop', 'left_right', 'up_down', 'width_ratio', 'height_ratio',
                    'squareness', 'square', 'square_ratio', 'vertical', 'distance_x', 'distance_y', 'height',
                    'width', 'blocks_len', 'words_count', 'mean_word_len', 'first_page', 'blocks_on_the_left',
                    'blocks_on_the_right', 'location_on_the_page', 'location_ratio_left', 'location_ratio_right']

    predicts = df_val.groupby('doc_id').progress_apply(lambda doc:
                                                       predictions.predict_signer(df=doc,
                                                                                  features=features,
                                                                                  model_signer_prediction=model))
    predict_signers = pd.DataFrame(predicts).reset_index()
    predict_signers.rename(columns={0: 'predict_signer'}, inplace=True)

    return predict_signers


def similarity_predict_with_true_signer(df: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    predict_signers = df.merge(val, on='doc_id')
    predict_signers.rename(columns={'Signer': 'true_signer'}, inplace=True)
    predict_signers['predict_signer'].fillna('no_name', inplace=True)
    predict_signers['similarity'] = predict_signers.progress_apply(lambda row: get_similarity(
        row.predict_signer,
        row.true_signer,
        estimator='lev_distance'),
                                                                   axis=1)
    return predict_signers


def validate_signer(df: pd.DataFrame, val: pd.DataFrame, model: RandomForestClassifier) -> pd.DataFrame:
    df_val = df[df.doc_id.isin(val.doc_id.to_list())]
    predict_signers = predict_only_one_signer_for_each_doc(df_val, model)
    predict_signers = similarity_predict_with_true_signer(predict_signers, val)

    return predict_signers


def predict_only_one_date_for_each_doc(df_val: pd.DataFrame,
                                       model: RandomForestClassifier,
                                       features=None,
                                       one_value_for_doc=False) -> pd.DataFrame:
    if not features:
        features = ['page_num', 'font_size', 'font_bold', 'number_uppercase_letters',
                    'uppercase_letter_ratio', 'dot_count', 'dot_ratio', 'is_cirillic',
                    'is_latin', 'is_number', 'is_space', 'is_other', 'Xmin', 'Ymin', 'Xmax', 'Ymax',
                    'xbot', 'ybot', 'xtop', 'ytop', 'left_right', 'up_down', 'width_ratio', 'height_ratio',
                    'squareness', 'square', 'square_ratio', 'vertical', 'distance_x', 'distance_y', 'height',
                    'width', 'blocks_len', 'words_count', 'mean_word_len', 'first_page', 'blocks_on_the_left',
                    'blocks_on_the_right', 'location_on_the_page', 'location_ratio_left', 'location_ratio_right',
                    'start_date', 'end_date', 'ratio_date', 'start_ratio', 'end_ratio']

    predicts = df_val.groupby('doc_id').progress_apply(lambda doc:
                                                       predictions.predict_date_df(df=doc,
                                                                                   features=features,
                                                                                   model_date_prediction=model,
                                                                                   one_value_for_doc=one_value_for_doc))
    predict_dates = pd.DataFrame(predicts).reset_index()
    predict_dates.rename(columns={0: 'predict_date'}, inplace=True)

    return predict_dates


def similarity_predict_with_true_date(df: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    val.rename(columns={'OutgoingDate': 'out_reg_date'}, inplace=True)
    
    val['out_reg_date'] = val['out_reg_date'].apply(lambda row: predictions.search_date(str(row))[0])
    val['out_reg_date'] = pd.to_datetime(val['out_reg_date'], errors='coerce', dayfirst=True).dt.date    
    val.dropna(inplace=True)
    
    predict_dates = df.merge(val, on='doc_id')
    predict_dates['similarity'] = predict_dates.apply(lambda row:
                                                      1 if row.predict_date == row.out_reg_date else 0, axis=1)
    return predict_dates


def validate_date(df: pd.DataFrame, val: pd.DataFrame,
                  model: RandomForestClassifier,
                  features=None,
                  one_value_for_doc=False) -> pd.DataFrame:
    df_val = df[df.doc_id.isin(val.doc_id.to_list())]
    predict_dates = predict_only_one_date_for_each_doc(df_val, model, features, one_value_for_doc=one_value_for_doc)
    predict_dates = similarity_predict_with_true_date(predict_dates, val)

    return predict_dates


def predict_only_one_number_for_each_doc(df_val: pd.DataFrame,
                                         model: RandomForestClassifier,
                                         features=None) -> pd.DataFrame:
    predicts = df_val.groupby('doc_id').progress_apply(lambda doc: predictions.predict_number(doc,
                                                                                              model,
                                                                                              features=features))
    predict_numbers = pd.DataFrame(predicts).reset_index()
    predict_numbers.rename(columns={0: 'predict_number'}, inplace=True)

    return predict_numbers


def similarity_predict_with_true_number(df: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    val.rename(columns={'OutgoingNumber': 'outgoing_number'}, inplace=True)
    predict_numbers = df.merge(val, on='doc_id')
    # замена None на 0 для возможности чистки пробелов в найденных номерах
    predict_numbers['predict_number'] = predict_numbers['predict_number'].fillna('*')
    predict_numbers['predict_number'] = predict_numbers.predict_number.apply(lambda row:
                                                                             row.replace(' ', ''))
    predict_numbers['outgoing_number'] = predict_numbers.outgoing_number.apply(lambda row:
                                                                               row.replace(' ', ''))
    predict_numbers['similarity'] = predict_numbers.progress_apply(lambda row:
                                                                   fuzz.partial_ratio(row.predict_number,
                                                                                      row.outgoing_number),
                                                                   axis=1)
    return predict_numbers


def validate_number(df: pd.DataFrame, val: pd.DataFrame, model: RandomForestClassifier) -> pd.DataFrame:
    df_val = df[df.doc_id.isin(val.doc_id.to_list())]
    predict_numbers = predict_only_one_number_for_each_doc(df_val, model, features=None)
    predict_numbers = similarity_predict_with_true_number(predict_numbers, val)

    return predict_numbers
