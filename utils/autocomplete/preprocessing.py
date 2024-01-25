import pandas as pd
import numpy as np
from pathlib import Path
import re
import math
from utils.extraction.prioritet_extractor import PrioritetExtractor

from tqdm import tqdm
tqdm.pandas()


# ################################### PARSING AND UNPACK FEATURES##############################################


def extract_pdf(path: Path) -> pd.DataFrame:
    extractor = PrioritetExtractor(file_handler='pdfminer')
    ext_doc = extractor.create_dataframe(path)
    ext_doc.dropna(inplace=True, axis=0)
    ext_doc.reset_index(inplace=True)
    return ext_doc


def create_features_from_files(df: pd.DataFrame) -> pd.DataFrame:
    """создание фичей из фичи files - список с мета инфой по документу и фичой 
       со словарями по каждой странице
    """
    files_df = pd.DataFrame([{**{'doc_id': row['doc_id']}, **x} for i, row in df.iterrows() for x in row['files']])
    # drop empty list with text and empty pages
    files_df.dropna(subset='pages', inplace=True, axis=0)
    files_df = files_df[files_df.pages.str.len() != 0]
    return files_df


def create_features_from_pages(df: pd.DataFrame) -> pd.DataFrame:
    """создание фичей из фичи pages(список со словарями по каждой странице),
       на выходе df, где одна строка - одна страница, в каждой фиче списки с инфой 
       по каждому текстовому блоку"""
    pages_df = pd.DataFrame([{**{'doc_id': row['doc_id']}, **x} for i, row
                             in df.iterrows() for x in row['pages']])
    return pages_df


def create_features_from_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """создание отдельных строк для каждого текстового блока.
       В фичах инфа по каждому текстовому блоку"""
    blocks_df = pd.DataFrame([{**{'doc_id': row['doc_id'],
                                  'page_num': row['page_num'],
                                  'meta': row['meta']}, **x} for i, row
                              in df.iterrows() for x in row['blocks']])
    return blocks_df


def create_extract_df(ext_doc: pd.DataFrame) -> pd.DataFrame:
    files_df = create_features_from_files(ext_doc)
    pages_df = create_features_from_pages(files_df)
    blocks_df = create_features_from_blocks(pages_df)
    return blocks_df


# ########### UNPACK COORDINATES, FOUND FIRST PAGE, SPLIT TEXT  BY CASE  ######################################

class NotFoundFirstPageException(Exception):
    """класс вызывается, если в документе не найдена первая страница. Основные сущности находятся
       на первой странице, использовать документ для предикта без первой страницы не целесообразно"""
    pass


def has_first_page(df: pd.DataFrame) -> bool:
    """ Имеет ли документ первую страницу
    """
    if int(df.page_num.unique()[0]) == 1:
        return True
    else:
        return False


def is_upper(text: str) -> bool:
    """Функция возвращает истину, если количество букв в верхнем регистре
       относится ко всем символам в строке больше чем 0.5"""
    uppercase_sum = [1 for symb in text if symb.isupper()]
    rate = sum(uppercase_sum) / (len(text) + 0.000001)
    if rate > 0.5:
        return True
    else:
        return False


def process_block(block: list) -> list:
    """ Принимает на вход список строк. Объединяет строки с одинаковым регистром букв,
        возвращает изменённый список"""
    if not block:
        return []
    new_example = []
    current_line = block[0]
    prev_status = is_upper(block[0])
    for line in block[1:]:
        curr_status = is_upper(line)
        if prev_status == curr_status:
            current_line += ' ' + line
        else:
            new_example.append(current_line)
            current_line = line
            prev_status = curr_status

    new_example.append(current_line)

    return new_example


def concate_str_same_case(df: pd.DataFrame) -> pd.DataFrame:
    # объединение строк в верхнем регистре и строк в нижнем регистре в единые строки в одном текстовом блоке
    df_same_case = df.copy()
    df_same_case['text'] = df_same_case.text.progress_apply(lambda x: process_block(x.split('\n')))
    return df_same_case


def unpack_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    # преобразование списка минимальных и максимальных координат в отдельные столбцы
    df_meta = df.meta.progress_apply(pd.Series)
    df_meta.rename({0: 'Xmin', 1: 'Ymin', 2: 'Xmax', 3: 'Ymax'}, axis=1, inplace=True)
    df = df.merge(df_meta, right_index=True, left_index=True).drop('meta', axis=1)

    # преобразование списка координат текстовых блоков в отдельные столбцы
    df_xy = df.coords.progress_apply(pd.Series)
    df_xy.rename({0: 'xbot', 1: 'ybot', 2: 'xtop', 3: 'ytop'}, axis=1, inplace=True)
    df = df.merge(df_xy, right_index=True, left_index=True) \
        .drop('coords', axis=1)
    # преобразуем списки текстов в блоках в отдельные строки датафрейма
    # в случае, если функция concate_str_same_case() не разбила текст на подстроки,
    # возвращается исходная строка
    s = df.progress_apply(lambda x: pd.Series(x['text']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'text'
    df = df.drop('text', axis=1).join(s)
    return df


def drop_not_properly_oriented_pages(df: pd.DataFrame) -> pd.DataFrame:
    orient_first_page = df.iloc[0]['Xmax'] // df.iloc[0]['Ymax']  # ориентация 1й страницы
    # вернём страницы с той же ориентацией, что и первая
    return df[df['Xmax'] // df['Ymax'] == orient_first_page]


def create_unpack_df(df_extract: pd.DataFrame) -> pd.DataFrame:
    try:
        if has_first_page(df_extract):  # проверка наличия первой страницы
            df_unpack = concate_str_same_case(df_extract)  # объединение соседних строк в одинаковом регистре
            df_unpack = unpack_coordinates(df_unpack)  # распаковка координат
            df_unpack = drop_not_properly_oriented_pages(
                df_unpack)  # удаление страниц с ориентацией отличной от первой страницы
            return df_unpack
        else:
            raise NotFoundFirstPageException(f'not found first page for doc: {df_extract.doc_id.iloc[0]}')
    except NotFoundFirstPageException as notfound:
        print(notfound)


def only_pages_in_order(pages_list: list) -> list:
    """ возвращает список номеров страниц идущих по порядку начиная с первой
        если порядок нарушается, то возвращает номера страниц до момента нарушения порядка
        [1,2,4,5] -> [1,2]
    """
    if pages_list == [1]:
        return pages_list
    else:
        pages_list_in_order = []
        for i in range(1, len(pages_list) + 1):
            if i in pages_list:
                pages_list_in_order.append(i)
            else:
                break
        return pages_list_in_order


def deleting_pages_out_of_order(doc: pd.DataFrame.groupby) -> pd.DataFrame.groupby:
    """возвращает документ с теми страницами, которые шли по порядку, начиная с первой.
       Если страница пропущена, то последующие страницы не возвращаются
       [1,2,4,5] -> [1,2]
    """
    pages_list = doc.page_num.unique().tolist()
    pages_list_in_order = only_pages_in_order(pages_list)
    if pages_list_in_order:
        doc = doc[doc.page_num.isin(pages_list_in_order)]
        return doc


# ##################### CREATE DF WITH PDF (PARSING,UNPACK) - ОБЪЕДИНЕНИЕ ПРЕДЫДУЩИХ ФУНКЦИЙ ###################

def create_df_from_pdf(path: Path) -> pd.DataFrame:
    ext_doc = extract_pdf(path)
    df_meta = create_extract_df(ext_doc)
    df_unpack = create_unpack_df(df_meta)
    df = df_unpack.groupby('doc_id', group_keys=False).apply(lambda doc: deleting_pages_out_of_order(doc))
    del ext_doc
    del df_meta
    del df_unpack
    return df


# ######################################## FEATURE ENGINEERING ################################################


def number_uppercase(text: str) -> int:
    """ Подсчёт количества букв в верхнем регистре и соотношение их к длине всего текста
    """
    upper_counter = np.where([x.isupper() for x in text], 1, 0).sum()
    return upper_counter


def character_identifier(text: str) -> list:
    """ Возвращает долевое соотношение:
        - количества кириллических символов к общему количеству символов в блоке
        - количества латинских символов к общему количеству символов в блоке
        - количества числовых символов к общему количеству символов в блоке
        - количества пробелов к общему количеству символов в блоке
        - количества иных символов к общему количеству символов в блоке
    """
    patterns = {
        'cyrillic': r'[а-яА-ЯёЁ]',
        'latin': r'[a-zA-Z]',
        'numbers': r'[0-9]',
        'spaces': r'\s',
        'other': r'[^а-яА-ЯёЁa-zA-Z0-9\s]'}

    ratios = []
    for pattern in patterns.values():
        counter = len(re.findall(pattern, text))
        ratio = round(counter / len(text), 5)
        ratios.append(ratio)

    return ratios


def add_location_features(page: pd.DataFrame.groupby) -> pd.DataFrame.groupby:
    # фичи порядковых отношений текстовых блоков на странице
    page['blocks_on_the_left'] = list(range(len(page)))
    page['blocks_on_the_right'] = list(range(len(page) - 1, -1, -1))
    page['location_on_the_page'] = page['blocks_on_the_left'] / len(page)

    page['location_ratio_left'] = ((page['blocks_on_the_left'] + 1) / (page['blocks_on_the_right'] + 1))
    page['location_ratio_left'] = page['location_ratio_left'].apply(lambda row: math.log(row, 2))

    page['location_ratio_right'] = ((page['blocks_on_the_right'] + 1) / (page['blocks_on_the_left'] + 1))
    page['location_ratio_right'] = page['location_ratio_right'].apply(lambda row: math.log(row, 2))
    return page


def drop_pages_less_three_blocks(doc: pd.DataFrame.groupby) -> pd.DataFrame.groupby:
    # удаление страниц с количеством текстовых блоков менее 3х
    if doc.shape[0] > 2:
        return doc


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """ Создание и добавление в датафрейм фичей
     """

    df = df[df.text != ''].reset_index(drop=True)
    # список жирности шрифта
    bold_font_list = list(df[df['font_name'].str.contains(r'Bold', na=False)].font_name.unique())
    bold_font_list.extend(df[df['font_name'].str.contains(r'Black', na=False)].font_name.unique())
    bold_font_list.extend(df[df['font_name'].str.contains(r'Heavy', na=False)].font_name.unique())

    # Создание фичи толщины текста. 0 - стандартный текст, 1 - жирный текст
    df['font_bold'] = df.font_name.isin(bold_font_list)

    # В текстовых блоках присутствуют блоки только с пробелами, 
    # причём с разным количеством пробелов - Заменим все возможные комбинации пробелов 
    # на один пробел во всех блоках и удалим блоки состоящие только из пробелов
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x))

    # удаление блоков с одними пробелами
    df = df[df.text != ' ']

    # Фича с длинной блока
    df['blocks_len'] = df.apply(lambda row: len(row['text']), axis=1)

    # удаление блоков с количеством символов меньше 4
    df.drop(df[df['blocks_len'] < 4].index, inplace=True)

    # Фича с количеством слов
    df['words_count'] = df.apply(lambda row: len(row['text'].split()), axis=1)

    # Фича со средней длиной слова
    df['mean_word_len'] = df['blocks_len'] / df['words_count']

    # Фича горизонтального или вертикального расположения листа
    df['vertical'] = df['Xmax'] / df['Ymax']
    df['vertical'] = np.where((df.vertical < 0.9), 1, 0)

    # нормализация фичей относительно каждого отдельного документа на основе спарсенных координат
    groups = df.groupby(['doc_id', 'page_num'])

    df['xbot'] = df.xbot / groups.xtop.cummax()
    df['ybot'] = df.ybot / groups.ytop.cummax()
    df['xtop'] = df.xtop / groups.xtop.cummax()
    df['ytop'] = df.ytop / groups.ytop.cummax()

    # фича ширины и высоты каждого текстового блока
    df['height'] = df.ytop - df.ybot
    df['width'] = df.xtop - df.xbot

    # фичи отношения к левой части страницы или к правой (чем меньше значение тем правее блок)
    df['left_right'] = df['width'] / df['xtop']

    # фичи отношения к верху или низу страницы (чем меньше значение тем выше блок)
    df['up_down'] = df['height'] / df['ytop']

    # фичи отношения ширины и высоты блока к ширине и высоте страницы
    df['width_ratio'] = df.width / groups.xtop.cummax()
    df['height_ratio'] = df.height / groups.ytop.cummax()

    # фича определяющая отношение высоты к ширине блока, чем ближе к единице, тем квадратнее
    df['squareness'] = df.height / df.width

    # площадь блока
    df['square'] = df.height * df.width

    # отношение площади блока к площади всей страницы
    df['square_ratio'] = df.square / (groups.xtop.cummax() * groups.ytop.cummax())

    # расстояние по осям Х и Y до предыдущего блока
    df['distance_y'] = df['ybot'].shift(1) - df['ytop']
    df['distance_x'] = df['xbot'] - df['xtop'].shift(1)
    df['distance_y'].fillna(0, inplace=True)
    df['distance_x'].fillna(0, inplace=True)

    # подсчёт количества букв в верхнем регистре и соотношение их к длине всего текста
    df['number_uppercase_letters'] = df['text'].apply(lambda x: number_uppercase(x))
    df['uppercase_letter_ratio'] = df['number_uppercase_letters'] / df['blocks_len']

    # отношения различных видов символов к количеству всех символов в блоке
    df['is_cirillic'], df['is_latin'], df['is_number'], df['is_space'], df['is_other'] = \
        zip(*df['text'].apply(lambda x: character_identifier(x)))

    # Количество точек в текстовом блоке. В блоке out_signed_by должно быть от 1 до 3 точек в среднем
    df['dot_count'] = df.text.apply(lambda x: x.count('.'))
    # количество точек в блоке, отношение количества точек к общему количеству символов
    df['dot_ratio'] = df['dot_count'] / df['words_count']

    df.rename(columns={'font_size_pt': 'font_size'}, inplace=True)
    # добавление фичи первой страницы:
    df['first_page'] = df.page_num == 1
    # добавление фичей порядковых отношений блоков на странице
    df = df.groupby(['doc_id', 'page_num']).progress_apply(add_location_features)
    # удаление "вертикального" текста
    df = df[df['squareness'] <= 1]
    
    # удаление неправильно распарсенных строк
#     df = df[(df['is_cirillic'] > 0.2)|(df['is_number']> 0.3)]
#     df = df[(df.is_other < 0.25) & (df.is_space < 0.25)]
    
    # удаление страниц с количеством блоков меньше 3х
    df = df.groupby('doc_id', group_keys=False).apply(lambda doc: drop_pages_less_three_blocks(doc))
    df.reset_index(drop=True, inplace=True)
    return df

# ###################  CREATE DF AUTOCOMPLETE (объединение всех функций в конечную) #####################


def get_df(path: Path) -> pd.DataFrame:
    df_pdf = create_df_from_pdf(path)
    df = feature_engineering(df_pdf)
    return df
