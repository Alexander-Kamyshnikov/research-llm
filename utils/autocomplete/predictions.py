import re
import pandas as pd
import typing as tp
import dateutil
import datetime
from rapidfuzz import process, fuzz
from pyjarowinkler import distance
from sklearn.ensemble import RandomForestClassifier

DEFAULT_FEATURES = ['page_num', 'font_size', 'font_bold', 'number_uppercase_letters',
                    'uppercase_letter_ratio', 'dot_count', 'dot_ratio', 'is_cirillic',
                    'is_latin', 'is_number', 'is_space', 'is_other', 'Xmin', 'Ymin', 'Xmax', 'Ymax',
                    'xbot', 'ybot', 'xtop', 'ytop', 'left_right', 'up_down', 'width_ratio', 'height_ratio',
                    'squareness', 'square', 'square_ratio', 'vertical', 'distance_x', 'distance_y', 'height',
                    'width', 'blocks_len', 'words_count', 'mean_word_len', 'first_page', 'blocks_on_the_left',
                    'blocks_on_the_right', 'location_on_the_page', 'location_ratio_left', 'location_ratio_right']


# ############################# CONTENT PREDICTION ##################################################

def search_content(text: str) -> tp.Optional[str]:
    """
       Принимает на вход текстовый блок строкового формата
       на выходе: 
                 - отформатированный контент ,если это контент
                 - None, если текст не относится к контенту

    """

    # Предыдущая версия регулярки: re.match(r'О\s[а-яА-Я]{3,100}|Об\s[а-яА-Я]{3,100}', text)
    content = re.search(r'(\s|^)О\s[а-яА-Я]{3,100}.*|(\s|^)Об\s[а-яА-Я]{3,100}.*', text)

    if content is None or len(content.group(0)) > 200:
        return None

    content = content.group(0)
    # Убираем из строк места для ввода рукописного текста: '__________'
    content = ' '.join(content.split('_'))
    return content.strip()


def predict_content(df: pd.DataFrame) -> str:
    df_content = df[df.page_num == 1].sort_values(by='ytop', ascending=False)
    df_content['content'] = df_content.text.apply(lambda x: search_content(x))
    df_content.dropna(inplace=True, axis=0)

    if df_content.empty:
        content = None
    else:
        content = df_content.iloc[0].content

    return content


# ################################ DATE PREDICTION ##################################################


def month_translator(month: str) -> str:
    """ Перевод русскоязычных названий месяцев в англоязычные.
        Требуется, так как библиотека dateutil не работает c
        кириллическими символами.
    """

    months = {'января': 'Jan',
              'февраля': 'Feb',
              'марта': 'Mar',
              'апреля': 'Apr',
              'мая': 'May',
              'июня': 'Jun',
              'июля': 'Jul',
              'августа': 'Aug',
              'сентября': 'Sept',
              'октября': 'Oct',
              'ноября': 'Nov',
              'декабря': 'Dec'}
    for russian, english in months.items():
        month = month.replace(russian, english)
    return month


def search_date(text: str, except_date_from=False) -> tuple:
    """
       Поиск в текстовых полях дат.
       Три варианта:
       - дата написана в числовом формате;
       - дата записана с названиями месяцев на русском языке;
       - нет дат.
       В первом случае возвращаем найденную дату.
       Во втором переводим русские месяцы в английские, и возвращаем дату.
       В третьем возвращаем None
       во всех случаях возвращаем также индексы найденных дат в строке с начала строки и с конца строки
       для случая когда дата не найдена, возвращаем -1 и для start_date и для end_date
    """

    # исключение из возможных дат, значений находящихся в строке "На_________от________"
    if except_date_from:
        date_from_one = r'(на)?\s*№?[\w\W]{0,20}от\s*[\w\W]{0,20}'
        text_without_from = re.sub(date_from_one, ' ', text, flags=re.IGNORECASE)
    else:
        text_without_from = text

    # парсинг возможных дат, два варианта - месяц написан числом, месяц написан словом

    date_one = re.search(r'(\s*\d{1,2}\.\d{2}\.\d{2,4})', text_without_from, flags=re.IGNORECASE)
    date_two = re.search(
        r'(\s*.?\d{1,2}.?\s*(января|февраля|марта|апреля|мая|'
        r'июня|июля|августа|сентября|октября|ноября|декабря)\s*\d{2,4})',
        text_without_from, flags=re.IGNORECASE)

    if date_one:
        start_date, end_date = date_one.start(), date_one.end()
        return date_one[0], start_date, end_date
    elif date_two:
        start_date, end_date = date_two.start(), date_two.end()
        date_two = date_two[0].replace('«', '')
        date_two = date_two.replace('»', '')
        temp = ''
        for i in date_two.split():
            temp += month_translator(i) + ' '
        return temp, start_date, end_date
    else:
        return None, -1, -1


def convert_date(date: str) -> tp.Optional[datetime.date]:
    """ Поиск и конвертация дат в числовой формат: год, месяц, дата"""
    try:
        return dateutil.parser.parse(date)
    except Exception:
        return None


def search_date_df(df: pd.DataFrame, one_value_for_doc=False,
                   y_bot=0.6, except_date_from=False) -> tp.Optional[pd.DataFrame]:
    """
       Поиск дат в поле text датафрейма.
       Фильтрация по:
        - первой странице;
        - нулевым значениям;
        - расположение даты выше 0,6 от ymax;
        - самая ранняя дата не ранее 01.01.2019;
        - выбор в одном документе даты с наивысшим расположением в документе;
       Создание фичи отношения длины даты к длине всего блока.
       Парсинг строковых дат различного вида в вид YYYY-MM-DD.
       Конвертация найденных дат в формат datetime.date

       """
    df_date = df.copy()
    # фильтрация по первой странице и расположению на странице
    df_date = df_date[(df_date.page_num == 1) & (df_date.ybot > y_bot)]
    if df_date.empty or df_date is None:
        return None
    else:
        df_date['date'], df_date['start_date'], df_date['end_date'] = zip(
            *df_date.text.apply(lambda x: search_date(x, except_date_from=except_date_from)))
        df_date.dropna(inplace=True)

    if df_date.empty or df_date is None:
        return None
    else:
        # фича отношения длины даты к длине всего текстового блока
        df_date['ratio_date'] = df_date.apply(lambda x: len(x.date) / x.blocks_len, axis=1)
        # приведение найденных дат к виду YYYY-MM-DD
        df_date['date'] = pd.to_datetime(df_date.date, errors='coerce', dayfirst=True).dt.date
        df_date.dropna(inplace=True)
        # фичи отражающие координатное расположение даты в текстовом блоке:
        df_date['start_ratio'] = df_date['start_date'] / len(df_date.text)
        df_date['end_ratio'] = df_date['end_date'] / len(df_date.text)

        df_date.dropna(inplace=True, axis=0)
        if df_date.empty or df_date is None:
            return None
        else:
            # Исключение всех дат ранее 2019 года
            df_date = df_date[df_date.date > datetime.date(2019, 1, 1)]
            if one_value_for_doc:
                # выбор для документа только одной возможной даты по наибольшему ytop
                found_date = df_date.sort_values(by=['ytop'],
                                                 ascending=False).iloc[0]
                return found_date

            return df_date


def predict_date(df: pd.DataFrame, model: RandomForestClassifier, features=None) -> tp.Optional[str]:
    if len(df) == 0 or df is None:
        return None
    else:
        if not features:
            features = DEFAULT_FEATURES.copy()
            features.extend(['start_date', 'end_date', 'ratio_date', 'start_ratio', 'end_ratio'])

        # предсказание даты
        # фичи отражающие координатное расположение даты в текстовом блоке:
        df['start_ratio'] = df['start_date'] / len(df.text)
        df['end_ratio'] = df['end_date'] / len(df.text)
        # predictions
        predictions_proba = model.predict_proba(df[features])
        # фича предсказаний
        df['target_proba'] = predictions_proba[:, 1]
        # если не осталось ни одной даты, возвращаем None
        # если даты есть, возвращаем дату с наибольшим ybot и с наибольшей вероятностью
        if len(df) == 0:
            date = None
        else:
            date = df.sort_values(by=['ybot', 'xbot', 'target_proba'],
                                  ascending=[False, True, False]).iloc[0].date
        return date


def predict_date_df(df: pd.DataFrame,
                    model_date_prediction: RandomForestClassifier,
                    features=None,
                    one_value_for_doc=False,
                    except_date_from=False) -> str:
    if not features:
        features = DEFAULT_FEATURES.copy()
        features.extend(['start_date', 'end_date', 'ratio_date', 'start_ratio', 'end_ratio'])

    df_date = search_date_df(df, one_value_for_doc=one_value_for_doc, 
                             y_bot=0.6, except_date_from=except_date_from)
    
    if df_date is None:
        date = None
    else:
        date = predict_date(df_date, model=model_date_prediction, features=features)
    return date


# ######################################## Outgoing Number Prediction ####################################


def search_number(text: str) -> tp.Optional[str]:
    """
       Поиск в текстовых полях номеров документов.
    """

    number_one = re.search(r'№\s*[\S]+|№', text)
    if number_one:
        return number_one[0]
    else:
        return None
    

def search_empty_number(text: str) -> bool:
    """ Если за знаком номера не следует буквенно-числовая последовательность,
        то сущность считается незаполненной и возвращается True,
        в противном случае False"""

    text = text.strip()
    empty_pattern = r'№\s*_*$|№\s*_*от_*$|№\.*$|№\s?\.*от.*$|№\s*(на|На)?$|№\s*$|№$'
    empty_number = re.search(empty_pattern, text, flags=re.IGNORECASE)
    if empty_number:
        return False
    else:
        return True


def found_number_in_text(df_numb: pd.DataFrame) -> tp.Optional[pd.DataFrame]:
    df = df_numb.copy()
    df = df[(df.page_num == 1) & (df.ybot > 0.6)].reset_index(drop=True)
    # поиск в текстовых блоках знака "№" после которого ненулевая последовательность
    # не пробельных символов (числа, буквы, знаки препинания, и прочее)
    df['found_outgoing_number'] = df.text.apply(lambda x: search_number(x))
    df.dropna(inplace=True)

    if len(df) == 0:
        return None
    else:
        # удаление незаполненных полей регистрационных номеров
        df = df[df.found_outgoing_number.apply(lambda x: search_empty_number(x))]

        # очистка найденных полей от лишних символов
        # удаление полей содержащих пустые строки
        df.found_outgoing_number = df.found_outgoing_number.str.strip(r'№|на|от|_*|!')
        df.found_outgoing_number = df.found_outgoing_number.str.strip()
        df = df[df.found_outgoing_number != '']
    if len(df) == 0:
        return None
    else:
        return df


def predict_outgoing_number(df: pd.DataFrame, model: RandomForestClassifier,
                            features: tp.Optional[list]) -> tp.Optional[str]:
    if not features:
        features = DEFAULT_FEATURES

    #         with open(PATH_TO_MODEL_PREDICT_NUMBER, 'wb') as f:
    #             model = pickle.load(f)

    # предсказание исходящего номера из найденных в тексте номеров
    predictions_proba = model.predict_proba(df[features])
    df['target_proba'] = predictions_proba[:, 1]
#     df = df[df['target_proba'] > 0.3]

    if len(df) == 0 or df is None:
        outgoing_number = None
    else:
        outgoing_number = df.sort_values(by=['ybot', 'target_proba'], ascending=False).iloc[0].found_outgoing_number

    return outgoing_number


def predict_number(df: pd.DataFrame,
                   model_number_prediction: RandomForestClassifier,
                   features=None) -> tp.Optional[str]:
    df_number = df.copy()
    if not features:
        features = DEFAULT_FEATURES

    df_number = found_number_in_text(df_number)
    if df_number is None:
        outgoing_number = None
    else:
        outgoing_number = predict_outgoing_number(df_number, model_number_prediction, features)
    return outgoing_number


# ################################### Sender Signer Predictions ##############################################


def predict_signer(df: pd.DataFrame,
                   model_signer_prediction: RandomForestClassifier,
                   features: tp.Optional[list] = None) -> tp.Optional[str]:
    if not features:
        features = DEFAULT_FEATURES

    rfc_predictions_signer = model_signer_prediction.predict_proba(df[features])
    df['out_signed_by_proba'] = rfc_predictions_signer[:, 1]
    df = df[df.out_signed_by_proba >= 0.5]
    if df.empty:
        return None
    else:
        # сортировка данных по наивысшему скору для каждого документа,
        # выбор только той строки, в которой скор выше всех
        df = df[['text', 'out_signed_by_proba']].sort_values(by=['out_signed_by_proba'], ascending=False)
        return df.iloc[0].text


def join_predict_with_neighbours(df: pd.DataFrame) -> str:
    """сущность sender в тексте часто разбивается парсером на две или более ячеек.
       Модель предсказывает только одну вероятную сущность.
       Функция позволяет объединять соседние с предиктом ячейки,
       если таковые соседние ячейки имеют вероятность больше определённого порога
    """
    # определение ячееек соседних максимальной вероятности
    df_neighbours = df[(df.index == df.sender_proba.idxmax()) |
                       (df.index == df.sender_proba.idxmax() - 1) |
                       (df.index == df.sender_proba.idxmax() + 1)].reset_index()
    # если вероятность соседней ячейки выше определённого порога, то считаем что такая ячейка,
    # тоже является частью искомой сущности. Ткую ячейку объеденим с найденной самой вероятной сущностью
    df_neighbours['sender_proba'] = df_neighbours.loc[:, 'sender_proba'] / df_neighbours.loc[:, 'sender_proba'].max()

    return ' '.join(df_neighbours[df_neighbours.sender_proba > 0.3].text.to_list())


def predict_sender(df: pd.DataFrame,
                   model_sender_prediction: RandomForestClassifier,
                   features: tp.Optional[list] = None) -> tp.Optional[str]:
    if not features:
        features = DEFAULT_FEATURES
    try:
        df_first_page = df[df.page_num == 1].reset_index(drop=True)
        rfc_predictions_sender = model_sender_prediction.predict_proba(df_first_page[features])
        df_first_page['sender_proba'] = rfc_predictions_sender[:, 1]
        # вариант возврата одного самого вероятного сендера и при большой вероятности 
        # у соседних с предиктом сущностей, объединение таких сущностей с предиктом.
        if df_first_page.sender_proba.max() > 0.4:
            sender = join_predict_with_neighbours(df_first_page)
            return sender
        else:
            return None
    except ValueError:
        print(f'первая страница документа должна быть на русском языке: {df["doc_id"].iloc[0]}')


# ###################################### Predict all classes #########################################

def predict_entities(df: pd.DataFrame,
                     model_sender_prediction: RandomForestClassifier,
                     model_signer_prediction: RandomForestClassifier,
                     model_date_prediction: RandomForestClassifier,
                     model_number_prediction: RandomForestClassifier,
                     features=None) -> dict:

    if not features:
        features = DEFAULT_FEATURES.copy()

    features_date = features.copy()
    features_date.extend(['start_date', 'end_date', 'ratio_date', 'start_ratio', 'end_ratio'])

    predicted_entities = {'predict_content': predict_content(df),
                          'predict_date': predict_date_df(df, model_date_prediction, features_date),
                          'predict_number': predict_number(df, model_number_prediction, features),
                          'predict_out_signed_by': predict_signer(df, model_signer_prediction, features),
                          'predict_sender': predict_sender(df, model_sender_prediction, features)}

    # проверка на наличие экземпляров всех искомых сущностей
    predicted_entities['all_classes'] = all(predicted_entities.values())
    return predicted_entities


# ############################# Search persons and organization from predicted signer and sender #############


def search_person_in_reference(name: str, reference_signer_sender: pd.DataFrame, limit=5) -> pd.DataFrame:
    names = process.extract(name, list(reference_signer_sender.Name), limit=limit)
    name_list = [el for el in names if el[1] >= 95]

    # Минимум один подписант должен быть в списке
    if not name_list:
        name_list.append(names[0])

    index_list = [i[2] for i in name_list]
    row_id_persons = reference_signer_sender.iloc[index_list]

    return row_id_persons


def search_organization_by_person_in_reference(org: str, df_persons: pd.DataFrame) -> pd.DataFrame:
    name_list = process.extractOne(org, list(df_persons.Organization))
    row_index = name_list[2]
    row_id_org = df_persons.iloc[[row_index]]
    return row_id_org


def predict_guid_signer_sender(signer: str, sender: str, reference: pd.DataFrame) -> pd.DataFrame:
    guid_df = search_organization_by_person_in_reference(sender,
                                                         search_person_in_reference(signer,
                                                                                    reference))
    return guid_df


def predict_guid(out_signed_by: str, sender: str, reference: pd.DataFrame) -> pd.DataFrame:
    """
    Сендер - предсказанная организация,
    Сигнер - предсказанный подписант,
    Организация - организации в справочнике.
    Подписант (персона) - подписант в справочнике.

    У нас есть предсказанные сендер и сигнер. В самом корне поиска нужного подписанта находится организация,
    так как каждый подписант привязан к своей организации и наименование организации уникально в то время как
    фамилии могут повторяться от организации к организации.
    Поэтому, сначала организуем поиск организации по предсказанному сендеру. Найдём все организации имеющие
    коэффициент схожести (sim) c сендером более определённого порога (по умолчанию 60).
    Далее в найденных организациях будем искать подписанта с наибольшей схожестью с сигнером.
    Отсортируем найденные сущности по схожести сендера с организациями и по схожести сигнера с подписантами.

    В случае, если у нас нет организации с sim больше 60, считаем что предсказанный сендер мало похож
    на искомый. Поэтому в таком случае будем искать организацию по подписанту.
    Существует опасность поиска организаций по фамилиям, так как могут присутствовать однофамильцы в разных
    организациях. Но так как сендер мало похож на организации (sim < 60) то в таком случае поиск организации
    по сендеру ещё более рисковый, так как по факту мы будем искать организацию по мало похожей сущности,
    в итоге можно найти всё что угодно. Поэтому - ищем по сигнеру все персоны в справочнике со схожестью
    больше определённого порога (по умолчанию 90). По найденным фамилиям ищем организации с большей
    схожестью с сендером. Сортируем полученный датафрейм по схожести сигнера с персонами и
    по схожести сендера с организациями.

    В случае, если нет персоны со схожестью с сигнером более 90, тогда ищем наилучшее по схожести
    пересечение персоны и организации. (Метод организованный ранее)

    """

    df_org_by_sender = reference.copy()
    df_org_by_sender['sim_org_by_sender'] = reference.apply(lambda x: fuzz.ratio(sender.lower(), x.Organization),
                                                            axis=1)
    df_org_by_sender = df_org_by_sender[df_org_by_sender.sim_org_by_sender > 60]
    if not df_org_by_sender.empty:
        df_org_by_sender['sim_per_by_signer'] = reference.apply(lambda x:
                                                                process.extractOne(out_signed_by,
                                                                                   [x.Name])[1], axis=1)
        df_org_by_sender = df_org_by_sender.sort_values(by=['sim_org_by_sender',
                                                            'sim_per_by_signer'], ascending=False)
        return df_org_by_sender[['RowID',
                                 'Name',
                                 'ParentCompany',
                                 'Organization',
                                 'NotAvailable']].iloc[[0]]

    else:
        df_per_by_signer = reference.copy()
        df_per_by_signer['sim_per_by_signer'] = reference.apply(lambda x: process.extractOne(out_signed_by,
                                                                                             [x.Name])[1],
                                                                axis=1)

        df_per_by_signer = df_per_by_signer[df_per_by_signer.sim_per_by_signer > 90]

        if not df_per_by_signer.empty:
            df_per_by_signer['sim_org_by_sender'] = reference.apply(lambda x:
                                                                    process.extractOne(sender,
                                                                                       [x.Organization])[1],
                                                                    axis=1)

            df_per_by_signer = df_per_by_signer.sort_values(by=['sim_per_by_signer',
                                                                'sim_org_by_sender'
                                                                ], ascending=False)
            return df_per_by_signer[['RowID',
                                     'Name',
                                     'ParentCompany',
                                     'Organization',
                                     'NotAvailable']].iloc[[0]]
        else:
            return predict_guid_signer_sender(out_signed_by, sender, reference)


def search_row_id(sender: str, out_signed_by: str, reference: pd.DataFrame, how='multi_sim') -> pd.Series:
    """ Несколько вариантов поиска в справочнике организации и подписанта
        по предсказанным сендеру и сигнеру:
        "multi_sim" предсказанные сендер и сигнер сравниваются со всеми организациями и персонами
            в справочнике, их коэффициенты схожести перемножаются, результат с самым большим произведением
            считается искомым.
        "by_sender" - эвристика описана в функции predict_guid в модуле predictions_utils
        "signer_by_sender" - находим организацию самую схожую с предсказанным сендером,
            далее в найденной организации ищем персону самую схожую с предсказанным сигнером.
        "sender_by_signer" - находим в справочнике фамилию максимально схожую с сигнером,
            далее в справочнике находим все организации имеющие такую персону. Из найденных организаций
            оставляем самую схожую с сендером.
    """
    sim_org_by_sender = reference.copy()

    if how == 'multi_sim':
        sim_org_by_sender['sim_sender_by_org'] = reference.apply(lambda x:
                                                                 fuzz.ratio(sender.lower(), x.Organization), axis=1)
        sim_org_by_sender['sim_signer_by_person'] = reference.apply(lambda x:
                                                                    process.extractOne(out_signed_by,
                                                                                       [x.Name])[1],
                                                                    axis=1)
        sim_org_by_sender['coef'] = sim_org_by_sender.sim_sender_by_org * sim_org_by_sender.sim_signer_by_person
        sort_row = sim_org_by_sender.sort_values(by='coef', ascending=False).iloc[0]
    elif how == 'by_sender':
        sort_row = predict_guid(out_signed_by, sender, reference).iloc[0]
    elif how == 'signer_by_sender':
        org = process.extractOne(sender.lower(), list(reference.Organization))[0]
        reference_org = reference[reference.Organization == org]
        persona = process.extractOne(out_signed_by, list(reference_org.Name))[0]
        sort_row = reference[(reference.Organization == org) & (reference.Name == persona)].iloc[0]
    elif how == 'sender_by_signer':
        persona = process.extractOne(out_signed_by, list(reference.Name))[0]
        reference_persona = reference[reference.Name == persona]
        org = process.extractOne(sender, list(reference_persona.Organization))[0]
        sort_row = reference[(reference.Name == persona) & (reference.Organization == org)].iloc[0]
    else:
        sort_row = "Организация не найдена"

    return sort_row


def search_row_id_multiple_methods(df: pd.DataFrame, reference: pd.DataFrame) -> tuple:
    """Создание 4-х датафреймов найденных организаций и персон по 4-м методам
       функции search_row_id
    """
    df_multi = pd.DataFrame()
    df_by_sender = pd.DataFrame()
    df_signer_by_sender = pd.DataFrame()
    df_sender_by_signer = pd.DataFrame()
    counter = 0
    for el in range(len(df)):
        sender = df.iloc[el]['predict_sender']
        out_signed_by = df.iloc[el]['predict_out_signed_by']

        # 4 варианта поиска организации и персоны:

        df_multi = df_multi.append(search_row_id(sender,
                                                 out_signed_by,
                                                 reference,
                                                 how='multi_sim'))
        df_by_sender = df_by_sender.append(search_row_id(sender,
                                                         out_signed_by,
                                                         reference,
                                                         how='by_sender'))

        df_signer_by_sender = df_signer_by_sender.append(search_row_id(sender,
                                                                       out_signed_by,
                                                                       reference,
                                                                       how='signer_by_sender'))
        df_sender_by_signer = df_sender_by_signer.append(search_row_id(sender,
                                                                       out_signed_by,
                                                                       reference,
                                                                       how='sender_by_signer'))
        counter += 1
        print('in process: {}'.format(counter), end='\r')
    #
    df_end_sim = pd.concat([df.reset_index(), df_multi.reset_index()], axis=1)
    df_end_sender = pd.concat([df.reset_index(), df_by_sender.reset_index()], axis=1)
    df_end_signer_by_sender = pd.concat([df.reset_index(), df_signer_by_sender.reset_index()], axis=1)
    df_end_sender_by_signer = pd.concat([df.reset_index(), df_sender_by_signer.reset_index()], axis=1)

    return df_end_sim, df_end_sender, df_end_signer_by_sender, df_end_sender_by_signer


# ######################## Поиск персон и организаций в справочнике #########################################

def get_jaro_distance_itter(str_1: str, list_2: list, winkler=True, scaling=0.2) -> tuple:
    """сравнение строки со списком строк. На выходе самая близкая строка и её similarity"""
    dist = [distance.get_jaro_distance(i, str_1, winkler=winkler, scaling=scaling) for i in list_2]
    max_distance_index = dist.index(max(dist))
    return list_2[max_distance_index], max(dist)


def search_person_in_reference_jaro(name: str, reference: pd.DataFrame) -> pd.DataFrame:
    name = get_jaro_distance_itter(name, reference.Name.to_list())
    persons = reference[reference.Name == name[0]]
    persons['sim_person_with_reference'] = name[1]
    return persons
