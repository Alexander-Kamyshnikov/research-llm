import pandas as pd
from pyjarowinkler import distance
from pathlib import Path
from rapidfuzz import fuzz
import re

from utils.extraction.prioritet_extractor import PrioritetExtractor

from tqdm import tqdm
tqdm.pandas()



def get_meta_for_clear_references(path: Path=Path('/data/docs/mer/vh_full/')):
    extractor = PrioritetExtractor()
    path_pdf = path
    df_meta = extractor.create_meta_df(path_pdf)
    df_meta = df_meta.reset_index()
    df_meta = df_meta[['doc_id','sender', 'signer', 'sender_id', 'signer_id']]
    
    return df_meta

# ########################################### PREPROCESSING_PARTNERS_AND_EMPLOYEES ############################################

def prepcocessing_employees(partners_employees: pd.DataFrame) -> pd.DataFrame:
    
    """ отчистка строк фамилий и инициалов от невалидных значений, удаление пустых фамилий,
        приведение всех первых букв фамилий к заглавной. Преобразование инициалов в первые заглавные буквы
        создание фичи списка из [Ф, И, О], удаление однобуквенных фамилий кроме  Ю, Е, О, И
    """
    nolastname = ['подписант','руководитель', 'специалист', 'получатель', 'фамилия', 'фио']
    
    # удаление не буквенных символов и не строк
    partners_employees.first_name = partners_employees.first_name.apply(lambda row: re.sub('[^А-Яа-яA-Za-z]','', row) 
                                                                        if type(row) == str else None)
    partners_employees.middle_name = partners_employees.middle_name.apply(lambda row: re.sub('[^А-Яа-яA-Za-z]','', row) 
                                                                        if type(row) == str else None)
    partners_employees.last_name = partners_employees.last_name.apply(lambda row: re.sub('[^А-Яа-яA-Za-z]','', row) 
                                                                        if type(row) == str else None)
    # удаление не фамилий ( служебные, невалидные фамилии)
    partners_employees.last_name = partners_employees.last_name.apply(lambda row: None if row.lower() in nolastname 
                                                                     else row if row else None)
    # удаление строк без фамилий
    partners_employees.dropna(subset='last_name', inplace=True)
    # инициалы для имени и отчества
    partners_employees.first_name = partners_employees.first_name.apply(lambda row: row[0].upper() 
                                                                        if type(row) == str and len(row) > 0 else None)
    partners_employees.middle_name = partners_employees.middle_name.apply(lambda row: row[0].upper() 
                                                                          if type(row) == str and len(row) > 0 else None)
    # все фамилии с большой буквы
    partners_employees.last_name = partners_employees.last_name.apply(lambda row: row.capitalize())
    partners_employees['split_fio'] = partners_employees.apply(lambda row: [row.last_name, row.middle_name, row.first_name], axis=1)
    # удаление однобуквенных фамилий (кроме Ю,Е,О - фамилии корейского и китайского происхождения)
    partners_employees = partners_employees[(partners_employees.last_name.agg(len)>1) |
                                                      (partners_employees.last_name.isin(['Ю', 'Е', 'О', 'И']))]
    
    return partners_employees


def prepcocessing_partners(partners: pd.DataFrame) -> pd.DataFrame:
    
    """ удаление  организаций в названиях которых только цифры,
        удаление невалидных названий организаций ( Физлица, #Новые, Новые),
        удаление строк не имеющих названий организаций,
        удаление организаций с длиной имени меньше 3х
    """
    nopartners = ['физлица', 'новые', '#новые', 'test', 'тест']
    
    def is_digit(string: str) -> bool:
        try:
            float(string)
            return True
        except TypeError:
            return False
        except ValueError:
            return False
    
    # удаление строк содержащих только цифры
    partners.partner_name = partners.partner_name.apply(lambda row: None if is_digit(row) else row)
    # удаление невалидных названий организаций ( Физлица, #Новые, Новые)
    partners.partner_name = partners.partner_name.apply(lambda row: (None if row.lower() in nopartners else row) 
                                                        if type(row) == str else None)
    # удаление строк не имеющих названий организаций
    partners.dropna(subset='partner_name', inplace=True)
    # удаление организаций с длиной имени меньше 3х
    partners = partners[partners.partner_name.agg(len) >= 3]
    
    return partners
    
def update_partners(partners: pd.DataFrame) -> pd.DataFrame:
    partners = partners[partners.partner_type == 0]
    partners = partners[partners.partner_not_available == False]
    partners = partners.drop_duplicates(subset=['partner_id', 'partner_name'])
    partners.partner_name = partners.partner_name.apply(lambda row: row.lower() if isinstance(row, str) else row)
    partners.partner_full_name = partners.partner_full_name.apply(lambda row: row.lower() if isinstance(row, str) else row)
    partners = prepcocessing_partners(partners)
    return partners

def update_partners_empoyees(partners_employees: pd.DataFrame) -> pd.DataFrame:
    # partners_employees = partners_employees[partners_employees.employee_not_available == False]
    partners_employees.dropna(subset='last_name', inplace=True)
    partners_employees.rename(columns={'middle_name': 'first_name', 'first_name':'middle_name'}, inplace=True)
    partners_employees = prepcocessing_employees(partners_employees)
    
    return partners_employees

#  ########################################## GET NONUSED ID #################################################################


def get_unused_partner_id(df_meta: pd.DataFrame, df: pd.DataFrame) -> list:
    
    """функция ищет одинаковые организации с разными id и сравнивает эти id c датасетом по всем документам в выборке.
       неиспользуемые id для одной и той же организации удаляются, остаются только id хотя бы раз используемые как
       разметка документа
    """
    
    df_sender = df_meta[['sender', 'sender_id']]
    df_sender = df_sender.drop_duplicates()
    partners = df.copy()
    
    def is_gemini( first: str, second: str, estimator = 'fuzz') -> bool:
    
        """ функция сравнивает две строки, если строки схожи более чем на 90%
            такие строки считаются двойниками
        """
    
        try:
            if estimator == 'jaro':
                if distance.get_jaro_distance(first, second, winkler=True, scaling=0.1) > 0.9:
                    return True
            else:
                if fuzz.token_sort_ratio(first, second) > 95:
                    return True
        except TypeError:
            pass

    def set_key(dictionary, key, value):
        if key not in dictionary:
            dictionary[key] = [value]
        else:
            dictionary[key].append(value)

    def get_gemini(simple_org: str, orgs_dict: dict) -> list:
        geminis = {}
        global keys

        for key in keys:
            if key in orgs_dict.keys():
                if is_gemini(simple_org, orgs_dict[key]):
                    set_key(geminis, key, orgs_dict[key])
        return geminis

    def get_dict_partners(partners: pd.DataFrame) -> dict:
        # Словарь, в котором каждому значению id соответствует имя организации
        keys = partners.partner_id.to_list()
        values = partners.partner_name.to_list()
        orgs = {keys[i]: values[i] for i in range(len(keys))}

        return orgs

    def get_gimini_for_each_partner(partners: pd.DataFrame, orgs: dict) -> pd.DataFrame:
        # Определение для каждой организации двойника
        partners['gemini'] = partners.partner_name.progress_apply(lambda row: get_gemini(row, orgs))
        # Фича количества двойников
        partners['count_gemini'] = partners.gemini.agg(len)

        return partners

    def get_dict_senders(df: pd.DataFrame) -> dict:
        # Словарь, в котором каждому значению id соответствует имя организации
        keys = df_sender.sender_id.to_list()
        values = df_sender.sender.to_list()
        orgs = {keys[i]: values[i] for i in range(len(keys))}

        return orgs

    def get_gimini_for_each_sender(df: pd.DataFrame, orgs: dict) -> pd.DataFrame:
        # Определение для каждой организации двойника
        df_sender['gemini'] = df_sender.sender.progress_apply(lambda row: get_gemini(row, orgs))
        # Фича количества двойников
        df_sender['count_gemini'] = df_sender.gemini.agg(len)
        
        return df_sender

    def get_used_id(gemini_id: list, sender_id: list) -> list:
        return list(set(gemini_id).intersection(sender_id))
    
    partners_orgs = get_dict_partners(partners)
    partners = get_gimini_for_each_partner(partners, partners_orgs)
    
    senders_orgs = get_dict_senders(df_sender)
    df_sender = get_gimini_for_each_sender(df_sender, senders_orgs)
    
    partners['gemini_id'] = partners.gemini.apply(lambda row: list(row.keys()))
    # все id всех сендеров в метa.
    senders_id = df_sender.sender_id.to_list()

    partners['used_sender_id'] = partners.gemini_id.apply(lambda row: get_used_id(row, senders_id))
    partners['numbers_used_id'] = partners.used_sender_id.agg(len)
    
    # выделение неиспользуемых id в отдельную фичу,объединение неиспользуемых айди в список
    partners['nonused_sender_id'] = partners.apply(lambda row: list(set(row.gemini_id) - set(row.used_sender_id)), 
                                                                   axis=1)
    
    unused_id = partners.nonused_sender_id.to_list()
    unused_id = [item for sublist in unused_id for item in sublist]
    
    return unused_id
    
    

