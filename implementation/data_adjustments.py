import string
import pandas as pd
import os
import calendar
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from inspect import getsourcefile


class DataAdjustment(object):

    def __init__(self, emoticon_lexicon='emoticon_lexicon.txt', vader_lexicon='vader_lexicon.txt'):
        module_file_path = os.path.abspath(getsourcefile(lambda: 0))
        self.emoticon_lexicon_file_path = os.path.join(os.path.dirname(module_file_path), emoticon_lexicon)
        self.vader_lexicon_file_path = os.path.join(os.path.dirname(module_file_path), vader_lexicon)

        self.emoticon_lexicon = self.get_emoticon_lexicon()
        self.vader_lexicon = self.get_vader_lexicon()

        self.STOP_WORDS = set(stopwords.words("english"))

    def get_vader_lexicon(self):
        word_list = {}
        with open(self.vader_lexicon_file_path, encoding='utf-8') as f:
            file = f.read()
        for line in file.split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            word_list[word] = float(measure)
        return word_list

    def get_emoticon_lexicon(self):
        word_list = list()
        with open(self.emoticon_lexicon_file_path, encoding='utf-8') as f:
            file = f.read()
        for line in file.split('\n'):
            word_list.append(line)
        return word_list

    @staticmethod
    def get_dict_from_tuple(tuples):
        data = {}
        for k, v in tuples:
            data[k] = int(v)
        return data

    @staticmethod
    def get_string_frequency_distribution(strings_container):
        return FreqDist(strings_container)

    @staticmethod
    def tokenize_words(words_string):
        return word_tokenize(words_string)

    @staticmethod
    def remove_string_punctuation(data):
        translator = str.maketrans('', '', string.punctuation)
        return data.translate(translator)

    def remove_string_stopwords(self, data):
        data_no_sw = []
        for word in data:
            if word not in self.STOP_WORDS:
                data_no_sw.append(word)
        return data_no_sw

    @staticmethod
    def remove_duplicate_rows_from_csv(csv_files, output_file_path):
        for csv in csv_files:
            data = pd.read_csv(csv)
            data = data.drop_duplicates()
        return csv_files, data, output_file_path

    @staticmethod
    def save_csv_file(file_name, data, output_folder_path):
        head, tail = os.path.split(file_name)
        data.to_csv(output_folder_path + "\\" + tail, index=None,
                    header=True)

    @staticmethod
    def merge_dictionaries(dict1, dict2):
        dict3 = {**dict1, **dict2}
        return dict3

    def separate_emoticons_and_words(self, init_dict):
        emoticon_dict = {}
        word_dict = {}
        for key, value in init_dict.items():
            lowered_key = str(key).lower()
            # exclude words that have no sentiment value in the vader lexicon
            if lowered_key in self.vader_lexicon:
                # check in a separate emoticon lexicon if the key is an emoticon :D, ;) ......
                if key in self.emoticon_lexicon:
                    emoticon_dict[key] = value
                else:
                    word_dict[lowered_key] = value
        return emoticon_dict, word_dict

    def get_valence_from_lexicon(self, item):
        return self.vader_lexicon[item.lower()]

    @staticmethod
    def get_month_from_int(number):
        return calendar.month_name[int(number)]

    @staticmethod
    def get_year_from_string(full_str):
        year_search = re.search('\d{4}', full_str)
        year = year_search.group(0) if year_search else 'XXXX'
        return year

    @staticmethod
    def get_file_name_from_path(full_str):
        head, tail = os.path.split(full_str)
        return tail[:-4]

    def concatenate_csv_data(self, output_folder_path, file_name, first_set, second_set, first_set_name,
                             second_set_column):
        first_set[first_set_name] = second_set[second_set_column]
        first_set.to_csv(output_folder_path + "\\" + file_name, index=None,
                         header=True)
