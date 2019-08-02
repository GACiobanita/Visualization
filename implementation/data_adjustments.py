import string
import pandas as pd
import os
from nltk.corpus import stopwords


class DataAdjustment(object):

    def __init__(self):
        self.STOP_WORDS = set(stopwords.words("english"))

    @staticmethod
    def create_dict_from_tuple(tuples):
        data = {}
        for k, v in tuples:
            data[k] = int(v)
        return data

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

    def remove_duplicate_rows_from_csv(self, csv_files, output_file_path):
        for csv in csv_files:
            data = pd.read_csv(csv)
            data = data.drop_duplicates()
            self.save_csv_file(csv, data, output_file_path)

    @staticmethod
    def save_csv_file(file_name, data, output_folder_path):
        head, tail = os.path.split(file_name)
        data.to_csv(output_folder_path + "\\" + tail, index=None,
                    header=True)
