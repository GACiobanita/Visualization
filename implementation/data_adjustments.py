import string
from nltk.corpus import stopwords


class DataAdjustment(object):

    def __init__(self):
        self.STOP_WORDS = set(stopwords.words("english"))

    def create_dict_from_tuple(self, tuples):
        data = {}
        for k, v in tuples:
            data[k] = int(v)
        return data

    def remove_string_punctuation(self, data):
        translator = str.maketrans('', '', string.punctuation)
        return data.translate(translator)

    def remove_string_stopwords(self, data):
        data_no_sw = []
        for word in data:
            if word not in self.STOP_WORDS:
                data_no_sw.append(word)
        return data_no_sw
