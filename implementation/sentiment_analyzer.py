from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os
from inspect import getsourcefile


def merge_dictionaries(dict1, dict2):
    dict3 = {**dict1, **dict2}
    return dict3


class SentimentAnalyzer:

    def __init__(self, emoticon_lexicon='emoticon_lexicon.txt', vader_lexicon='vader_lexicon.txt'):
        module_file_path = os.path.abspath(getsourcefile(lambda: 0))
        self.emoticon_lexicon_file_path = os.path.join(os.path.dirname(module_file_path), emoticon_lexicon)
        self.vader_lexicon_file_path = os.path.join(os.path.dirname(module_file_path), vader_lexicon)

        self.emoticon_lexicon = self.make_emoticon_lexicon()
        self.vader_lexicon = self.make_vader_lexicon()

        self.analyzer = SentimentIntensityAnalyzer()
        self.csv_files = []

        self.sentiment_dataframes = []
        self.negation_words = {}
        self.words = {}
        self.emoticons = {}
        self.positive_review_count = 0
        self.negative_review_count = 0
        self.neutral_review_count = 0
        self.star_count = {5.0: 0, 4.0: 0, 3.0: 0, 2.0: 0, 1.0: 0}

    def make_vader_lexicon(self):
        word_list = {}
        with open(self.vader_lexicon_file_path, encoding='utf-8') as f:
            file = f.read()
        for line in file.split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            word_list[word] = float(measure)
        return word_list

    def make_emoticon_lexicon(self):
        word_list = list()
        with open(self.emoticon_lexicon_file_path, encoding='utf-8') as f:
            file = f.read()
        for line in file.split('\n'):
            word_list.append(line)
        return word_list

    def sentiment_analyzer_scores(self, sentence):
        score, negation_usage, word_and_emoticon_usage = self.analyzer.polarity_scores(sentence)
        self.negation_words = merge_dictionaries(self.negation_words, negation_usage)
        self.separate_emoticons_and_words(word_and_emoticon_usage)
        return sentence, score

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
        self.emoticons = merge_dictionaries(self.emoticons, emoticon_dict)
        self.words = merge_dictionaries(self.words, word_dict)

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def create_data_frames_with_result_columns(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
            csv_data = self.create_csv_dataframe(existing_csv_columns=data.columns)
            for index, row in data.iterrows():
                text = str(row['Text'])
                result = self.sentiment_analyzer_scores(text)
                self.update_csv_data(csv_data=csv_data, row=row, result=result)
            self.sentiment_dataframes.append(csv_data)

    @staticmethod
    def create_csv_dataframe(existing_csv_columns):
        csv_columns = list(existing_csv_columns)
        csv_columns.append('Neg')
        csv_columns.append('Neu')
        csv_columns.append('Pos')
        csv_columns.append('Compound')
        csv_data = pd.DataFrame(columns=csv_columns)
        return csv_data

    @staticmethod
    def update_csv_data(csv_data, row, result):
        new_csv_row = [row['User'], row['Day'], row['Month'], row['Year'], row['Title'], row['Text'],
                       row['Rating'],
                       row['App_ID'],
                       result[1]['neg'], result[1]['neu'], result[1]['pos'], result[1]['compound']]
        csv_data.loc[len(csv_data)] = new_csv_row

    def save_sentiment_csv_file(self, output_folder_path):
        count = 0
        for data_frame in self.sentiment_dataframes:
            head, tail = os.path.split(self.csv_files[count])
            data_frame.to_csv(output_folder_path + "\\" + tail[:-4] + "_including_sentiment_score.csv", index=None,
                              header=True)
            count += 1

    def sentiment_classification(self, compound):
        if compound >= 0.05:
            self.positive_review_count += 1
        elif compound <= -0.05:
            self.negative_review_count += 1
        else:
            self.neutral_review_count += 1

    def review_score_classification(self, score):
        self.star_count[score] += 1
