from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from implementation.data_adjustments import DataAdjustment
import pandas as pd
import os


# interface to access and modify the VADER sentiment analyzer
class SentimentAnalyzer(object):

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.csv_files = []

        self.data_adjuster = DataAdjustment()

        self.sentiment_data_frames = []

    # modified the SentimentIntensityAnalyzer to return all words used and emoticons alongside the initial
    # score return
    def calculate_text_score_and_word_appearance(self, sentence):
        score, negation_usage, word_and_emoticon_usage = self.analyzer.polarity_scores(sentence)
        return sentence, score, negation_usage, word_and_emoticon_usage

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    # used to create csv files with Compound , Positive, Negative and Neutral columns
    # containing scores from the VADER analyzer
    def create_data_frames_with_result_columns(self):
        for file in self.csv_files:
            print("Started file " + str(file))
            data = pd.read_csv(file)
            csv_data = self.create_csv_data_frame(existing_csv_columns=data.columns)
            for index, row in data.iterrows():
                text = str(row['Text'])
                result = self.calculate_text_score_and_word_appearance(text)
                self.update_csv_data(csv_data=csv_data, row=row, result=result)
            self.sentiment_data_frames.append(csv_data)
            print("Completed for file " + str(file))

    @staticmethod
    def create_csv_data_frame(existing_csv_columns):
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
        for data_frame in self.sentiment_data_frames:
            head, tail = os.path.split(self.csv_files[count])
            data_frame.to_csv(output_folder_path + "\\" + tail[:-4] + "_including_sentiment_score.csv", index=None,
                              header=True)
            count += 1

    @staticmethod
    def largest_sentiment_value(positive, negative, neutral):
        if (positive >= negative) and (positive >= neutral):
            largest = positive
        elif (negative >= positive) and (negative >= neutral):
            largest = negative
        else:
            largest = neutral
        return largest
