import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os
from .data_adjustments import DataAdjustment
from .sentiment_analyzer import SentimentAnalyzer

DATA_ADJUSTER = DataAdjustment()


class WordCloudSentiment(object):

    def __init__(self):
        self.positive_word_frequency = {}
        self.negative_word_frequency = {}
        self.positive_emoticon_frequency = {}
        self.negative_emoticon_frequency = {}

    def word_classification(self, word_list):
        for word in word_list:
            valence = DATA_ADJUSTER.get_valence_from_lexicon(word)
            if valence > 0:
                self.positive_word_frequency[word] = self.positive_word_frequency.get(word, 1) + 1
            else:
                self.negative_word_frequency[word] = self.negative_word_frequency.get(word, 1) + 1

    def emoticon_classification(self, emoticon_list):
        for emoticon in emoticon_list:
            valence = DATA_ADJUSTER.get_valence_from_lexicon(emoticon)
            if valence > 0:
                self.positive_emoticon_frequency[emoticon] = self.positive_emoticon_frequency.get(emoticon, 1) + 1
            else:
                self.negative_emoticon_frequency[emoticon] = self.negative_emoticon_frequency.get(emoticon, 1) + 1

    def __str__(self):
        return "Positive:\n" + str(self.positive_word_frequency) + "\n " + "Negative:\n" + str(
            self.negative_word_frequency) + "\n "


class WordCloudGenerator(object):

    def __init__(self):
        self.word_clouds = []
        self.word_frequency = []
        self.csv_files = []
        self.sentiment_analyzer = SentimentAnalyzer()

        self.sentiment_word_cloud_data = {}
        self.positive_word_clouds = {}
        self.negative_word_clouds = {}
        self.positive_emoticon_clouds = {}
        self.negative_emoticon_clouds = {}

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def create_dictionaries(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
            string_container = data['Text'].str.lower().str.cat(sep=' ')
            string_container = DATA_ADJUSTER.remove_string_punctuation(string_container)
            tokenized_words = DATA_ADJUSTER.tokenize_words(string_container)
            words = DATA_ADJUSTER.remove_string_stopwords(tokenized_words)
            word_frequency = DATA_ADJUSTER.get_string_frequency_distribution(words)
            results = pd.DataFrame(word_frequency.most_common(100), columns=['Word', 'Frequency'])
            self.word_frequency.append(results)

    def create_sentiment_dictionaries(self):
        for file in self.csv_files:
            csv_data = pd.read_csv(file)
            year = DATA_ADJUSTER.get_year_from_string(file)
            sentiment_word_cloud = WordCloudSentiment()
            for index, row in csv_data.iterrows():
                text, score, negation_usage, word_and_emoticon_usage = self.sentiment_analyzer.calculate_text_score_and_word_appearance(
                    str(row['Text']))
                separated_emoticon_usage, separated_word_usage = DATA_ADJUSTER.separate_emoticons_and_words(
                    word_and_emoticon_usage)
                sentiment_word_cloud.word_classification(separated_word_usage)
                sentiment_word_cloud.emoticon_classification(separated_emoticon_usage)
            self.sentiment_word_cloud_data[year] = sentiment_word_cloud

    def calculate_individual_tf_idf_scores_of_text(self):
        for file in self.csv_files:
            csv_data = pd.read_csv(file)
            year = DATA_ADJUSTER.get_year_from_string(file)
            result = DATA_ADJUSTER.get_tf_idf_scores_per_text(csv_data, 'Text')
            print(result)

    def create_word_cloud(self):
        for freq in self.word_frequency:
            subset = freq[['Word', 'Frequency']]  # get subset of data
            tuples = [tuple(x) for x in subset.values]  # transform data into tuples
            data = DATA_ADJUSTER.get_dict_from_tuple(tuples)  # so we can transform it into a dictionary
            word_cloud = WordCloud(width=800, height=400, background_color="white")
            word_cloud.generate_from_frequencies(frequencies=data)
            self.word_clouds.append(word_cloud)

    def create_sentiment_word_cloud(self):
        for year, cloud_sentiment in self.sentiment_word_cloud_data.items():
            self.positive_word_clouds[year] = self.create_individual_sentiment_cloud(
                cloud_sentiment.positive_word_frequency)
            self.negative_word_clouds[year] = self.create_individual_sentiment_cloud(
                cloud_sentiment.negative_word_frequency)

            self.positive_emoticon_clouds[year] = self.create_individual_sentiment_cloud(
                cloud_sentiment.positive_emoticon_frequency)
            self.negative_emoticon_clouds[year] = self.create_individual_sentiment_cloud(
                cloud_sentiment.negative_emoticon_frequency)

    @staticmethod
    def create_individual_sentiment_cloud(token_freq):
        if len(token_freq) > 0:
            token_cloud = WordCloud(width=800, height=400, background_color="white")
            token_cloud.generate_from_frequencies(frequencies=token_freq)
            return token_cloud
        else:
            return None

    def display_word_cloud(self):
        for word_cloud in self.word_clouds:
            plt.figure()
            plt.imshow(word_cloud, interpolation="bilinear")
            plt.axis("off")
        plt.show()

    def save_sentiment_clouds(self, output_folder_path):
        count = 0
        for year, word_cloud in self.positive_word_clouds.items():
            head, tail = os.path.split(self.csv_files[count])
            word_cloud.to_file(output_folder_path + "\\" + tail[:-4] + "_positive_word_cloud.png")
            count += 1
        count = 0
        for year, word_cloud in self.negative_word_clouds.items():
            head, tail = os.path.split(self.csv_files[count])
            word_cloud.to_file(output_folder_path + "\\" + tail[:-4] + "_negative_word_cloud.png")
            count += 1

        count = 0
        for year, emoticon_cloud in self.positive_emoticon_clouds.items():
            head, tail = os.path.split(self.csv_files[count])
            emoticon_cloud.to_file(output_folder_path + "\\" + tail[:-4] + "_positive_emoticon_cloud.png")
            count += 1
        count = 0
        for year, emoticon_cloud in self.negative_emoticon_clouds.items():
            head, tail = os.path.split(self.csv_files[count])
            emoticon_cloud.to_file(output_folder_path + "\\" + tail[:-4] + "_negative_emoticon_cloud.png")
            count += 1

    def save_word_cloud(self, output_folder_path):
        count = 0
        for word_cloud in self.word_clouds:
            head, tail = os.path.split(self.csv_files[count])
            word_cloud.to_file(output_folder_path + "\\" + tail[:-4] + "_word_cloud.png")
            count += 1
