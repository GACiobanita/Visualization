import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os
from .data_adjustments import DataAdjustment


class WordCloudGenerator(object):

    def __init__(self):
        self.word_clouds = []
        self.word_frequency = []
        self.csv_files = []
        self.data_adjuster = DataAdjustment()

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def create_dictionaries(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
            string_container = data['Text'].str.lower().str.cat(sep=' ')
            string_container = self.data_adjuster.remove_string_punctuation(string_container)
            tokenized_words = self.data_adjuster.tokenize_words(string_container)
            words = self.data_adjuster.remove_string_stopwords(tokenized_words)
            word_frequency = self.data_adjuster.get_string_frequency_distribution()
            results = pd.DataFrame(word_frequency.most_common(100), columns=['Word', 'Frequency'])
            self.word_frequency.append(results)

    def create_word_cloud(self):
        for freq in self.word_frequency:
            subset = freq[['Word', 'Frequency']] #get subset of data
            tuples = [tuple(x) for x in subset.values] #transform data into tuples
            data = self.data_adjuster.get_dict_from_tuple(tuples) #so we can transform it into a dictionary
            word_cloud = WordCloud(background_color="white")
            word_cloud.generate_from_frequencies(frequencies=data)
            self.word_clouds.append(word_cloud)

    def display_word_cloud(self):
        for word_cloud in self.word_clouds:
            plt.figure()
            plt.imshow(word_cloud, interpolation="bilinear")
            plt.axis("off")
        plt.show()

    def save_word_cloud(self, output_folder_path):
        count = 0
        for word_cloud in self.word_clouds:
            head, tail = os.path.split(self.csv_files[count])
            word_cloud.to_file(output_folder_path + "\\" + tail[:-4] + "_word_cloud.png")
            count += 1
