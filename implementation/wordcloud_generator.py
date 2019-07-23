import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import os
from .data_adjustments import DataAdjustment


class WordCloudGenerator(object):

    def __init__(self):
        self.word_clouds = []
        self.word_freqs = []
        self.csv_files = []
        self.data_adjuster = DataAdjustment()

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def create_dictionaries(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
            container = data['Text'].str.lower().str.cat(sep=' ')
            container = self.data_adjuster.remove_string_punctuation(container)
            words = word_tokenize(container)
            words = self.data_adjuster.remove_string_stopwords(words)
            word_dist = FreqDist(words)
            results = pd.DataFrame(word_dist.most_common(100), columns=['Word', 'Frequency'])
            self.word_freqs.append(results)

    def create_wordcloud(self):
        for freq in self.word_freqs:
            subset = freq[['Word', 'Frequency']]
            tuples = [tuple(x) for x in subset.values]
            data = self.data_adjuster.create_dict_from_tuple(tuples)
            wordcloud = WordCloud(background_color="white")
            wordcloud.generate_from_frequencies(frequencies=data)
            self.word_clouds.append(wordcloud)

    def display_word_cloud(self):
        for wordcloud in self.word_clouds:
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
        plt.show()

    def save_word_cloud(self, output_folder_path):
        count = 0
        for wordcloud in self.word_clouds:
            head, tail = os.path.split(self.csv_files[count])
            wordcloud.to_file(output_folder_path + "\\" + tail[:-4] + "_word_cloud.png")
            count += 1
