import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
import string


class WordCloudGenerator(object):

    def __init__(self):
        self.INPUT_ROOT_FOLDER = "C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage"
        self.OUTPUT_ROOT_FOLDER = "C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Visualizations"
        self.input_folder_path = self.INPUT_ROOT_FOLDER
        self.output_folder_path = self.OUTPUT_ROOT_FOLDER
        self.STOP_WORDS = set(stopwords.words("english"))
        self.csv_files = []
        self.word_clouds = []
        self.word_freqs = []

    def acquire_input_path(self):
        print("Default file path to input folder is: C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage")
        filepath = input("Input a new file path to a folder if you wish to change it: ")
        if os.path.exists(filepath) is True:
            print("New file path selected.")
            self.input_folder_path = filepath
            for r, d, f in os.walk(self.input_folder_path):
                for file in f:
                    self.csv_files.append(r + "\\" + file)
        else:
            print("Input path is not correct, default will be used: C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage")
            for r, d, f in os.walk(self.input_folder_path):
                for file in f:
                    self.csv_files.append(r + "\\" + file)

    def acquire_output_path(self):
        print("Default file path to output folder is: C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage")
        filepath = input("Input a new file path to a folder if you wish to change it: ")
        if os.path.exists(filepath) is True:
            print("New file path selected.")
            self.output_folder_path = filepath
        else:
            print("Output path is not correct, default will be used: C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Visualizations")

    def create_dictionaries(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
            container = data['Text'].str.lower().str.cat(sep=' ')
            container = self.remove_string_punctuation(container)
            words = word_tokenize(container)
            words = self.remove_string_stopwords(words)
            word_dist = FreqDist(words)
            results = pd.DataFrame(word_dist.most_common(100), columns=['Word', 'Frequency'])
            self.word_freqs.append(results)

    def create_wordcloud(self):
        for freq in self.word_freqs:
            subset = freq[['Word', 'Frequency']]
            tuples = [tuple(x) for x in subset.values]
            data = self.create_dict_from_tuple(tuples)
            wordcloud = WordCloud(background_color="white")
            wordcloud.generate_from_frequencies(frequencies=data)
            self.word_clouds.append(wordcloud)

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

    def display_word_cloud(self):
        for wordcloud in self.word_clouds:
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
        plt.show()

    def save_word_cloud(self):
        count = 0
        for wordcloud in self.word_clouds:
            head, tail = os.path.split(self.csv_files[count])
            wordcloud.to_file(self.output_folder_path + "\\" + tail[:-4] + ".png")
            count += 1