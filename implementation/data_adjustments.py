import string
import pandas as pd
import os
import calendar
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from inspect import getsourcefile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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

    @staticmethod
    def get_tf_idf_scores_per_text(pd_data, data_column):
        tf_idf = TfidfVectorizer(decode_error='ignore', encoding='string', stop_words='english', lowercase=True,
                                 preprocessor='callable', tokenizer='callable')
        data = tf_idf.fit_transform(pd_data[data_column].values.astype(str))

        previous_df = pd.DataFrame(data[0].T.todense(), index=tf_idf.get_feature_names(), columns=['tf_idf'])
        previous_df.index.name = 'word'
        previous_df = previous_df[previous_df.tf_idf != 0]
        previous_df = previous_df.sort_values(by=['tf_idf'], ascending=False)

        for item in data[1:]:
            df = pd.DataFrame(item.T.todense(), index=tf_idf.get_feature_names(), columns=['tf_idf'])
            df = df[df.tf_idf != 0]
            df = df.sort_values(by=['tf_idf'], ascending=False)
            result = pd.concat([previous_df, df])
            previous_df = result

        result = result.rename_axis('word').sort_values(by=['tf_idf', 'word'], ascending=[False, True])
        result = result.groupby(result.index).first()
        result.to_csv('D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\test_tf_idf_per_review.csv',
                      header=True)
        print(result)

    @staticmethod
    def get_tf_idf_from_entire_text(pd_data, data_column):
        cv = CountVectorizer(decode_error='ignore', encoding='string', stop_words='english', lowercase=True)

        all_text_data = pd_data[data_column].str.cat(sep='. ')

        word_count_vector = cv.fit_transform([all_text_data])

        feature_names = cv.get_feature_names()

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        tf_idf_vector = tfidf_transformer.transform(word_count_vector)

        for data in tf_idf_vector:
            first_document_vector = data
            df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=['tf_idf'])
            df = df[df.tf_idf != 0]
            df = df.sort_values(by=['tf_idf'], ascending=False)

            print(df)

        df = df.rename_axis('word').sort_values(by=['word'], ascending=[True])
        df = df.sort_values(by=['tf_idf'], ascending=False)
        df.to_csv('D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\test_tf_idf_entire_text.csv',
                  header=True)

    @staticmethod
    def create_tf_idf_matrix_from_texts(plots_text):
        # create vectorizer for tf-idf matrix creation
        vectorizer = TfidfVectorizer(lowercase=True, min_df=3, max_df=0.9, stop_words='english')
        # fit and transform the plots, creating the tf-idf matrix
        # a tf-idf score reflects how import a word is in a collection of texts
        tf_idf_matrix = vectorizer.fit_transform(plots_text.values.astype(str))
        tf_idf_dataframe = pd.DataFrame(tf_idf_matrix.toarray(), index=plots_text.values.astype(str),
                                        columns=vectorizer.get_feature_names())
        return tf_idf_matrix

    @staticmethod
    def create_semantic_matrix_from_tf_idf(tf_idf_matrix, plots_compound, plots_text):
        # create the TruncatedSVD object for dimensionality reduction of the tf-idf matrix
        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        # pass a normalizer and the svd object to the latent semantic analysis object
        lsa = make_pipeline(svd, normalizer)
        # normalize the tf-idf matrix and reduce it
        normalized_matrix = lsa.fit_transform(tf_idf_matrix)
        xs = [w[0] for w in normalized_matrix]
        ys = [w[1] for w in normalized_matrix]
        data = {'x pos': xs, 'y pos': ys, 'compound': plots_compound.values.astype(float)}
        normalized_dataframe = pd.DataFrame(data)
        return normalized_matrix, normalized_dataframe

    def create_k_means_clusters_from_semantic_matrix(self, normalized_matrix, normalized_dataframe, cluster_count):
        km = KMeans(n_clusters=cluster_count, init='k-means++', n_init=10, verbose=0)
        cluster_matrix = km.fit_transform(normalized_matrix)
        positive_points, neutral_points, negative_points = self.compound_separation_from_data(normalized_dataframe)
        for point in positive_points:
            plt.plot(point[0], point[1], 'k.', color='g', markersize=2)
        for point in negative_points:
            plt.plot(point[0], point[1], 'k.', color='r', markersize=2)
        for point in neutral_points:
            plt.plot(point[0], point[1], 'k.', color='k', markersize=2)
        centroids = km.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, zorder=10)
        plt.title('K-means clustering on the sentiment dataset')
        plt.show()

    @staticmethod
    def compound_separation_from_data(normalized_dataframe):
        positive_points = []
        negative_points = []
        neutral_points = []
        for index, row in normalized_dataframe.iterrows():
            if row['compound'] >= 0.05:
                positive_points.append((row['x pos'], row['y pos']))
            elif row['compound'] <= -0.05:
                negative_points.append((row['x pos'], row['y pos']))
            else:
                neutral_points.append((row['x pos'], row['y pos']))
        return positive_points, neutral_points, negative_points

    @staticmethod
    def train_engine(plots_text, plots_compound):
        # create vectorizer for tf-idf matrix creation
        vectorizer = TfidfVectorizer(lowercase=True, min_df=3, max_df=0.9, stop_words='english')
        # fit and transform the plots, creating the tf-idf matrix
        # a tf-idf score reflects how import a word is in a collection of texts
        tf_idf_matrix = vectorizer.fit_transform(plots_text.values.astype(str))
        print(tf_idf_matrix.shape)
        tf_idf_dataframe = pd.DataFrame(tf_idf_matrix.toarray(), index=plots_text.values.astype(str),
                                        columns=vectorizer.get_feature_names())
        # create the TruncatedSVD object for dimensionality reduction of the tf-idf matrix
        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        # pass a normalizer and the svd object to the latent semantic analysis object
        lsa = make_pipeline(svd, normalizer)
        # normalize the tf-idf matrix and reduce it
        normalized_matrix = lsa.fit_transform(tf_idf_matrix)
        print(normalized_matrix.shape)

        similarity = np.asarray(np.asmatrix(normalized_matrix) * np.asmatrix(normalized_matrix).T)
        sim_df = pd.DataFrame(similarity, index=plots_text, columns=plots_text)

        xs = [w[0] for w in normalized_matrix]
        ys = [w[1] for w in normalized_matrix]

        plt.figure()
        plt.scatter(xs, ys)
        plt.title('Plot of points against LSA principal components')
        # plt.show()

        # do K means clustering, where reviews with similar weighted features will be near each other
        # clusters should represent a topic
        # specific topics can be figured out by looking at the words that are most heavily weighted
        inertia = []
        K = range(1, 15)
        for k in K:
            km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, verbose=0)
            cluster_matrix = km.fit_transform(normalized_matrix)
            inertia.append(km.inertia_)

        plt.plot(K, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum of Square Distances')
        plt.title('Elbow Method for optimal k')
        plt.show()

        xs = [w[0] for w in cluster_matrix]
        ys = [w[1] for w in cluster_matrix]

        plt.figure()
        plt.scatter(xs, ys)
        plt.title('Plot of points in clusters')
        # plt.show()
