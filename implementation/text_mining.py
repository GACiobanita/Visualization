from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from .data_adjustments import DataAdjustment
import pandas as pd
import re
import gensim
import numpy as np
import matplotlib.pyplot as plt


class TextMining(object):

    def __init__(self):
        self.data_adjuster = DataAdjustment()

    def chunk_iterator(self, plots_text):
        for document in plots_text.values.astype('U'):
            document = self.text_processing(document)
            yield document

    def text_processing(self, sentences):
        short_words = re.compile(r'\W*\b\w{1,3}\b')
        long_words = re.compile(r'\W*\b\w{15,}\b')

        word_data = list(self.sentence_to_words(sentences))
        lemmatizer = WordNetLemmatizer()
        lemmatized_word_data = []
        for words in word_data:
            for word in words:
                lemmatized_word_data.append(lemmatizer.lemmatize(word, self.get_wordnet_pos(word)))
        lemmatized_word_data = " ".join(lemmatized_word_data)
        lemmatized_word_data = short_words.sub('', lemmatized_word_data)
        lemmatized_word_data = long_words.sub('', lemmatized_word_data)
        return lemmatized_word_data

    @staticmethod
    def get_wordnet_pos(word):
        # map pos tag to first character lemmatize() accepts
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def sentence_to_words(sentence):
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    @staticmethod
    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

    def create_tf_idf_matrix_from_texts(self, corpus):
        # create vectorizer for tf-idf matrix creation
        vectorizer = TfidfVectorizer(lowercase=True, min_df=3, max_df=0.7, analyzer='word')
        # fit and transform the plots, creating the tf-idf matrix
        # a tf-idf score reflects how import a word is in a collection of texts
        tf_idf_matrix = vectorizer.fit_transform(corpus)
        return tf_idf_matrix

    def create_term_frequency_from_text(self, corpus):
        vectorizer = CountVectorizer(lowercase=True, min_df=3, max_df=0.7, stop_words='english')
        tf = vectorizer.fit_transform(corpus)
        tf_feature_names = vectorizer.get_feature_names()
        return tf, tf_feature_names

    @staticmethod
    def create_non_negative_matrix_model(tf_idf_matrix, beta_loss):
        nmf = NMF(n_components=10, random_state=1, beta_loss=beta_loss, solver='mu', max_iter=1000, alpha=.1,
                  l1_ratio=.5).fit(tf_idf_matrix)
        return nmf

    @staticmethod
    def create_latent_dirichlet_allocation_model(term_frequency, component_count, learning_decay):
        lda = LatentDirichletAllocation(n_components=component_count, max_iter=5, learning_method='online',
                                        learning_offset=50.,
                                        random_state=0, learning_decay=learning_decay)
        lda.fit(term_frequency)
        return lda

    @staticmethod
    def create_semantic_matrix_from_tf_idf(tf_idf_matrix):
        # create the TruncatedSVD object for dimensionality reduction of the tf-idf matrix
        svd = TruncatedSVD(100)
        normalizer = Normalizer(copy=False)
        # pass a normalizer and the svd object to the latent semantic analysis object
        lsa = make_pipeline(svd, normalizer)
        # normalize the tf-idf matrix and reduce it
        normalized_matrix = lsa.fit_transform(tf_idf_matrix)
        return normalized_matrix

    @staticmethod
    def search_for_best_lda_model(data_vectorized, n_topics, learning_decay):

        search_param = {'n_components': n_topics, 'learning_decay': learning_decay}

        lda = LatentDirichletAllocation()

        model = GridSearchCV(lda, param_grid=search_param, cv=3, iid=True)

        model.fit(data_vectorized)

        return model

    @staticmethod
    def search_for_best_nmf_model(tf_idf_matrix, beta_loss, n_topics):

        search_param = {'beta_loss': beta_loss, 'n_components': n_topics}

        nmf = NMF()

        model = GridSearchCV(nmf, param_grid=search_param, cv=3, iid=True)

        model.fit(tf_idf_matrix)

        return model

    @staticmethod
    def get_dominant_topics_from_text(lda, term_freq, doc_length, n_topics):
        lda_output = lda.transform(term_freq)

        topic_names = ["Topic" + str(i + 1) for i in range(n_topics)]

        doc_names = ["Doc" + str(i + 1) for i in range(doc_length)]
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topic_names, index=doc_names)

        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        df_document_topic['dominant_topic'] = dominant_topic

        return df_document_topic

    @staticmethod
    def get_topic_distribution_across_texts(df_topic):
        df_topic_distributions = df_topic['dominant_topic'].value_counts().reset_index(name="Num Texts")
        df_topic_distributions.columns = ['Topic Num', 'Num Texts']
        return df_topic_distributions

    @staticmethod
    def get_topic_keywords(model_components, feature_names, n_topics):
        df_topic_keywords = pd.DataFrame(model_components)
        df_topic_keywords.columns = feature_names
        df_topic_keywords.index = n_topics

        return df_topic_keywords

    @staticmethod
    def get_top_keywords_per_topic(feature_names, lda_components, n_words=15):
        keywords = np.array(feature_names)
        topic_keywords = []
        for topic_weights in lda_components:
            top_keyword = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword))
        topic_keywords_df = pd.DataFrame(topic_keywords)
        topic_keywords_df.columns = ['Word' + str(i) for i in range(topic_keywords_df.shape[1])]
        topic_keywords_df.index = ['Topic' + str(i) for i in range(topic_keywords_df.shape[0])]
        return topic_keywords_df

    @staticmethod
    def get_cluster_data_based_on_topic_similarity(model_topic_count, lda_output):
        # create clusters from the document-topic matrix
        clusters = KMeans(n_clusters=model_topic_count, random_state=100).fit_predict(lda_output)

        # perform linear dim. reduction on lda_output
        svd_model = TruncatedSVD(n_components=2)  # 2 components from lda_output
        lda_output_svd = svd_model.fit_transform(lda_output)

        x = lda_output_svd[:, 0]
        y = lda_output_svd[:, 1]

        # plt.figure(figsize=(12, 12))
        # plt.scatter(x, y, c=clusters)
        # plt.ylabel('Y')
        # plt.xlabel('X')
        # plt.title("Segregation of Topic Clusters", )
        # plt.show()

        return x, y, clusters
