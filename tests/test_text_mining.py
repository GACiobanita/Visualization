import unittest
from implementation.text_mining import TextMining
import pandas as pd


class TestTextMining(unittest.TestCase):

    def setUp(self):
        self.text_miner = TextMining()

    def test_text_processing(self):
        result = self.text_miner.text_processing(
            "Keeps force closing! I would like to be able to go from list to grid view in album artist sections though and have a widget. I'll give a 5 then; but until then 3")

    def test_chunk_iterator(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        generator = self.text_miner.chunk_iterator(pd_data['Text'])

    def test_create_tf_idf_matrix_from_texts(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        tf_idf_matrix, tf_idf_feature_names = self.text_miner.create_tf_idf_matrix_from_texts(corpus)
        print(tf_idf_matrix.shape)

    def test_create_semantic_matrix_from_tf_idf(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        tf_idf_matrix, tf_idf_feature_names = self.text_miner.create_tf_idf_matrix_from_texts(corpus)
        semantic_matrix = self.text_miner.create_semantic_matrix_from_tf_idf(tf_idf_matrix)

    def test_create_term_count_features_from_text(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        text_length = len(pd_data['Text'])
        term_frequency, tf_idf_feature_names = self.text_miner.create_term_frequency_from_text(corpus, text_length)

    def test_create_non_negative_matrix_model_frobenius_beta_loss(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        tf_idf_matrix, tf_idf_feature_names = self.text_miner.create_tf_idf_matrix_from_texts(corpus)
        nmf = self.text_miner.create_non_negative_matrix_model(tf_idf_matrix, beta_loss='frobenius')

    def test_create_non_negative_matrix_model_kullback_leibler_beta_loss(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        tf_idf_matrix, tf_idf_feature_names = self.text_miner.create_tf_idf_matrix_from_texts(corpus)
        nmf = self.text_miner.create_non_negative_matrix_model(tf_idf_matrix, beta_loss='kullback-leibler')

    def test_print_top_words_nmf_input_frobenius_loss(self):
        topic_count = 10
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        tf_idf_matrix = self.text_miner.create_tf_idf_matrix_from_texts(corpus)
        text_length = len(pd_data['Text'])
        term_frequency, tf_feature_names = self.text_miner.create_term_frequency_from_text(corpus, text_length)
        nmf = self.text_miner.create_non_negative_matrix_model(tf_idf_matrix, beta_loss='frobenius')
        self.text_miner.print_top_words(nmf, tf_feature_names, topic_count)

    def test_print_top_words_nmf_input_kullback_leibler(self):
        topic_count = 10
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        tf_idf_matrix = self.text_miner.create_tf_idf_matrix_from_texts(corpus)
        text_length = len(pd_data['Text'])
        term_frequency, tf_feature_names = self.text_miner.create_term_frequency_from_text(corpus, text_length)
        nmf = self.text_miner.create_non_negative_matrix_model(tf_idf_matrix, beta_loss='kullback-leibler')
        self.text_miner.print_top_words(nmf, tf_feature_names, topic_count)

    def test_create_latent_dirichlet_allocation_model(self):
        topic_count = 5
        learning_decay = 0.9
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        tf_idf_matrix = self.text_miner.create_tf_idf_matrix_from_texts(corpus)
        text_length = len(pd_data['Text'])
        term_frequency, tf_feature_names = self.text_miner.create_term_frequency_from_text(corpus, text_length)
        lda = self.text_miner.create_latent_dirichlet_allocation_model(term_frequency, topic_count, learning_decay)

    def test_print_top_words_lda_input(self):
        topic_count = 5
        learning_decay = 0.5
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_all_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        text_length = len(pd_data['Text'])
        term_frequency, tf_feature_names = self.text_miner.create_term_frequency_from_text(corpus, text_length)
        lda = self.text_miner.create_latent_dirichlet_allocation_model(term_frequency, topic_count, learning_decay)
        self.text_miner.print_top_words(lda, tf_feature_names, 10)

    def test_search_for_best_lda_model(self):
        n_topics = [10, 15, 20, 25]
        learning_decay = [.5, .7, .9]

        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2016\\sentiment\\fraud_apps_2016_all_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        text_length = len(pd_data['Text'])
        print("Creating term freq.\n")
        term_frequency, tf_idf_feature_names = self.text_miner.create_term_frequency_from_text(corpus, text_length)
        print("Getting best model.\n")
        models = self.text_miner.search_for_best_lda_model(term_frequency, n_topics, learning_decay)
        best_lda_model = models.best_estimator_
        print("Getting topics.\n")
        dominant_topics = self.text_miner.get_dominant_topics_from_text(best_lda_model, term_frequency,
                                                                        len(pd_data['Text']),
                                                                        models.best_params_.get("n_components", 0))
        print("Getting top words.\n")
        topic_keywords = self.text_miner.get_top_keywords_per_topic(tf_idf_feature_names, best_lda_model.components_,
                                                                    15)
        print(topic_keywords.to_string())

    def test_get_dominant_topics_from_text(self):
        topic_count = 5
        learning_decay = 0.9
        # n_topics = [5]
        # learning_decay = [.9]

        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        text_length = len(pd_data['Text'])
        term_frequency, tf_feature_names = self.text_miner.create_term_frequency_from_text(corpus, text_length)
        lda = self.text_miner.create_latent_dirichlet_allocation_model(term_frequency, topic_count, learning_decay)
        df_result = self.text_miner.get_dominant_topics_from_text(lda, term_frequency, len(pd_data['Text']),
                                                                  topic_count)
        print(df_result.to_string())

    def test_get_top_key_words_per_topic(self):
        topic_count = 5
        learning_decay = 0.9

        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        text_length = len(pd_data['Text'])
        term_frequency, tf_feature_names = self.text_miner.create_term_frequency_from_text(corpus, text_length)
        lda = self.text_miner.create_latent_dirichlet_allocation_model(term_frequency, topic_count, learning_decay)
        result = self.text_miner.get_top_keywords_per_topic(tf_feature_names, lda.components_, 15)

    def test_get_cluster_data_based_on_text_topic_similarity(self):
        topic_count = 5
        learning_decay = 0.9

        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        corpus = self.text_miner.chunk_iterator(pd_data['Text'])
        text_length = len(pd_data['Text'])
        term_frequency, tf_feature_names = self.text_miner.create_term_frequency_from_text(corpus, text_length)
        lda = self.text_miner.create_latent_dirichlet_allocation_model(term_frequency, topic_count, learning_decay)
        lda_output = lda.transform(term_frequency)
        self.text_miner.get_cluster_data_based_on_topic_similarity(topic_count, lda_output)
