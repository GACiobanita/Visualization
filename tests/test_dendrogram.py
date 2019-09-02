import unittest
from implementation.data_adjustments import DataAdjustment
from implementation.file_receiver import FileReceiver
from implementation.dendrogram_generator import DendrogramGenerator
import pandas as pd
import matplotlib.pyplot as plt


class Test_DendrogramGenerator(unittest.TestCase):

    def setUp(self):
        self.dendrogram_generator = DendrogramGenerator()
        self.file_receiver = FileReceiver()

    def test_construct_topics_from_files_valid_input(self):
        csv_data = pd.read_csv(
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012\\Keywords\\topic_keywords_fraud_apps_2012_all_reviews_including_sentiment_score.csv")
        topics = self.dendrogram_generator.construct_topics_from_file(csv_data)
        self.assertNotEqual(0, len(topics))
        print(topics)

    # csv that does not contain topics
    def test_construct_topics_from_files_invalid_input(self):
        csv_data = pd.read_csv(
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012\\Dominant Topics\\dominant_topics_fraud_apps_2012_all_reviews_including_sentiment_score.csv")
        topics = self.dendrogram_generator.construct_topics_from_file(csv_data)
        self.assertNotEqual(0, len(topics))
        print(topics)

    def test_construct_topics_vocabulary_from_file(self):
        csv_data = pd.read_csv(
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012\\Keywords\\topic_keywords_fraud_apps_2012_all_reviews_including_sentiment_score.csv")
        topics = self.dendrogram_generator.construct_topics_from_file(csv_data)
        topic_dicts = self.dendrogram_generator.construct_topic_vocabulary_from_file(csv_data, topics)

    def test_construct_topics_vocabulary_from_file_invalid_input(self):
        csv_data = pd.read_csv(
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012\\Keywords\\topic_keywords_fraud_apps_2012_all_reviews_including_sentiment_score.csv")
        topics = self.dendrogram_generator.construct_topics_from_file(csv_data)
        csv_data = None
        self.dendrogram_generator.construct_topic_vocabulary_from_file(csv_data, topics)

    def test_construct_tree_structure(self):
        csv_data = pd.read_csv(
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012\\Keywords\\topic_keywords_fraud_apps_2012_all_reviews_including_sentiment_score.csv")
        topics = self.dendrogram_generator.construct_topics_from_file(csv_data)
        topic_dicts = self.dendrogram_generator.construct_topic_vocabulary_from_file(csv_data, topics)
        tree_structure = self.dendrogram_generator.construct_tree_structure(topic_dicts)
        self.assertNotEqual(0, len(tree_structure))
        print(tree_structure)
        print(type(tree_structure))

    def test_construct_tree_structure_invalid_input(self):
        csv_data = pd.read_csv(
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012\\Keywords\\topic_keywords_fraud_apps_2012_all_reviews_including_sentiment_score.csv")
        topics = self.dendrogram_generator.construct_topics_from_file(csv_data)
        topic_dicts = self.dendrogram_generator.construct_topic_vocabulary_from_file(csv_data, topics)
        topic_dicts['Music'] = 0
        tree_structure = self.dendrogram_generator.construct_tree_structure(topic_dicts)
        self.assertNotEqual(0, len(tree_structure))

    def test_construct_linkage_matrix(self):
        csv_data = pd.read_csv(
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012\\Keywords\\topic_keywords_fraud_apps_2012_all_reviews_including_sentiment_score.csv")
        topics = self.dendrogram_generator.construct_topics_from_file(csv_data)
        topic_dicts = self.dendrogram_generator.construct_topic_vocabulary_from_file(csv_data, topics)
        tree_structure = self.dendrogram_generator.construct_tree_structure(topic_dicts)
        linkage_matrix = None
        leaf_names = None
        linkage_matrix, leaf_names = self.dendrogram_generator.construct_linkage_matrix(tree_structure)
        self.assertNotEqual(None, linkage_matrix)
        self.assertNotEqual(0, len(leaf_names))

    def test_construct_dendrogram(self):
        csv_data = pd.read_csv(
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2016\\Keywords\\topic_keywords_fraud_apps_2016_top_10_no_anon_reviews_including_sentiment_score.csv")
        topics = self.dendrogram_generator.construct_topics_from_file(csv_data)
        topic_dicts = self.dendrogram_generator.construct_topic_vocabulary_from_file(csv_data, topics)
        tree_structure = self.dendrogram_generator.construct_tree_structure(topic_dicts)
        linkage_matrix = None
        leaf_names = None
        linkage_matrix, leaf_names = self.dendrogram_generator.construct_linkage_matrix(tree_structure)
        dendrogram = self.dendrogram_generator.construct_dendrogram(linkage_matrix, leaf_names,
                                                                    tree_structure['Topics'])
        self.dendrogram_generator.display_dendrogram()
