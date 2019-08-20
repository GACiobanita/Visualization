import unittest
from implementation.data_adjustments import DataAdjustment
from implementation.file_receiver import FileReceiver
import pandas as pd


class TestDataAdjustment(unittest.TestCase):

    def setUp(self):
        self.data_adjuster = DataAdjustment()
        self.file_receiver = FileReceiver()

    def test_removeStopWords_valid(self):
        result = self.data_adjuster.remove_string_stopwords(["apple", "an", "the"])
        self.assertEqual(1, len(result))
        print(result)

    def test_removeStopWords_invalid(self):
        result = self.data_adjuster.remove_string_stopwords(["an", "the"])
        self.assertEqual(1, len(result))
        print(result)

    def test_remove_string_punctuation_valid(self):
        test = "this string. has ! a bunch , of ? punctuation "
        test = self.data_adjuster.remove_string_punctuation(test)
        self.assertEqual(-1, test.find('.'))
        print(test)

    def test_remove_string_punctuation_invalid(self):
        test = "this string has no punctuation"
        test = self.data_adjuster.remove_string_punctuation(test)
        self.assertNotEqual(-1, test.find(','))
        print(test)

    def test_create_dict_from_tuple_valid(self):
        test = (("lol", 5), ("test", 4), ("trash", 3))
        result = self.data_adjuster.get_dict_from_tuple(test)
        self.assertNotEqual(0, len(result))

    def test_create_dict_from_tuple_invalid(self):
        test = (("lol", 5), ("test", 4), ("trash", 3))
        result = self.data_adjuster.get_dict_from_tuple(test)
        self.assertNotEqual(3, len(result))

    def test_remove_duplicate_rows_from_csv(self):
        self.file_receiver.acquire_input_path()
        self.data_adjuster.remove_duplicate_rows_from_csv(self.file_receiver.csv_files,
                                                          'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test')

    def test_get_tf_idf_scores_per_text(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2014\\fraud_apps_2014_no_anon_reviews.csv')
        result = self.data_adjuster.get_tf_idf_scores_per_text(pd_data, 'Text')
        # self.assertNotEqual(0, result.items())

    def test_get_tf_idf_from_entire_text(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2014\\fraud_apps_2013_no_anon_reviews.csv')
        self.data_adjuster.get_tf_idf_from_entire_text(pd_data, 'Text')

    def test_train_engine(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        result = self.data_adjuster.train_engine(pd_data['Text'], pd_data['Compound'])

    def test_create_tf_idf_matrix_from_texts(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        tf_idf_matrix = self.data_adjuster.create_tf_idf_matrix_from_texts(pd_data['Text'])

    def test_create_semantic_matrix_from_tf_idf(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_no_anon_reviews_including_sentiment_score.csv')
        tf_idf_matrix = self.data_adjuster.create_tf_idf_matrix_from_texts(pd_data['Text'])
        semantic_matrix = self.data_adjuster.create_semantic_matrix_from_tf_idf(tf_idf_matrix, pd_data['Compound'],
                                                                                pd_data['Text'])

    def test_create_k_means_clusters_from_semantic_matrix(self):
        pd_data = pd.read_csv(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2014\\sentiment\\fraud_apps_2014_no_anon_reviews_including_sentiment_score.csv')
        tf_idf_matrix = self.data_adjuster.create_tf_idf_matrix_from_texts(pd_data['Text'])
        semantic_matrix, semantic_dataframe = self.data_adjuster.create_semantic_matrix_from_tf_idf(tf_idf_matrix,
                                                                                                    pd_data['Compound'],
                                                                                                    pd_data['Text'])
        cluster_matrix = self.data_adjuster.create_k_means_clusters_from_semantic_matrix(semantic_matrix,
                                                                                         semantic_dataframe, 4)


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
