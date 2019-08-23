import unittest
import pandas as pd
from wordcloud import WordCloud
from implementation.word_cloud_generator import WordCloudGenerator
from implementation.file_receiver import FileReceiver


class TestWordCloud(unittest.TestCase):

    def setUp(self):
        self.wordcloud = WordCloudGenerator()
        self.file_receiver = FileReceiver()

    def test_display_wordcloud_valid(self):
        wc = WordCloud(background_color="white")
        wc.generate_from_text("this is a word cloud text lmao lmao lmao lmao lo lo lo lol lol lol lol")
        self.wordcloud.word_clouds.append(wc)
        self.wordcloud.display_word_cloud()

    def test_display_wordcloud_invalid(self):
        self.wordcloud.display_word_cloud()

    def test_create_dictionaries_valid(self):
        self.file_receiver.acquire_input_path()
        self.wordcloud.acquire_csv_files(self.file_receiver.csv_files)
        self.wordcloud.create_dictionaries()
        self.assertNotEqual(0, len(self.wordcloud.csv_files))
        self.assertNotEqual(0, len(self.wordcloud.word_frequency))

    def test_create_dictionaries_invalid(self):
        self.file_receiver.acquire_input_path()
        self.wordcloud.acquire_csv_files(self.file_receiver.csv_files)
        self.wordcloud.create_dictionaries()
        self.assertEqual(0, len(self.wordcloud.csv_files))
        self.assertEqual(0, len(self.wordcloud.word_frequency))

    def test_create_word_cloud_valid(self):
        test_data = {'Word': ["nice", "very nice"], 'Frequency': [100, 200]}
        test_frame = pd.DataFrame(test_data, columns=['Word', 'Frequency'])
        self.wordcloud.word_frequency.append(test_frame)
        self.wordcloud.create_word_cloud()
        self.assertNotEqual(0, len(self.wordcloud.word_clouds))
        self.wordcloud.display_word_cloud()

    def test_create_word_cloud_invalid(self):
        test_data = {'Word': ["nice", "very nice"], 'Frequency': [100, 200]}
        test_frame = pd.DataFrame(test_data, columns=['Word', 'Frequency'])
        self.wordcloud.word_frequency.append(test_frame)
        self.wordcloud.create_word_cloud()
        self.assertNotEqual(1, len(self.wordcloud.word_clouds))
        self.wordcloud.display_word_cloud()

    def test_save_word_cloud_valid(self):
        self.wordcloud.acquire_csv_files(
            ['D:\Google_Play_Fraud_Benign_Malware\Fraud\Test\fraud_apps_640_review_info_final_2012_top_10.csv'])
        self.wordcloud.create_dictionaries()
        self.wordcloud.create_word_cloud()
        self.assertNotEqual(0, len(self.wordcloud.word_clouds))
        self.wordcloud.save_word_cloud("D:\\Google_Play_Fraud_Benign_Malware\\Visualizations")

    def test_save_word_cloud_invalid(self):
        self.wordcloud.acquire_csv_files(
            ['D:\Google_Play_Fraud_Benign_Malware\Fraud\Test\fraud_apps_640_review_info_final_2012_top_10.csv'])
        self.wordcloud.create_dictionaries()
        self.wordcloud.create_word_cloud()
        self.assertNotEqual(0, len(self.wordcloud.word_clouds))
        self.wordcloud.save_word_cloud("D:\\Google_Play_Fraud_Benign_Malware\\Visualizations")

    def test_create_sentiment_dictionaries_valid(self):
        self.file_receiver.acquire_input_path()
        self.wordcloud.acquire_csv_files(self.file_receiver.csv_files)
        self.wordcloud.create_sentiment_dictionaries()
        self.assertNotEqual(0, len(self.wordcloud.sentiment_word_cloud_data))

    def test_create_sentiment_word_cloud_valid(self):
        self.wordcloud.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\Sentiment_word_cloud\\fraud_apps_640_review_info_final_2012_top_10_including_sentiment_score.csv'])
        self.wordcloud.create_sentiment_dictionaries()
        self.assertNotEqual(0, len(self.wordcloud.sentiment_word_cloud_data))
        self.wordcloud.create_sentiment_word_cloud()

    def test_save_sentiment_word_cloud_valid(self):
        self.wordcloud.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\Sentiment_word_cloud\\fraud_apps_640_review_info_final_2013_top_10_including_sentiment_score.csv'])
        self.wordcloud.create_sentiment_dictionaries()
        self.assertNotEqual(0, len(self.wordcloud.sentiment_word_cloud_data))
        self.wordcloud.create_sentiment_word_cloud()
        self.wordcloud.save_sentiment_clouds('D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\Sentiment_word_cloud')

    def test_calculate_individual_tf_idf_scores_of_text(self):
        self.wordcloud.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2015\\fraud_apps_2015_all_reviews.csv'
        ])
        self.wordcloud.calculate_individual_tf_idf_scores_of_text()



if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
