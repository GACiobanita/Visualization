import unittest
from implementation.sentiment_analyzer import SentimentAnalyzer
from implementation.file_receiver import FileReceiver


class Test_SentimentAnalyzer(unittest.TestCase):

    def setUp(self):
        self.file_receiver = FileReceiver()
        self.sentiment_analyzer = SentimentAnalyzer()

    def test_sentiment_analyzer_scores_valid(self):
        self.assertNotEqual({'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0},
                            self.sentiment_analyzer.calculate_text_score_and_word_appearance(
                                "I ordered some things but they were to small I asked if I could send them back for size changes. They told me I would have to repay for shipping to and from...WTF like whyyyy thoooo I only wanted different sizes not even a refund. I asked if it was a number where I could call to speak with someone on a direct line..Of course they didn't say anything back & claims to have refunded my money which was a LIE!!! I will not and advice others to not order from them bcuz their services lack terribly")[
                                1])
        self.assertNotEqual({'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0},
                            self.sentiment_analyzer.calculate_text_score_and_word_appearance(
                                "After installing the application i realized i wasn't really using it much. So i requested the Crownit team to delete my account as i don't want my data (Phone number etc.) to remain with them. I got a reply saying i can unistall the app if i wanted to but they won't delete my account. I mean WTF?")[
                                1])
        self.assertNotEqual({'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0},
                            self.sentiment_analyzer.calculate_text_score_and_word_appearance(
                                "Very nice app to use in my new android. Love to use it. :)")[
                                1])
        self.assertNotEqual({'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0},
                            self.sentiment_analyzer.calculate_text_score_and_word_appearance(
                                "XD :)")[
                                1])

    def test_sentiment_analyzer_scores_invalid(self):
        self.assertNotEqual({'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0},
                            self.sentiment_analyzer.calculate_text_score_and_word_appearance("")[1])

    def test_calculate_scores_for_reviews_valid(self):
        self.sentiment_analyzer.acquire_csv_files(
            ['D:\\Google_Play_Fraud_Benign_Malware\\Fraud\Test\\fraud_apps_640_review_info_final_2012_top_10.csv'])
        self.sentiment_analyzer.create_data_frames_with_result_columns()
        self.sentiment_analyzer.save_sentiment_csv_file('D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test')
        self.assertNotEqual(0, len(self.sentiment_analyzer.sentiment_data_frames))

    # missing data row
    def test_calculate_scores_for_reviews_invalid(self):
        self.sentiment_analyzer.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\Test\\fraud_apps_640_review_info_final_2014_top_10_no_title_column.csv'])
        self.sentiment_analyzer.create_data_frames_with_result_columns()
        self.sentiment_analyzer.save_sentiment_csv_file('D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test')
        self.assertNotEqual(0, len(self.sentiment_analyzer.sentiment_data_frames))


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
