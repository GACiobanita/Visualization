import unittest
from implementation.bar_chart_generator import BarChartGenerator
from implementation.file_receiver import FileReceiver


class TestBarChart(unittest.TestCase):

    def setUp(self):
        self.barchart_generator = BarChartGenerator()
        self.file_receiver = FileReceiver()

    def test_calculate_text_length_valid(self):
        self.file_receiver.acquire_input_path()
        self.barchart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.barchart_generator.categorize_text_by_word_count()
        self.assertNotEqual(0, len(self.barchart_generator.csv_files))

    def test_calculate_text_length_invalid(self):
        self.file_receiver.acquire_input_path()
        self.barchart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.barchart_generator.categorize_text_by_word_count()
        self.assertEqual(0, len(self.barchart_generator.csv_files))

    def test_create_horizontal_bar_chart_valid(self):
        self.barchart_generator.total_word_count = {'0-50': 1, '51-100': 9, '101-200': 11, '201-300': 25, '301-400': 60,
                                                    '400+': 5}
        self.barchart_generator.per_file_word_count = (
            ('2012', {'0-50': 1, '51-100': 9, '101-200': 11, '201-300': 25, '301-400': 60, '400+': 5}),
            ('2013', {'0-50': 1, '51-100': 9, '101-200': 11, '201-300': 25, '301-400': 60, '400+': 5}),
            ('2014', {'0-50': 1, '51-100': 9, '101-200': 11, '201-300': 25, '301-400': 60, '400+': 5}))
        self.barchart_generator.create_overall_bar_charts()
        self.barchart_generator.display_bar_charts()

    def test_create_horizontal_bar_chart_invalid(self):
        self.barchart_generator.total_word_count = {'0-50': "invalid", '51-100': 9, '101-200': 11, '201-300': 25,
                                                    '301-400': 60,
                                                    '400+': 5}
        self.barchart_generator.per_file_word_count = (
            ('2012', {'0-50': 1, '51-100': 9, '101-200': 11, '201-300': 25, '301-400': 60, '400+': 5}),
            ('2013', {'0-50': 1, '51-100': 9, '101-200': 11, '201-300': 25, '301-400': 60, '400+': 5}),
            ('2014', {'0-50': 1, '51-100': 9, '101-200': 11, '201-300': 25, '301-400': 60, '400+': 5}))
        self.barchart_generator.create_overall_bar_charts()
        self.barchart_generator.display_bar_charts()

    def test_save_bar_charts_valid(self):
        self.file_receiver.acquire_input_path()
        self.barchart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.barchart_generator.categorize_text_by_word_count()
        self.barchart_generator.create_overall_bar_charts()
        self.barchart_generator.display_bar_charts()
        self.barchart_generator.save_overall_bar_charts("D:\\Google_Play_Fraud_Benign_Malware\\Visualizations")

    def test_save_bar_charts_invalid(self):
        self.file_receiver.acquire_input_path()
        self.barchart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.barchart_generator.categorize_text_by_word_count()
        self.barchart_generator.create_overall_bar_charts()
        self.barchart_generator.display_bar_charts()
        self.barchart_generator.save_overall_bar_charts("D:\\Google_Play_Fraud_Benign_Malware\\Visualizations")

    def test_calculate_word_occurrence_valid(self):
        self.barchart_generator.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2012_top_10.csv'])
        self.barchart_generator.categorize_words_by_valence()
        self.assertNotEqual(0, len(self.barchart_generator.file_valence_data))

    # missing text column
    def test_calculate_word_occurence_invalid(self):
        self.barchart_generator.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\fraud_apps_640_review_info_final_2016_top_10_missing_text_column.csv'])
        self.barchart_generator.categorize_words_by_valence()
        self.assertNotEqual(0, len(self.barchart_generator.file_valence_data))

    def test_create_divergent_valence_bar_chart_valid(self):
        self.barchart_generator.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2012_top_10.csv'])
        self.barchart_generator.categorize_words_by_valence()
        self.barchart_generator.create_divergent_valence_bar_chart()
        self.assertNotEqual(0, len(self.barchart_generator.file_valence_data))

    def test_create_divergent_valence_bar_chart_invalid(self):
        self.barchart_generator.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2012_top_10.csv'])
        self.barchart_generator.categorize_words_by_valence()
        self.barchart_generator.file_valence_data = None
        self.barchart_generator.create_divergent_valence_bar_chart()
        self.assertNotEqual(0, len(self.barchart_generator.file_valence_data))

    def test_categorize_rating(self):
        self.barchart_generator.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2012_top_10.csv'])
        self.barchart_generator.categorize_ratings()

    def test_create_rating_bar_charts(self):
        self.barchart_generator.acquire_csv_files([
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2012\\sentiment\\fraud_apps_2012_all_anon_reviews_including_sentiment_score.csv'])
        self.barchart_generator.categorize_ratings()
        self.barchart_generator.create_rating_bar_charts()


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
