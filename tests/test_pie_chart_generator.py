from implementation.pie_chart_generator import PieChartGenerator
import pandas as pd
import unittest
import matplotlib.pyplot as plt


class TestPieChartGenerator(unittest.TestCase):

    def setUp(self):
        self.pie_chart_generator = PieChartGenerator()

    def test_get_counts_from_csv_files_valid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2012\\fraud_apps_2012_all_reviews_including_sentiment_score.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files()
        self.assertNotEqual(0, self.pie_chart_generator.chart_data[0][1].neutral_chart_section.section_count)
        self.assertNotEqual(0, self.pie_chart_generator.chart_data[0][1].negative_chart_section.section_count)
        self.assertNotEqual(0, self.pie_chart_generator.chart_data[0][1].positive_chart_section.section_count)
        self.assertNotEqual(0, self.pie_chart_generator.chart_data[0][1].neutral_chart_section.section_topic_count)
        print(self.pie_chart_generator.chart_data[0][1].neutral_chart_section.section_topic_count)
        self.assertNotEqual(0, self.pie_chart_generator.chart_data[0][1].negative_chart_section.section_topic_count)
        print(self.pie_chart_generator.chart_data[0][1].negative_chart_section.section_topic_count)
        self.assertNotEqual(0, self.pie_chart_generator.chart_data[0][1].positive_chart_section.section_topic_count)
        print(self.pie_chart_generator.chart_data[0][1].positive_chart_section.section_topic_count)

    # csv file is missing the compound column
    def test_get_counts_from_csv_files_invalid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2012_top_10.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files()
        self.assertNotEqual(0, self.pie_chart_generator.chart_data[0][1].neutral_chart_section.section_count)
        self.assertNotEqual(0, self.pie_chart_generator.chart_data[0][1].negative_chart_section.section_count)
        self.assertNotEqual(0, self.pie_chart_generator.chart_data[0][1].positive_chart_section.section_count)

    def test_create_basic_pie_chart_valid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2014_top_10_including_sentiment_score.csv',
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2012_top_10_including_sentiment_score.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files()
        self.pie_chart_generator.create_basic_pie_chart()
        self.pie_chart_generator.display_basic_pie_chart()
        self.assertNotEqual(0, len(self.pie_chart_generator.basic_charts))

    # invalid chart data values
    def test_create_basic_pie_chart_invalid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2014_top_10_including_sentiment_score.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files()
        self.pie_chart_generator.chart_data.positive_chart_section = 0
        self.pie_chart_generator.create_basic_pie_chart()
        self.assertNotEqual(0, len(self.pie_chart_generator.basic_charts))

    def test_create_nested_pie_chart_sentiment_and_topic_valid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2016\\fraud_apps_2016_all_anon_reviews_including_sentiment_score.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files(acquire_topics=True)
        self.pie_chart_generator.create_nested_pie_chart_sentiment_and_topic()
        plt.show()
        # self.assertNotEqual(0, len(self.pie_chart_generator.nested_charts))

    def test_create_nested_pie_chart_rating_and_topic_valid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2016\\fraud_apps_2016_all_anon_reviews_including_sentiment_score.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files(acquire_topics=True)
        self.pie_chart_generator.create_nested_pie_chart_rating_and_topic()
        plt.show()
        # self.assertNotEqual(0, len(self.pie_chart_generator.nested_charts))

    def test_save_basic_pie_charts_valid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2014_top_10_including_sentiment_score.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files()
        self.pie_chart_generator.create_basic_pie_chart()
        self.pie_chart_generator.save_basic_pie_charts(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest')

    def test_create_nested_pie_chart_sentiment_and_rating_valid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2013\\fraud_apps_2013_all_reviews_including_sentiment_score.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files()
        self.pie_chart_generator.create_nested_pie_chart_sentiment_and_rating()
        self.assertNotEqual(0, len(self.pie_chart_generator.nested_charts))

    # invalid chart data
    def test_create_nested_pie_chart_sentiment_and_rating_invalid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2012_top_10_including_sentiment_score.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files()
        self.pie_chart_generator.create_nested_pie_chart_sentiment_and_rating()
        self.assertNotEqual(0, len(self.pie_chart_generator.nested_charts))

    def test_save_nested_pie_chart_sentiment_and_rating_valid(self):
        self.pie_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest\\fraud_apps_640_review_info_final_2016_top_10_including_sentiment_score.csv'])
        self.pie_chart_generator.get_chart_data_from_csv_files()
        self.pie_chart_generator.create_nested_pie_chart_sentiment_and_rating()
        self.pie_chart_generator.save_nested_pie_chart(
            'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\PieChartTest')
        self.assertNotEqual(0, len(self.pie_chart_generator.nested_charts))


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
