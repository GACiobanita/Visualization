import unittest
from implementation.bar_chart_generator import BarChartGenerator
from implementation.file_receiver import FileReceiver


class Test_BarChart(unittest.TestCase):

    def setUp(self):
        self.barchart_generator = BarChartGenerator()
        self.file_receiver = FileReceiver()

    def test_calculate_text_length_valid(self):
        self.file_receiver.acquire_input_path()
        self.barchart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.barchart_generator.calculate_text_length()
        self.assertNotEqual(0, len(self.barchart_generator.csv_files))

    def test_calculate_text_length_invalid(self):
        self.file_receiver.acquire_input_path()
        self.barchart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.barchart_generator.calculate_text_length()
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
        self.barchart_generator.calculate_text_length()
        self.barchart_generator.create_overall_bar_charts()
        self.barchart_generator.display_bar_charts()
        self.barchart_generator.save_bar_charts("D:\\Google_Play_Fraud_Benign_Malware\\Visualizations")

    def test_save_bar_charts_invalid(self):
        self.file_receiver.acquire_input_path()
        self.barchart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.barchart_generator.calculate_text_length()
        self.barchart_generator.create_overall_bar_charts()
        self.barchart_generator.display_bar_charts()
        self.barchart_generator.save_bar_charts("D:\\Google_Play_Fraud_Benign_Malware\\Visualizations")


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
