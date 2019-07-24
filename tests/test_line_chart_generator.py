import unittest
from implementation.line_chart_generator import LineChartGenerator
from implementation.file_receiver import FileReceiver


class Test_LineChart(unittest.TestCase):

    def setUp(self):
        self.file_receiver = FileReceiver()
        self.line_chart_generator = LineChartGenerator()

    def test_calculate_monthly_reviews(self):
        self.file_receiver.acquire_input_path()
        self.line_chart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.line_chart_generator.calculate_monthly_reviews()
        self.assertEqual(5, len(self.line_chart_generator.all_data))
        print(self.line_chart_generator.all_data)

    def test_create_line_chart(self):
        self.file_receiver.acquire_input_path()
        self.line_chart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.line_chart_generator.calculate_monthly_reviews()
        self.line_chart_generator.create_line_charts()

if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
