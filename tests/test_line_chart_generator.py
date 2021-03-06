import unittest
from implementation.line_chart_generator import LineChartGenerator
from implementation.file_receiver import FileReceiver


class TestLineChart(unittest.TestCase):

    def setUp(self):
        self.file_receiver = FileReceiver()
        self.line_chart_generator = LineChartGenerator()

    def test_categorize_text_by_word_count(self):
        self.line_chart_generator.acquire_csv_files([
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_all_anon_reviews_including_sentiment_score.csv"
        ])
        self.line_chart_generator.categorize_text_by_character_count()
        self.assertNotEqual(0, len(self.line_chart_generator.per_file_character_count))

    def test_create_simple_line_chart(self):
        self.line_chart_generator.acquire_csv_files([
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2012\\sentiment\\fraud_apps_2012_all_anon_reviews_including_sentiment_score.csv",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment\\fraud_apps_2013_all_anon_reviews_including_sentiment_score.csv",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2014\\sentiment\\fraud_apps_2014_all_anon_including_sentiment_score.csv",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2015\\sentiment\\fraud_apps_2015_all_anon_reviews_including_sentiment_score.csv",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2016\\sentiment\\fraud_apps_2016_all_anon_reviews_including_sentiment_score.csv"
        ])
        self.line_chart_generator.categorize_text_by_character_count()
        self.line_chart_generator.create_simple_line_chart()
        self.assertNotEqual(0, len(self.line_chart_generator.line_charts))

    def test_calculate_monthly_app_reviews_valid(self):
        self.line_chart_generator.acquire_csv_files(
            ['D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\fraud_apps_640_review_info_final_2014_top_10.csv'])
        self.line_chart_generator.calculate_reviews_by_identifiable_individuals_per_app()

    # missing data column - month
    def test_calculate_monthly_app_reviews_invalid(self):
        self.line_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\fraud_apps_640_review_info_final_2014_top_10_no_month_column.csv'])
        self.line_chart_generator.calculate_reviews_by_identifiable_individuals_per_app()

    def test_create_charts_valid(self):
        self.line_chart_generator.create_charts([('APP_1',
                                                  [(1, 47), (2, 51), (3, 30), (4, 34), (5, 18), (6, 22), (7, 11),
                                                   (8, 14), (9, 11), (10, 343), (11, 142), (12, 121)]), ('APP_2',
                                                                                                         [(1, 22),
                                                                                                          (2, 22),
                                                                                                          (3, 26),
                                                                                                          (4, 6),
                                                                                                          (5, 26),
                                                                                                          (6, 61),
                                                                                                          (7, 22),
                                                                                                          (8, 34),
                                                                                                          (9, 26),
                                                                                                          (
                                                                                                              10,
                                                                                                              174),
                                                                                                          (
                                                                                                              11,
                                                                                                              148),
                                                                                                          (12,
                                                                                                           22)]), (
                                                     'APP_3',
                                                     [(1, 25), (2, 34), (3, 14), (4, 22), (5, 20), (6, 119),
                                                      (7, 72),
                                                      (8, 24), (9, 17), (10, 61), (11, 13), (12, 24)]), ('APP_4',
                                                                                                         [(1, 10),
                                                                                                          (2, 16),
                                                                                                          (3, 6),
                                                                                                          (4, 34),
                                                                                                          (5, 36),
                                                                                                          (6, 30),
                                                                                                          (7, 30),
                                                                                                          (8, 18),
                                                                                                          (9, 36),
                                                                                                          (
                                                                                                              10,
                                                                                                              138),
                                                                                                          (
                                                                                                              11,
                                                                                                              112),
                                                                                                          (12,
                                                                                                           22)]), (
                                                     'APP_5',
                                                     [(1, 132), (2, 162), (3, 136), (4, 104), (5, 110), (6, 144),
                                                      (7, 141), (8, 156), (9, 122), (10, 78), (11, 102),
                                                      (12, 118)]), (
                                                     'APP_6',
                                                     [(1, 17), (2, 13), (3, 17), (4, 40), (5, 44), (6, 46),
                                                      (7, 47),
                                                      (8, 58), (9, 52), (10, 59), (11, 80), (12, 168)]), ('APP_7',
                                                                                                          [(1, 32),
                                                                                                           (2, 88),
                                                                                                           (
                                                                                                               3,
                                                                                                               162),
                                                                                                           (
                                                                                                               4,
                                                                                                               121),
                                                                                                           (
                                                                                                               5,
                                                                                                               165),
                                                                                                           (
                                                                                                               6,
                                                                                                               112),
                                                                                                           (
                                                                                                               7,
                                                                                                               149),
                                                                                                           (
                                                                                                               8,
                                                                                                               145),
                                                                                                           (9, 83),
                                                                                                           (10,
                                                                                                            259),
                                                                                                           (11,
                                                                                                            212),
                                                                                                           (12,
                                                                                                            221)]),
                                                 ('APP_8',
                                                  [(1, 247), (2, 178), (3, 226), (4, 182), (5, 160), (6, 121),
                                                   (7, 82), (8, 130), (9, 152), (10, 104), (11, 143), (12, 112)]),
                                                 ('APP_9',
                                                  [(1, 27), (2, 33), (3, 19), (4, 68), (5, 31), (6, 17), (7, 30),
                                                   (8, 37), (9, 52), (10, 149), (11, 89), (12, 53)]), ('APP_10',
                                                                                                       [(1, 96),
                                                                                                        (2, 56),
                                                                                                        (3, 142),
                                                                                                        (4, 172),
                                                                                                        (5, 219),
                                                                                                        (6, 211),
                                                                                                        (7, 130),
                                                                                                        (8, 140),
                                                                                                        (9, 186),
                                                                                                        (10, 147),
                                                                                                        (11, 138),
                                                                                                        (
                                                                                                            12,
                                                                                                            98)])])

    # missing APP_2 in the second element
    def test_create_charts_invalid(self):
        self.line_chart_generator.create_charts([('APP_1',
                                                  [(1, 47), (2, 51), (3, 30), (4, 34), (5, 18), (6, 22), (7, 11),
                                                   (8, 14), (9, 11), (10, 343), (11, 142), (12, 121)]), (
                                                     [(1, 22),
                                                      (2, 22),
                                                      (3, 26),
                                                      (4, 6),
                                                      (5, 26),
                                                      (6, 61),
                                                      (7, 22),
                                                      (8, 34),
                                                      (9, 26),
                                                      (
                                                          10,
                                                          174),
                                                      (
                                                          11,
                                                          148),
                                                      (12,
                                                       22)]), (
                                                     'APP_3',
                                                     [(1, 25), (2, 34), (3, 14), (4, 22), (5, 20), (6, 119),
                                                      (7, 72),
                                                      (8, 24), (9, 17), (10, 61), (11, 13), (12, 24)]), ('APP_4',
                                                                                                         [(1, 10),
                                                                                                          (2, 16),
                                                                                                          (3, 6),
                                                                                                          (4, 34),
                                                                                                          (5, 36),
                                                                                                          (6, 30),
                                                                                                          (7, 30),
                                                                                                          (8, 18),
                                                                                                          (9, 36),
                                                                                                          (
                                                                                                              10,
                                                                                                              138),
                                                                                                          (
                                                                                                              11,
                                                                                                              112),
                                                                                                          (12,
                                                                                                           22)]), (
                                                     'APP_5',
                                                     [(1, 132), (2, 162), (3, 136), (4, 104), (5, 110), (6, 144),
                                                      (7, 141), (8, 156), (9, 122), (10, 78), (11, 102),
                                                      (12, 118)]), (
                                                     'APP_6',
                                                     [(1, 17), (2, 13), (3, 17), (4, 40), (5, 44), (6, 46),
                                                      (7, 47),
                                                      (8, 58), (9, 52), (10, 59), (11, 80), (12, 168)]), ('APP_7',
                                                                                                          [(1, 32),
                                                                                                           (2, 88),
                                                                                                           (
                                                                                                               3,
                                                                                                               162),
                                                                                                           (
                                                                                                               4,
                                                                                                               121),
                                                                                                           (
                                                                                                               5,
                                                                                                               165),
                                                                                                           (
                                                                                                               6,
                                                                                                               112),
                                                                                                           (
                                                                                                               7,
                                                                                                               149),
                                                                                                           (
                                                                                                               8,
                                                                                                               145),
                                                                                                           (9, 83),
                                                                                                           (10,
                                                                                                            259),
                                                                                                           (11,
                                                                                                            212),
                                                                                                           (12,
                                                                                                            221)]),
                                                 ('APP_8',
                                                  [(1, 247), (2, 178), (3, 226), (4, 182), (5, 160), (6, 121),
                                                   (7, 82), (8, 130), (9, 152), (10, 104), (11, 143), (12, 112)]),
                                                 ('APP_9',
                                                  [(1, 27), (2, 33), (3, 19), (4, 68), (5, 31), (6, 17), (7, 30),
                                                   (8, 37), (9, 52), (10, 149), (11, 89), (12, 53)]), ('APP_10',
                                                                                                       [(1, 96),
                                                                                                        (2, 56),
                                                                                                        (3, 142),
                                                                                                        (4, 172),
                                                                                                        (5, 219),
                                                                                                        (6, 211),
                                                                                                        (7, 130),
                                                                                                        (8, 140),
                                                                                                        (9, 186),
                                                                                                        (10, 147),
                                                                                                        (11, 138),
                                                                                                        (
                                                                                                            12,
                                                                                                            98)])])

    def test_create_yearly_app_data_charts_valid(self):
        self.line_chart_generator.yearly_app_data = [[('APP_1',
                                                       [(1, 47), (2, 51), (3, 30), (4, 34), (5, 18), (6, 22), (7, 11),
                                                        (8, 14), (9, 11), (10, 343), (11, 142), (12, 121)]), ('APP_2',
                                                                                                              [(1, 22),
                                                                                                               (2, 22),
                                                                                                               (3, 26),
                                                                                                               (4, 6),
                                                                                                               (5, 26),
                                                                                                               (6, 61),
                                                                                                               (7, 22),
                                                                                                               (8, 34),
                                                                                                               (9, 26),
                                                                                                               (
                                                                                                               10, 174),
                                                                                                               (
                                                                                                               11, 148),
                                                                                                               (12,
                                                                                                                22)]), (
                                                      'APP_3',
                                                      [(1, 25), (2, 34), (3, 14), (4, 22), (5, 20), (6, 119), (7, 72),
                                                       (8, 24), (9, 17), (10, 61), (11, 13), (12, 24)]), ('APP_4',
                                                                                                          [(1, 10),
                                                                                                           (2, 16),
                                                                                                           (3, 6),
                                                                                                           (4, 34),
                                                                                                           (5, 36),
                                                                                                           (6, 30),
                                                                                                           (7, 30),
                                                                                                           (8, 18),
                                                                                                           (9, 36),
                                                                                                           (10, 138),
                                                                                                           (11, 112),
                                                                                                           (12, 22)]), (
                                                      'APP_5',
                                                      [(1, 132), (2, 162), (3, 136), (4, 104), (5, 110), (6, 144),
                                                       (7, 141), (8, 156), (9, 122), (10, 78), (11, 102), (12, 118)]), (
                                                      'APP_6',
                                                      [(1, 17), (2, 13), (3, 17), (4, 40), (5, 44), (6, 46), (7, 47),
                                                       (8, 58), (9, 52), (10, 59), (11, 80), (12, 168)]), ('APP_7',
                                                                                                           [(1, 32),
                                                                                                            (2, 88),
                                                                                                            (3, 162),
                                                                                                            (4, 121),
                                                                                                            (5, 165),
                                                                                                            (6, 112),
                                                                                                            (7, 149),
                                                                                                            (8, 145),
                                                                                                            (9, 83),
                                                                                                            (10, 259),
                                                                                                            (11, 212),
                                                                                                            (12, 221)]),
                                                      ('APP_8',
                                                       [(1, 247), (2, 178), (3, 226), (4, 182), (5, 160), (6, 121),
                                                        (7, 82), (8, 130), (9, 152), (10, 104), (11, 143), (12, 112)]),
                                                      ('APP_9',
                                                       [(1, 27), (2, 33), (3, 19), (4, 68), (5, 31), (6, 17), (7, 30),
                                                        (8, 37), (9, 52), (10, 149), (11, 89), (12, 53)]), ('APP_10',
                                                                                                            [(1, 96),
                                                                                                             (2, 56),
                                                                                                             (3, 142),
                                                                                                             (4, 172),
                                                                                                             (5, 219),
                                                                                                             (6, 211),
                                                                                                             (7, 130),
                                                                                                             (8, 140),
                                                                                                             (9, 186),
                                                                                                             (10, 147),
                                                                                                             (11, 138),
                                                                                                             (
                                                                                                             12, 98)])],
                                                     [('APP_1',
                                                       [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
                                                        (9, 0), (10, 0), (11, 0), (12, 2)]), ('APP_2',
                                                                                              [(1, 0), (2, 0), (3, 0),
                                                                                               (4, 0), (5, 0), (6, 0),
                                                                                               (7, 0), (8, 0), (9, 2),
                                                                                               (10, 0), (11, 0),
                                                                                               (12, 2)]), ('APP_3',
                                                                                                           [(1, 0),
                                                                                                            (2, 0),
                                                                                                            (3, 0),
                                                                                                            (4, 0),
                                                                                                            (5, 2),
                                                                                                            (6, 0),
                                                                                                            (7, 2),
                                                                                                            (8, 0),
                                                                                                            (9, 2),
                                                                                                            (10, 2),
                                                                                                            (11, 4),
                                                                                                            (12, 24)]),
                                                      ('APP_4',
                                                       [(1, 0), (2, 0), (3, 1), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
                                                        (9, 0), (10, 0), (11, 0), (12, 1)]), ('APP_5',
                                                                                              [(1, 0), (2, 0), (3, 0),
                                                                                               (4, 0), (5, 0), (6, 0),
                                                                                               (7, 0), (8, 0), (9, 0),
                                                                                               (10, 0), (11, 2),
                                                                                               (12, 0)]), ('APP_6',
                                                                                                           [(1, 0),
                                                                                                            (2, 0),
                                                                                                            (3, 0),
                                                                                                            (4, 0),
                                                                                                            (5, 0),
                                                                                                            (6, 2),
                                                                                                            (7, 0),
                                                                                                            (8, 2),
                                                                                                            (9, 0),
                                                                                                            (10, 2),
                                                                                                            (11, 4),
                                                                                                            (12, 4)]), (
                                                      'APP_7',
                                                      [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
                                                       (9, 0), (10, 0), (11, 0), (12, 1)])]]
        self.line_chart_generator.create_yearly_app_data_charts()

    # missing second element in the list
    def test_create_yearly_app_data_charts_invalid(self):
        self.line_chart_generator.yearly_app_data = [[('APP_1',
                                                       [(1, 47), (2, 51), (3, 30), (4, 34), (5, 18), (6, 22), (7, 11),
                                                        (8, 14), (9, 11), (10, 343), (11, 142), (12, 121)]), ('APP_2',
                                                                                                              [(1, 22),
                                                                                                               (2, 22),
                                                                                                               (3, 26),
                                                                                                               (4, 6),
                                                                                                               (5, 26),
                                                                                                               (6, 61),
                                                                                                               (7, 22),
                                                                                                               (8, 34),
                                                                                                               (9, 26),
                                                                                                               (
                                                                                                                   10,
                                                                                                                   174),
                                                                                                               (
                                                                                                                   11,
                                                                                                                   148),
                                                                                                               (12,
                                                                                                                22)]), (
                                                          'APP_3',
                                                          [(1, 25), (2, 34), (3, 14), (4, 22), (5, 20), (6, 119),
                                                           (7, 72),
                                                           (8, 24), (9, 17), (10, 61), (11, 13), (12, 24)]), ('APP_4',
                                                                                                              [(1, 10),
                                                                                                               (2, 16),
                                                                                                               (3, 6),
                                                                                                               (4, 34),
                                                                                                               (5, 36),
                                                                                                               (6, 30),
                                                                                                               (7, 30),
                                                                                                               (8, 18),
                                                                                                               (9, 36),
                                                                                                               (
                                                                                                               10, 138),
                                                                                                               (
                                                                                                               11, 112),
                                                                                                               (12,
                                                                                                                22)]), (
                                                          'APP_5',
                                                          [(1, 132), (2, 162), (3, 136), (4, 104), (5, 110), (6, 144),
                                                           (7, 141), (8, 156), (9, 122), (10, 78), (11, 102),
                                                           (12, 118)]), (
                                                          'APP_6',
                                                          [(1, 17), (2, 13), (3, 17), (4, 40), (5, 44), (6, 46),
                                                           (7, 47),
                                                           (8, 58), (9, 52), (10, 59), (11, 80), (12, 168)]), ('APP_7',
                                                                                                               [(1, 32),
                                                                                                                (2, 88),
                                                                                                                (
                                                                                                                3, 162),
                                                                                                                (
                                                                                                                4, 121),
                                                                                                                (
                                                                                                                5, 165),
                                                                                                                (
                                                                                                                6, 112),
                                                                                                                (
                                                                                                                7, 149),
                                                                                                                (
                                                                                                                8, 145),
                                                                                                                (9, 83),
                                                                                                                (10,
                                                                                                                 259),
                                                                                                                (11,
                                                                                                                 212),
                                                                                                                (12,
                                                                                                                 221)]),
                                                      ('APP_8',
                                                       [(1, 247), (2, 178), (3, 226), (4, 182), (5, 160), (6, 121),
                                                        (7, 82), (8, 130), (9, 152), (10, 104), (11, 143), (12, 112)]),
                                                      ('APP_9',
                                                       [(1, 27), (2, 33), (3, 19), (4, 68), (5, 31), (6, 17), (7, 30),
                                                        (8, 37), (9, 52), (10, 149), (11, 89), (12, 53)]), ('APP_10',
                                                                                                            [(1, 96),
                                                                                                             (2, 56),
                                                                                                             (3, 142),
                                                                                                             (4, 172),
                                                                                                             (5, 219),
                                                                                                             (6, 211),
                                                                                                             (7, 130),
                                                                                                             (8, 140),
                                                                                                             (9, 186),
                                                                                                             (10, 147),
                                                                                                             (11, 138),
                                                                                                             (
                                                                                                                 12,
                                                                                                                 98)])],
                                                     []]
        self.line_chart_generator.create_yearly_app_data_charts()

    def test_calculate_all_year_data_valid(self):
        self.line_chart_generator.acquire_csv_files(
            ['D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\fraud_apps_640_review_info_final_2014_top_10.csv'])
        self.line_chart_generator.calculate_overall_reviews_by_identifiable_individuals()

    def test_calculate_all_year_data_invalid(self):
        self.line_chart_generator.acquire_csv_files(
            [
                'D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test\\fraud_apps_640_review_info_final_2014_top_10_no_month_column.csv'])
        self.line_chart_generator.calculate_overall_reviews_by_identifiable_individuals()

    def test_create_all_year_data_chart_valid(self):
        self.line_chart_generator.all_year_data = [('2014', [(1, 655), (2, 653), (3, 778), (4, 783), (5, 829), (6, 883),
                                                             (7, 714), (8, 756), (9, 737), (10, 1512), (11, 1179),
                                                             (12, 959)])]
        self.line_chart_generator.create_all_year_data_chart()

    # missing year tag
    def test_create_all_year_data_chart_invalid(self):
        self.line_chart_generator.all_year_data = [([(1, 655), (2, 653), (3, 778), (4, 783), (5, 829), (6, 883),
                                                             (7, 714), (8, 756), (9, 737), (10, 1512), (11, 1179),
                                                             (12, 959)])]
        self.line_chart_generator.create_all_year_data_chart()

    def test_save_all_year_charts_valid(self):
        self.file_receiver.acquire_input_path()
        self.line_chart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.line_chart_generator.calculate_overall_reviews_by_identifiable_individuals()
        self.line_chart_generator.calculate_reviews_by_identifiable_individuals_per_app()
        self.line_chart_generator.create_all_year_data_chart()
        self.line_chart_generator.create_yearly_app_data_charts()
        self.line_chart_generator.save_all_year_charts('D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test')
        self.line_chart_generator.save_yearly_app_charts('D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test')

    #invalid output path
    def test_save_all_year_charts_invalid(self):
        self.file_receiver.acquire_input_path()
        self.line_chart_generator.acquire_csv_files(self.file_receiver.csv_files)
        self.line_chart_generator.calculate_overall_reviews_by_identifiable_individuals()
        self.line_chart_generator.calculate_reviews_by_identifiable_individuals_per_app()
        self.line_chart_generator.create_all_year_data_chart()
        self.line_chart_generator.create_yearly_app_data_charts()
        self.line_chart_generator.save_all_year_charts('')
        self.line_chart_generator.save_yearly_app_charts('D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\Test')

if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
