import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .data_adjustments import DataAdjustment
import os


class BarChartGenerator(object):

    def __init__(self):
        self.csv_files = []
        self.total_word_count = {'0-50': 0, '51-100': 0, '101-200': 0, '201-300': 0, '301-400': 0, '400+': 0}
        self.per_file_word_count = []
        self.bar_charts = []
        self.data_adjuster = DataAdjustment()

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def calculate_text_length(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
            for text in data['Text']:
                text = str(text)
                text = self.data_adjuster.remove_string_punctuation(text)
                self.check_bin(len(text))
            self.per_file_word_count.append(self.total_word_count)
            self.total_word_count = {'0-50': 0, '51-100': 0, '101-200': 0, '201-300': 0, '301-400': 0, '400+': 0}
        self.total_word_count['0-50'] = sum([item['0-50'] for item in self.per_file_word_count])
        self.total_word_count['51-100'] = sum([item['51-100'] for item in self.per_file_word_count])
        self.total_word_count['101-200'] = sum([item['51-100'] for item in self.per_file_word_count])
        self.total_word_count['201-300'] = sum([item['51-100'] for item in self.per_file_word_count])
        self.total_word_count['301-400'] = sum([item['51-100'] for item in self.per_file_word_count])
        self.total_word_count['400+'] = sum([item['400+'] for item in self.per_file_word_count])

    def check_bin(self, length):
        for key, value in self.total_word_count.items():
            min_max = key.split("-")
            if len(min_max) != 1:
                min = int(min_max[0])
                max = int(min_max[1])
                if min <= length <= max:
                    self.total_word_count[key] += 1
            else:
                min = int(min_max[0][:-1])
                if min < length:
                    self.total_word_count[key] += 1

    def create_overall_bar_charts(self):
        self.create_horizontal_bar_chart(self.total_word_count, "Total Word Usage 2012-2016")
        year = 2012
        for dictionary in self.per_file_word_count:
            self.create_horizontal_bar_chart(dictionary, "Total Word Usage for " + str(year))
            year += 1

    def display_bar_charts(self):
        for chart in self.bar_charts:
            chart.show()

    def create_horizontal_bar_chart(self, dictionary, title):
        fig = plt.figure()
        ax = fig.add_subplot()

        # data
        bar_height = []
        bar_name = []
        for key, value in dictionary.items():
            bar_height.append(value)
            bar_name.append(key)

        y_pos = np.arange(len(bar_name))

        ax.barh(y_pos, bar_height, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_name)
        ax.invert_yaxis()
        ax.set_xlabel('Words Per Rating')
        ax.set_title(title)
        ax.axvline(np.mean(list(dictionary.values())), color='red',
                   label='Average = ' + str(int(np.mean(list(dictionary.values())))))

        plt.legend(loc=0)
        self.bar_charts.append(fig)

    def save_bar_charts(self, output_folder_path):
        count = 0
        self.bar_charts[0].savefig(output_folder_path + "\\" + "fraud_apps_640_review_info_final_total_word_usage_bar_chart.png")
        for chart in self.bar_charts[1:]:
            head, tail = os.path.split(self.csv_files[count])
            chart.savefig(output_folder_path + "\\" + tail[:-4] + "_bar_chart.png")
            count += 1

