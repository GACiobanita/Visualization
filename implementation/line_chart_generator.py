import matplotlib.pyplot as plt
import pandas as pd
from .data_adjustments import DataAdjustment
import math
import os
import numpy as np


class LineChartGenerator(object):

    def __init__(self):
        self.csv_files = []
        self.line_charts = []
        self.data_adjuster = DataAdjustment()
        self.yearly_app_data = []
        self.yearly_app_data_line_charts = []
        self.all_year_data = []
        self.all_year_data_line_charts = []

        self.total_word_count = {'0-50': 0, '51-100': 0, '101-200': 0, '201-300': 0, '301-400': 0, '400+': 0}
        self.per_file_character_count = []

        # chart creation variables
        self.SMALLEST_SYMBOL_SIZE = 200
        self.SYMBOL_SIZE_DIFFERENCE = 300
        self.NUMBER_OF_BUCKETS = 4

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def calculate_overall_reviews_by_identifiable_individuals(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
            # find the total number of review per month for all apps
            found_monthly_reviews = sorted(data['Month'].value_counts().to_dict().items())

            for i in range(1, 13):
                found = False
                for x, y in found_monthly_reviews:
                    if x == i:
                        found = True
                if found is False:
                    found_monthly_reviews.append((i, 0))

            year = self.data_adjuster.get_year_from_string(file)
            self.all_year_data.append((year, found_monthly_reviews))

    def create_all_year_data_chart(self):
        self.all_year_data_line_charts.append(self.create_charts(self.all_year_data))

    def calculate_reviews_by_identifiable_individuals_per_app(self):
        for file in self.csv_files:
            yearly_data = []
            data = pd.read_csv(file)
            app_reviews = sorted(data['App_ID'].unique())
            app_count = 1
            for app_id in app_reviews:
                all_monthly_data = []
                # for each app we find the total number of reviews for each month
                all_monthly_data.append(data.loc[data['App_ID'] == app_id, 'Month'].value_counts().to_dict())
                per_app_monthly_data = []
                for value in all_monthly_data:
                    for i in range(1, 13):  # won't use 13
                        # for each month in the year, get the number of reviews
                        per_app_monthly_data.append((i, value.get(i, 0)))
                # initially used app_id but since it uses the package name of the app it would not help in the visualization
                yearly_data.append(("APP_" + str(app_count), per_app_monthly_data))
                app_count += 1
            self.yearly_app_data.append(yearly_data)

    def create_yearly_app_data_charts(self):
        for yearly_app_data in self.yearly_app_data:
            self.yearly_app_data_line_charts.append(self.create_charts(yearly_app_data))

    def create_charts(self, all_data):

        fig = plt.figure(figsize=(10, len(all_data)))
        plt.title('Reviews created by identifiable individuals', fontsize=18)
        ax = fig.add_subplot(1, 1, 1)

        figure_size_value = self.get_maximum_data_value(all_data)

        x_labels = ['']
        y_labels = ['']
        self.create_axis_labels(data_container=all_data, x_axis_labels=x_labels, y_axis_labels=y_labels)

        x_axis_points = []
        y_axis_points = []
        self.create_axis_points(data_container=all_data, x_axis_points=x_axis_points, y_axis_points=y_axis_points)

        for y_point in y_axis_points:
            previous_x = 1
            previous_y = y_point
            for x_point in x_axis_points:
                # plot the lines that will be draw for each app
                # on these lines symbols of different sizes will appear at specific time points
                ax.plot((x_point, previous_x), (previous_y, y_point), c='k', zorder=1, linewidth=0.5)

        # there are 2 extra labels as [] in order to space out labels on the Y axis
        y_count = len(y_labels) - 2
        for year in all_data:
            for key, value in year[1]:
                # place each symbol on the Y axis , and key represents the X axis location
                ax.scatter(key, y_count,
                           s=self.categorise_symbol_size(value, self.round_up(figure_size_value)),
                           c='r',
                           edgecolors='k',
                           linewidths='2',
                           zorder=2,
                           marker='s',
                           label=None)
            y_count -= 1

        # And a corresponding grid
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(list(range(0, len(x_labels), 1)))
        plt.yticks(list(range(0, len(y_labels), 1)))
        plt.tick_params(pad=0)
        ax.set_xticklabels(
            x_labels, rotation=45, fontsize=10, va='top', ha='right')
        ax.set_yticklabels(
            y_labels, fontsize=10)

        self.create_legend(self.round_up(figure_size_value, -len(str(figure_size_value)) + 1))

        plt.tight_layout()
        return fig

    @staticmethod
    # depending on the amount of data in the container the points are created
    def create_axis_points(data_container, x_axis_points, y_axis_points):
        x_count = 1
        y_count = 1
        for data in data_container:
            y_axis_points.append(y_count)
            y_count += 1
        for data in data_container[0][-1]:
            x_axis_points.append(x_count)
            x_count += 1

    # extract the keys from the data set to be used as labels in the graph
    def create_axis_labels(self, data_container, x_axis_labels, y_axis_labels):
        for key, value in data_container[0][-1]:
            x_axis_labels.append(self.data_adjuster.get_month_from_int(key))
        for data_name, data_list in data_container:
            y_axis_labels.append(data_name)
        x_axis_labels.append('')
        y_axis_labels.append('')

    # legend based on the symbols that are currently in the chart
    # uses the sizes created for classification
    def create_legend(self, max_value):
        plt.scatter([], [], c='r', s=self.SMALLEST_SYMBOL_SIZE, marker='s', edgecolors='k', linewidths='2',
                    label="0 <-> " + str(max_value // self.NUMBER_OF_BUCKETS))
        plt.scatter([], [], c='r', s=self.SYMBOL_SIZE_DIFFERENCE + self.SMALLEST_SYMBOL_SIZE, marker='s',
                    edgecolors='k', linewidths='2',
                    label=str(max_value // self.NUMBER_OF_BUCKETS + 1) + " <-> " + str(
                        max_value // self.NUMBER_OF_BUCKETS * 2))
        plt.scatter([], [], c='r', s=self.SYMBOL_SIZE_DIFFERENCE * 2 + self.SMALLEST_SYMBOL_SIZE, marker='s',
                    edgecolors='k', linewidths='2',
                    label=str(max_value // self.NUMBER_OF_BUCKETS * 2 + 1) + " <-> " + str(
                        max_value // self.NUMBER_OF_BUCKETS * 3))
        plt.scatter([], [], c='r', s=self.SYMBOL_SIZE_DIFFERENCE * 3 + self.SMALLEST_SYMBOL_SIZE, marker='s',
                    edgecolors='k', linewidths='2',
                    label=str(max_value // self.NUMBER_OF_BUCKETS * 4) + " +")
        plt.legend(loc='upper left', title='Review Count', title_fontsize=18, bbox_to_anchor=(1.04, 1), borderpad=1.5,
                   labelspacing=2.5, handletextpad=1.5, frameon=False)

    # symbol size is determined by the amount of reviews in the time point
    def categorise_symbol_size(self, value, max_value):
        size = self.SMALLEST_SYMBOL_SIZE
        difference = self.SYMBOL_SIZE_DIFFERENCE
        if value == 0:
            return 0
        elif 0 < value <= max_value // self.NUMBER_OF_BUCKETS:
            return size
        elif 751 <= value <= max_value // self.NUMBER_OF_BUCKETS * 2:
            return size + difference * 1
        elif 1501 <= value <= max_value // self.NUMBER_OF_BUCKETS * 3:
            return size + difference * 2
        else:
            return size + difference * 3

    # the maximum amount of reviews that happened overall, to determine a maximum symbol size
    @staticmethod
    def get_maximum_data_value(all_data):
        max_val = 0
        for data in all_data:
            for key, value in data[1]:
                if value > max_val:
                    max_val = value
        return max_val

    # round up the maximum amount in order to avoid maximum values such as 2875 or 3
    # this would round up to 3000 and 10 in such cases
    @staticmethod
    def round_up(n, decimals=0):
        multiplier = 10 ** decimals
        return int(math.ceil(n * multiplier) / multiplier)

    def save_all_year_charts(self, output_folder_path):
        count = 0
        for chart in self.all_year_data_line_charts:
            chart.savefig(output_folder_path + "\\" + "fraud_apps_640_yearly_line_chart" + str(count) + ".png")
            count += 1

    def save_yearly_app_charts(self, output_folder_path):
        count = 0
        for chart in self.yearly_app_data_line_charts:
            head, tail = os.path.split(self.csv_files[count])
            chart.savefig(output_folder_path + "\\" + tail[:-4] + "_line_chart.png")
            count += 1

    def display_line_charts(self):
        for chart in self.line_charts:
            chart.show()

    # classify each review in the csv_files into different categories based on the number of words contained
    def categorize_text_by_character_count(self):
        for file in self.csv_files:
            current_word_count = {'0-50': 0, '51-100': 0, '101-200': 0, '201-300': 0, '301-400': 0, '400+': 0}
            data = pd.read_csv(file)
            for text in data['Text']:
                text = str(text)
                text = self.data_adjuster.remove_string_punctuation(text)
                self.allocate_review_to_count_category(len(text), current_word_count)
            self.per_file_character_count.append((self.data_adjuster.get_year_from_string(file), current_word_count))
        self.calculate_total_word_count_of_files()

    def calculate_total_word_count_of_files(self):
        self.total_word_count['0-50'] = sum([item['0-50'] for year, item in self.per_file_character_count])
        self.total_word_count['51-100'] = sum([item['51-100'] for year, item in self.per_file_character_count])
        self.total_word_count['101-200'] = sum([item['51-100'] for year, item in self.per_file_character_count])
        self.total_word_count['201-300'] = sum([item['51-100'] for year, item in self.per_file_character_count])
        self.total_word_count['301-400'] = sum([item['51-100'] for year, item in self.per_file_character_count])
        self.total_word_count['400+'] = sum([item['400+'] for year, item in self.per_file_character_count])

    @staticmethod
    def allocate_review_to_count_category(length, category_container):
        for key, value in category_container.items():
            # split the key into min and max values in order to increase the count based on text length
            min_max = key.split("-")
            if len(min_max) != 1:
                minimum = int(min_max[0])
                maximum = int(min_max[1])
                if minimum <= length <= maximum:
                    category_container[key] += 1
            else:
                minimum = int(min_max[0][:-1])
                if minimum < length:
                    category_container[key] += 1

    def create_overall_bar_charts(self):
        # create the total word count bar chart
        self.create_simple_line_chart(self.total_word_count, "Total Word Usage 2012-2016")
        # create a bar chart for each file
        for year, dictionary in self.per_file_character_count:
            self.create_simple_line_chart(dictionary, "Total Word Usage for " + str(year))

    def create_simple_line_chart(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()
        color_list = ['b', 'g', 'r', 'y', 'c']
        year_list = []
        color_count = 0
        lines = []
        for year, dictionary in self.per_file_character_count:
            # data
            line_height = []
            line_name = []
            for key, value in dictionary.items():
                # bar height is the value of the key, in this case the total count
                line_height.append(value)
                # and the name of the bar is the key itself
                line_name.append(key)

            x_pos = np.arange(len(line_name))  # is the number of items in the dict

            ax.plot(x_pos, line_height, color=color_list[color_count])
            year_list.append(color_list[color_count]+' '+str(year))
            color_count+=1

            # create a legend explaining the average
            # loc 0 places the legend in the 'best' location where there would be minimum overlap with the chart
        ax.set_xticks(x_pos)
        ax.set_ylabel('Number of reviews')
        ax.set_xlabel('Characters per review')
        ax.set_xticklabels(line_name)
        ax.set_title("Character usage between 2012-2016")
        plt.legend(["2012", "2013", "2014", "2015", "2016"], loc='0')
        self.line_charts.append(fig)

    def save_line_chart(self, output_folder_path):
        for chart in self.line_charts:
            chart.savefig(output_folder_path + "\\total_word_usage_line_chart.png")
