import matplotlib.pyplot as plt
import pandas as pd
from re import search
from .data_adjustments import DataAdjustment
import math
import os


class LineChartGenerator(object):

    def __init__(self):
        self.csv_files = []
        # a list containing lists which are composed of all of the months of the year and the review count for each month
        # in this form
        self.line_charts = []
        self.data_adjuster = DataAdjustment()
        self.yearly_app_data = []
        self.yearly_app_data_line_charts = []
        self.all_year_data = []
        self.all_year_data_line_charts = []

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def calculate_all_year_data(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
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

    def calculate_yearly_app_data(self):
        for file in self.csv_files:
            yearly_data = []
            data = pd.read_csv(file)
            app_reviews = sorted(data['App_ID'].unique())
            app_count = 1
            for app_id in app_reviews:
                all_monthly_data = []
                all_monthly_data.append(data.loc[data['App_ID'] == app_id, 'Month'].value_counts().to_dict())
                app_monthly_data = []
                for value in all_monthly_data:
                    for i in range(1, 13):
                        app_monthly_data.append((i, value.get(i, 0)))
                yearly_data.append(("APP_" + str(app_count), app_monthly_data))
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
                ax.plot((x_point, previous_x), (previous_y, y_point), c='k', zorder=1, linewidth=0.5)

        y_count = len(y_labels) - 2
        for year in all_data:
            for key, value in year[1]:
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

    @staticmethod
    def create_legend(max_value):
        plt.scatter([], [], c='r', s=200, marker='s', edgecolors='k', linewidths='2',
                    label="0 <-> " + str(max_value // 4))
        plt.scatter([], [], c='r', s=500, marker='s', edgecolors='k', linewidths='2',
                    label=str(max_value // 4 + 1) + " <-> " + str(max_value // 4 * 2))
        plt.scatter([], [], c='r', s=800, marker='s', edgecolors='k', linewidths='2',
                    label=str(max_value // 4 * 2 + 1) + " <-> " + str(max_value // 4 * 3))
        plt.scatter([], [], c='r', s=1100, marker='s', edgecolors='k', linewidths='2',
                    label=str(max_value // 4 * 4) + " +")
        plt.legend(loc='upper left', title='Review Count', title_fontsize=18, bbox_to_anchor=(1.04, 1), borderpad=1.5,
                   labelspacing=2.5, handletextpad=1.5, frameon=False)

    @staticmethod
    def categorise_symbol_size(value, max_value):
        size = 200
        difference = 300
        if value == 0:
            return 0
        elif 0 < value <= max_value // 4:
            return size
        elif 751 <= value <= max_value // 4 * 2:
            return size + difference * 1
        elif 1501 <= value <= max_value // 4 * 3:
            return size + difference * 2
        else:
            return size + difference * 3

    @staticmethod
    def get_maximum_data_value(all_data):
        max_val = 0
        for data in all_data:
            for key, value in data[1]:
                if value > max_val:
                    max_val = value
        return max_val

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

    def display_bar_charts(self):
        for chart in self.line_charts:
            chart.show()
