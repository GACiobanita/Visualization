import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .data_adjustments import DataAdjustment
import os


class LineChartGenerator(object):

    def __init__(self):
        self.csv_files = []
        # a list containing lists which are composed of all of the months of the year and the review count for each month
        # in this form
        # ((1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0), (..), (..) ...)
        self.all_data = []
        self.line_charts = []
        self.data_adjuster = DataAdjustment()

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def calculate_monthly_reviews(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
            monthly_reviews = sorted(data['Month'].value_counts().to_dict().items())
            self.all_data.append(monthly_reviews)

    def create_line_charts(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        xlabels = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                   'October',
                   'November', 'December', '']
        ylabels = ['', '2016', '2015', '2014', '2013', '2012', '']
        colours = ['b', 'r', 'y', 'k', 'm', 'c']

        x_axis_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        y_axis_points = [1, 2, 3, 4, 5]
        colour_count = 0

        for y_point in y_axis_points:
            previous_x = 1
            previous_y = y_point
            for x_point in x_axis_points:
                plt.plot((x_point, previous_x), (previous_y, y_point), c='k', zorder=1)
            colour_count += 1

        colour_count = 0
        y_count = len(self.all_data)
        for year in self.all_data:
            for key, value in year:
                plt.scatter(key, y_count, s=self.clamp(value, 50, 4000), c=colours[colour_count], zorder=2)
            y_count -= 1
            colour_count += 1

        # And a corresponding grid
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(list(range(0, 14, 1)))
        plt.yticks(list(range(0, 7, 1)))
        plt.tick_params(pad=0)
        ax.set_xticklabels(
            xlabels, rotation=45, fontsize=20, va='top', ha='right')
        ax.set_yticklabels(
            ylabels, fontsize=20)

        plt.tight_layout()
        plt.show()

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
