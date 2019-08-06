import matplotlib.pyplot as plt
import matplotlib.lines as Line2D
import pandas as pd
import os
from implementation.data_adjustments import DataAdjustment
from implementation.sentiment_analyzer import SentimentAnalyzer


class ChartSection(object):

    def __init__(self):
        self.section_count = 0
        self.section_star_count = {5.0: 0, 4.0: 0, 3.0: 0, 2.0: 0, 1.0: 0}

    def update_section_count(self):
        self.section_count += 1

    def update_review_star_count(self, key):
        self.section_star_count[key] += 1

    def __str__(self):
        return "Section Count:" + str(self.section_count) + "\n " + "Star Count:" + str(
            self.section_star_count)


class ChartData(object):

    def __init__(self):
        self.positive_chart_section = ChartSection()
        self.negative_chart_section = ChartSection()
        self.neutral_chart_section = ChartSection()

    def sentiment_and_rating_classification(self, compound, key):
        if float(compound) >= 0.05:
            self.positive_chart_section.update_section_count()
            self.positive_chart_section.update_review_star_count(key)
        elif float(compound) <= -0.05:
            self.negative_chart_section.update_section_count()
            self.negative_chart_section.update_review_star_count(key)
        else:
            self.neutral_chart_section.update_section_count()
            self.neutral_chart_section.update_review_star_count(key)

    def __str__(self):
        return "Positive:\n" + str(self.positive_chart_section) + "\n " + "Negative:\n" + str(
            self.negative_chart_section) + "\n " + "Neutral:\n" + str(self.neutral_chart_section)


class PieChartGenerator(object):

    def __init__(self):
        self.data_adjuster = DataAdjustment()

        self.csv_files = []
        self.sentiment_analyzer = SentimentAnalyzer()
        self.basic_charts = []
        self.nested_charts = []
        self.chart_data = []

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def get_counts_from_csv_files(self):
        for file in self.csv_files:
            chart_data = ChartData()
            data = pd.read_csv(file)
            for index, row in data.iterrows():
                chart_data.sentiment_and_rating_classification(row['Compound'], row['Rating'])
            self.chart_data.append((self.data_adjuster.get_year_from_string(file), chart_data))

    def create_basic_pie_chart(self):
        for year, chart_data in self.chart_data:
            labels = 'Positive', 'Negative', 'Neutral'
            sizes = [chart_data.positive_chart_section.section_count,
                     chart_data.negative_chart_section.section_count,
                     chart_data.neutral_chart_section.section_count]
            explode = (0, 0, 0)  # explode means removing a part from the pie chart to make it more pronounced

            positive_colour, negative_colour, neutral_colour = [plt.cm.Greens, plt.cm.Reds, plt.cm.Greys]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.pie(sizes, explode=explode, autopct='%1.1f%%', startangle=0,
                   colors=[positive_colour(0.6), negative_colour(0.6), neutral_colour(0.6)], textprops=dict(color="w"))
            plt.title('Distribution of sentiment in ' + year, fontsize=30, pad=10)
            self.create_legend(labels)
            self.basic_charts.append(fig)

    def display_basic_pie_chart(self):
        for chart in self.basic_charts:
            chart.show()

    def save_basic_pie_charts(self, output_folder_path):
        count = 0
        for chart in self.basic_charts:
            head, tail = os.path.split(self.csv_files[count])
            chart.savefig(output_folder_path + "\\" + tail[:-4] + "_basic_pie_chart.png")
            count += 1

    def create_nested_pie_chart(self):
        for year, chart_data in self.chart_data:
            groups_size = [chart_data.positive_chart_section.section_count,
                           chart_data.negative_chart_section.section_count,
                           chart_data.neutral_chart_section.section_count]
            total = chart_data.positive_chart_section.section_count + chart_data.negative_chart_section.section_count + chart_data.neutral_chart_section.section_count

            group_labels = self.create_group_percentage_labels(chart_data, total)
            subgroup_labels = self.create_subgroup_labels(chart_data)

            positive_size = self.create_subgroup_sizes(chart_data.positive_chart_section)
            negative_size = self.create_subgroup_sizes(chart_data.negative_chart_section)
            neutral_size = self.create_subgroup_sizes(chart_data.neutral_chart_section)

            subgroup_sizes = self.create_overall_subgroup_size_list(positive_size, negative_size, neutral_size)

            positive_colour, negative_colour, neutral_colour = [plt.cm.Greens, plt.cm.Reds, plt.cm.Greys]
            # outside ring
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('equal')
            outside_ring, outside_labels = ax.pie(groups_size, radius=1.3, labels=group_labels, labeldistance=0.9,
                                                  colors=[positive_colour(0.6), negative_colour(0.6),
                                                          neutral_colour(0.6)], textprops=dict(color="w"))
            for label in outside_labels:
                label.set_horizontalalignment('center')
                label.set_fontsize(15)

            plt.setp(outside_ring, width=0.3, edgecolor='white')

            # inside ring
            inside_ring, inside_labels = ax.pie(subgroup_sizes, radius=1.3 - 0.3, labels=subgroup_labels,
                                                labeldistance=0.9,
                                                colors=self.create_subgroup_colors(positive_colour, positive_size,
                                                                                   negative_colour,
                                                                                   negative_size, neutral_colour,
                                                                                   neutral_size))

            for label in inside_labels:
                label.set_horizontalalignment('center')
                label.set_fontsize(12)

            size = fig.get_size_inches() * fig.dpi
            ax.annotate('Total', xy=(size[0] / 2 + 30 / 2, size[1] / 2 + 30), xycoords='figure pixels', fontsize=30,
                        horizontalalignment='center', verticalalignment='bottom')
            ax.annotate('Reviews:', xy=(size[0] / 2 + 30 / 2, size[1] / 2), xycoords='figure pixels', fontsize=30,
                        horizontalalignment='center', verticalalignment='center')
            ax.annotate(str(total), xy=(size[0] / 2 + 30 / 2, size[1] / 2 - 30), xycoords='figure pixels', fontsize=30,
                        horizontalalignment='center', verticalalignment='top')

            plt.setp(inside_ring, width=0.4, edgecolor='white')
            plt.title('Distribution of sentiment and ratings in ' + year, fontsize=30, pad=30)
            self.create_legend(['Positive', 'Negative', 'Neutral'])
            self.nested_charts.append(fig)

    def display_nested_pie_chart(self):
        for chart in self.nested_charts:
            chart.show()

    def save_nested_pie_chart(self, output_folder_path):
        count = 0
        for chart in self.nested_charts:
            head, tail = os.path.split(self.csv_files[count])
            chart.savefig(output_folder_path + "\\" + tail[:-4] + "_nested_pie_chart.png")
            count += 1

    @staticmethod
    def create_subgroup_sizes(chart_section):
        size_list = list()
        for item in chart_section.section_star_count:
            if chart_section.section_star_count[item] is not 0:
                size_list.append(chart_section.section_star_count[item])
        return size_list

    @staticmethod
    def create_group_percentage_labels(chart_data, total):
        name_list = list()
        positive_percentage = chart_data.positive_chart_section.section_count / total * 100
        name_list.append(str(positive_percentage)[:4] + '%')
        negative_percentage = chart_data.negative_chart_section.section_count / total * 100
        name_list.append(str(negative_percentage)[:4] + '%')
        neutral_percentage = chart_data.neutral_chart_section.section_count / total * 100
        name_list.append(str(neutral_percentage)[:4] + '%')
        return name_list

    @staticmethod
    def create_subgroup_labels(chart_data):
        name_list = list()
        for item in chart_data.positive_chart_section.section_star_count:
            if chart_data.positive_chart_section.section_star_count[item] is not 0:
                name_list.append(str(int(item)))
        for item in chart_data.negative_chart_section.section_star_count:
            if chart_data.negative_chart_section.section_star_count[item] is not 0:
                name_list.append(str(int(item)))
        for item in chart_data.neutral_chart_section.section_star_count:
            if chart_data.neutral_chart_section.section_star_count[item] is not 0:
                name_list.append(str(int(item)))
        return name_list

    @staticmethod
    def create_overall_subgroup_size_list(positive_list, negative_list, neutral_list):
        overall_size_list = list()
        for size in positive_list:
            overall_size_list.append(size)
        for size in negative_list:
            overall_size_list.append(size)
        for size in neutral_list:
            overall_size_list.append(size)
        return overall_size_list

    @staticmethod
    def create_subgroup_colors(positive_colour, positive_list, negative_colour, negative_list, neutral_colour,
                               neutral_list):
        overall_colour_list = list()
        shade = 0.5
        for size in positive_list:
            overall_colour_list.append(positive_colour(shade))
            shade -= 0.1
        shade = 0.5
        for size in negative_list:
            overall_colour_list.append(negative_colour(shade))
            shade -= 0.1
        shade = 0.5
        for size in neutral_list:
            overall_colour_list.append(neutral_colour(shade))
            shade -= 0.1
        return overall_colour_list

    @staticmethod
    def create_legend(labels):
        plt.legend(labels, loc=(0.75, 0.75), title='Sentiment Legend:', title_fontsize=16, frameon=False, fontsize=12)
