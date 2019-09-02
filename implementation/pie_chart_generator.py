import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from implementation.data_adjustments import DataAdjustment


# data of the pie chart will be separated in sections
# each section presents a cut in the pie chart
class ChartSection(object):

    def __init__(self):
        # the total count of reviews for the section
        self.section_count = 0
        # allocate each review based on the review's score
        self.section_star_count = {5.0: 0, 4.0: 0, 3.0: 0, 2.0: 0, 1.0: 0}
        self.section_topic_count = {}
        for star_key in self.section_star_count.keys():
            self.section_topic_count[star_key] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def update_section_count(self):
        self.section_count += 1

    def update_review_star_count(self, key):
        if key is not str(np.nan):
            self.section_star_count[key] += 1

    def update_topic_count(self, rating_key, topic_key):
        self.section_topic_count[rating_key][topic_key] = self.section_topic_count.get(rating_key, {}).get(topic_key,
                                                                                                           0) + 1

    def __str__(self):
        return "Section Count:" + str(self.section_count) + "\n " + "Star Count:" + str(
            self.section_star_count)


# all the charts in the proposed pie charts
# each pie chart will be divided into 3 section: positive, negative and neutral
# each section with it's review counts and review allocation based on their score
class ChartData(object):

    def __init__(self):
        self.positive_chart_section = ChartSection()
        self.negative_chart_section = ChartSection()
        self.neutral_chart_section = ChartSection()

    # compound value is calculated by VADER , and the number that it is compared against
    # is the recommended number to use on their Github repository for sentiment allocation of text
    def sentiment_and_rating_classification(self, compound, rating_key, topic_key):
        if float(compound) >= 0.05:
            self.positive_chart_section.update_section_count()
            self.positive_chart_section.update_review_star_count(rating_key)
            self.positive_chart_section.update_topic_count(rating_key, topic_key)
        elif float(compound) <= -0.05:
            self.negative_chart_section.update_section_count()
            self.negative_chart_section.update_review_star_count(rating_key)
            self.negative_chart_section.update_topic_count(rating_key, topic_key)
        else:
            self.neutral_chart_section.update_section_count()
            self.neutral_chart_section.update_review_star_count(rating_key)
            self.neutral_chart_section.update_topic_count(rating_key, topic_key)

    def __str__(self):
        return "Positive:\n" + str(self.positive_chart_section) + "\n " + "Negative:\n" + str(
            self.negative_chart_section) + "\n " + "Neutral:\n" + str(self.neutral_chart_section)


class PieChartGenerator(object):

    def __init__(self):
        self.data_adjuster = DataAdjustment()

        self.csv_files = []
        self.basic_charts = []
        self.nested_charts = []
        self.chart_data = []

        self.CENTER_TEXT_FONT_SIZE = 30
        self.BOTTOM_TEXT_FONT_SIZE = 10

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def get_counts_from_csv_files(self):
        for file in self.csv_files:
            chart_data = ChartData()
            data = pd.read_csv(file)
            for index, row in data.iterrows():
                # from the csv file, classify ech review based on compound and rating
                if row['Rating'] > 0:
                    chart_data.sentiment_and_rating_classification(row['Compound'], row['Rating'],
                                                                   int(row['Dominant Topic']))
            self.chart_data.append((self.data_adjuster.get_year_from_string(file), chart_data))

    # basic pie chart contains the positive, negative and neutral slices
    def create_basic_pie_chart(self):
        for year, chart_data in self.chart_data:
            labels = 'Positive', 'Negative', 'Neutral'
            sizes = [chart_data.positive_chart_section.section_count,
                     chart_data.negative_chart_section.section_count,
                     chart_data.neutral_chart_section.section_count]

            positive_colour, negative_colour, neutral_colour = [plt.cm.Greens, plt.cm.Reds, plt.cm.Greys]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.pie(sizes, autopct='%1.1f%%',
                   colors=[positive_colour(0.6), negative_colour(0.6), neutral_colour(0.6)], textprops=dict(color="w"))
            plt.title('Distribution of sentiment in ' + year, fontsize=30, pad=10)
            self.create_legend(labels, 'Sentiment Legend:',
                               'upper right')
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

    # creates a nested pie chart in the shape of a donut with a sentiment outer later and a rating inner layer
    def create_nested_pie_chart_sentiment_and_rating(self):
        for year, chart_data in self.chart_data:
            groups_size = [chart_data.positive_chart_section.section_count,
                           chart_data.negative_chart_section.section_count,
                           chart_data.neutral_chart_section.section_count]

            # total to be displayed in the empty space of the "donut"
            total = chart_data.positive_chart_section.section_count + chart_data.negative_chart_section.section_count + chart_data.neutral_chart_section.section_count

            # the outer labels and inner labels respectively
            group_labels = self.create_labels_for_sentiment(chart_data, total)
            subgroup_labels = self.create_labels_for_ratings(chart_data)

            positive_size = self.get_rating_section_size(chart_data.positive_chart_section)
            negative_size = self.get_rating_section_size(chart_data.negative_chart_section)
            neutral_size = self.get_rating_section_size(chart_data.neutral_chart_section)

            subgroup_sizes = list(positive_size) + list(negative_size) + list(neutral_size)

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

            size = fig.get_size_inches() * fig.dpi  # get a vector of X and Y sizes, in pixels
            ax.annotate('Total',
                        xy=(size[0] / 2 + self.CENTER_TEXT_FONT_SIZE / 2, size[1] / 2 + self.CENTER_TEXT_FONT_SIZE),
                        xycoords='figure pixels', fontsize=self.CENTER_TEXT_FONT_SIZE,
                        horizontalalignment='center', verticalalignment='bottom')
            ax.annotate('Reviews:', xy=(size[0] / 2 + self.CENTER_TEXT_FONT_SIZE / 2, size[1] / 2),
                        xycoords='figure pixels', fontsize=self.CENTER_TEXT_FONT_SIZE,
                        horizontalalignment='center', verticalalignment='center')
            ax.annotate(str(total),
                        xy=(size[0] / 2 + self.CENTER_TEXT_FONT_SIZE / 2, size[1] / 2 - self.CENTER_TEXT_FONT_SIZE),
                        xycoords='figure pixels', fontsize=self.CENTER_TEXT_FONT_SIZE,
                        horizontalalignment='center', verticalalignment='top')

            plt.setp(inside_ring, width=0.4, edgecolor='white')
            plt.title('Distribution of sentiment and ratings in ' + year, fontsize=30, pad=30)
            self.create_legend(['Positive', 'Negative', 'Neutral'], 'Sentiment Legend:',
                               'upper right')
            self.nested_charts.append(fig)

    def create_nested_pie_chart_sentiment_and_topic(self, topic_names):
        for year, chart_data in self.chart_data:
            groups_size = [chart_data.positive_chart_section.section_count,
                           chart_data.negative_chart_section.section_count,
                           chart_data.neutral_chart_section.section_count]

            # total to be displayed in the empty space of the "donut"
            total = chart_data.positive_chart_section.section_count + chart_data.negative_chart_section.section_count + chart_data.neutral_chart_section.section_count
            print(total)

            # the outer labels and inner labels
            group_labels = self.create_labels_for_sentiment(chart_data, total)
            # print(group_labels)

            positive_data = dict(self.get_topic_section_size(chart_data.positive_chart_section))
            negative_data = dict(self.get_topic_section_size(chart_data.negative_chart_section))
            neutral_data = dict(self.get_topic_section_size(chart_data.neutral_chart_section))

            positive_colour, negative_colour, neutral_colour = [plt.cm.Greens, plt.cm.Reds, plt.cm.Greys]

            subgroup_sizes = list(positive_data.values()) + list(negative_data.values()) + list(neutral_data.values())
            subgroup_labels = list(self.get_new_labels(positive_data, 1, total)) + list(
                self.get_new_labels(neutral_data, 1, total)) + list(self.get_new_labels(negative_data, 1, total))
            print(subgroup_sizes)
            print(subgroup_labels)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('equal')
            outside_ring, outside_labels = ax.pie(groups_size, radius=1.3, labels=group_labels, labeldistance=0.9,
                                                  colors=[positive_colour(0.9), negative_colour(0.9),
                                                          neutral_colour(0.9)], textprops=dict(color="w"))
            for label in outside_labels:
                label.set_horizontalalignment('center')
                label.set_fontsize(15)

            for label in outside_labels:
                label.set_horizontalalignment('center')
                label.set_fontsize(15)

            plt.setp(outside_ring, width=0.3, edgecolor='white')

            # inside ring
            inside_ring, inside_labels, _ = ax.pie(subgroup_sizes, radius=1.3 - 0.3, labels=subgroup_labels,
                                                   autopct=self.autopct_label_value(1, total),
                                                   labeldistance=0.9,
                                                   colors=self.create_subgroup_colors(positive_colour,
                                                                                      positive_data.values(),
                                                                                      negative_colour,
                                                                                      negative_data.values(),
                                                                                      neutral_colour,
                                                                                      neutral_data.values()))
            for label in inside_labels:
                label.set_horizontalalignment('center')
                label.set_fontsize(12)

            size = fig.get_size_inches() * fig.dpi  # get a vector of X and Y sizes, in pixels
            ax.annotate('Total',
                        xy=(size[0] / 2 + self.CENTER_TEXT_FONT_SIZE / 2, size[1] / 2 + self.CENTER_TEXT_FONT_SIZE),
                        xycoords='figure pixels', fontsize=self.CENTER_TEXT_FONT_SIZE,
                        horizontalalignment='center', verticalalignment='bottom')
            ax.annotate('Reviews:', xy=(size[0] / 2 + self.CENTER_TEXT_FONT_SIZE / 2, size[1] / 2),
                        xycoords='figure pixels', fontsize=self.CENTER_TEXT_FONT_SIZE,
                        horizontalalignment='center', verticalalignment='center')
            ax.annotate(str(total),
                        xy=(size[0] / 2 + self.CENTER_TEXT_FONT_SIZE / 2, size[1] / 2 - self.CENTER_TEXT_FONT_SIZE),
                        xycoords='figure pixels', fontsize=self.CENTER_TEXT_FONT_SIZE,
                        horizontalalignment='center', verticalalignment='top')
            ax.annotate(
                'Labels for topics with a count under <1% are not displayed in the inner circle of the pie chart.',
                xy=(size[0] / 2, 0), xycoords='figure pixels',
                fontsize=self.BOTTOM_TEXT_FONT_SIZE, verticalalignment='bottom', horizontalalignment='center')

            plt.setp(inside_ring, width=0.4, edgecolor='white')
            plt.title('Distribution of sentiment and ratings in ' + year, fontsize=30, pad=30)
            sentiment_legend = self.create_legend(['Positive', 'Negative', 'Neutral'], 'Sentiment Legend:',
                                                  'upper right')
            topic_legend = self.create_legend(topic_names, 'Topics:',
                                              'lower right', False)
            plt.gca().add_artist(sentiment_legend)
            plt.show()
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
    def get_rating_section_size(chart_section):
        size_list = list()
        for item in chart_section.section_star_count:
            if chart_section.section_star_count[item] is not 0:
                size_list.append(chart_section.section_star_count[item])
        return size_list

    @staticmethod
    def get_topic_section_size(chart_section):
        size_list = {}
        for rating_key in chart_section.section_star_count.keys():
            if chart_section.section_star_count[rating_key] is not 0:
                for topic_key, topic_value in chart_section.section_topic_count[rating_key].items():
                    if topic_value > 0:
                        size_list[topic_key] = size_list.get(topic_key, 0) + topic_value
        return sorted(size_list.items(), key=lambda kv: kv[1], reverse=True)

    @staticmethod
    def create_labels_for_sentiment(chart_data, total):
        name_list = list()
        positive_percentage = chart_data.positive_chart_section.section_count / total * 100
        name_list.append(str(positive_percentage)[:4] + '%')
        negative_percentage = chart_data.negative_chart_section.section_count / total * 100
        name_list.append(str(negative_percentage)[:4] + '%')
        neutral_percentage = chart_data.neutral_chart_section.section_count / total * 100
        name_list.append(str(neutral_percentage)[:4] + '%')
        return name_list

    @staticmethod
    def create_labels_for_topics(chart_data):
        positive_name_list = {}
        neutral_name_list = {}
        negative_name_list = {}
        for rating_key in chart_data.positive_chart_section.section_star_count.keys():
            if chart_data.positive_chart_section.section_star_count[rating_key] is not 0:
                for topic_key, topic_value in sorted(
                        chart_data.positive_chart_section.section_topic_count[rating_key].items(), key=lambda kv: kv[1],
                        reverse=True):
                    if topic_value > 0:
                        positive_name_list[topic_key] = topic_value
        for rating_key in chart_data.neutral_chart_section.section_star_count.keys():
            if chart_data.neutral_chart_section.section_star_count[rating_key] is not 0:
                for topic_key, topic_value in sorted(
                        chart_data.neutral_chart_section.section_topic_count[rating_key].items(), key=lambda kv: kv[1],
                        reverse=True):
                    if topic_value > 0:
                        neutral_name_list[topic_key] = topic_value
        for rating_key in chart_data.negative_chart_section.section_star_count.keys():
            if chart_data.negative_chart_section.section_star_count[rating_key] is not 0:
                for topic_key, topic_value in sorted(
                        chart_data.negative_chart_section.section_topic_count[rating_key].items(), key=lambda kv: kv[1],
                        reverse=True):
                    if topic_value > 0:
                        negative_name_list[topic_key] = topic_value
        return list(list(positive_name_list.keys()) + list(neutral_name_list.keys()) + list(negative_name_list.keys()))

    @staticmethod
    def autopct_label_value(minimum_percentage, total):
        def inner_autopct(pct):
            return ('%.2f' % pct) if pct / total * 100 > minimum_percentage else ''

        return inner_autopct

    @staticmethod
    def get_new_labels(data_dict, minimum_percentage, total):
        new_labels = [key if value / total * 100 > minimum_percentage else '' for key, value in data_dict.items()]
        return new_labels

    @staticmethod
    def create_labels_for_ratings(chart_data):
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
    def create_subgroup_colors(positive_colour, positive_list, negative_colour, negative_list, neutral_colour,
                               neutral_list):
        overall_colour_list = list()
        shade = 0.7
        for size in positive_list:
            overall_colour_list.append(positive_colour(shade))
            shade -= 0.05
        shade = 0.7
        for size in negative_list:
            overall_colour_list.append(negative_colour(shade))
            shade -= 0.05
        shade = 0.7
        for size in neutral_list:
            overall_colour_list.append(neutral_colour(shade))
            shade -= 0.05
        return overall_colour_list

    @staticmethod
    # simply creates a legend, in the top right corner of the visualization
    # with the colours used in the pie chart for descriptions
    # (0,0) is bottom left, (1,1) is top rights
    def create_legend(labels, title, loc, use_handles=True):
        if use_handles is False:
            legend = plt.legend(labels, loc=loc, title=title, title_fontsize=16, frameon=False, fontsize=12,
                                handlelength=0,
                                handletextpad=0)
        else:
            legend = plt.legend(labels, loc=loc, title=title, title_fontsize=16, frameon=False, fontsize=12)
        return legend
