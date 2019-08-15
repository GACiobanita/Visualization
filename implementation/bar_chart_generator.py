import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from .data_adjustments import DataAdjustment
from .sentiment_analyzer import SentimentAnalyzer
import os


# Data type specifically created to contain positive and negative valence words/emoticons
# per file
class FileValenceData(object):

    def __init__(self):
        self.positive_valence_words = {}
        self.negative_valence_words = {}
        self.positive_valence_emoticons = {}
        self.negative_valence_emoticons = {}
        self.data_adjuster = DataAdjustment()

    def get_word_valence(self, word_data):
        sorted_word_data = sorted(word_data.items(), key=lambda kv: kv[1], reverse=True)
        for key, value in sorted_word_data:
            # get the valence score for the key, each key is a word in the VADER lexicon
            # the value is the number of appearances of the key in the text
            valence = self.data_adjuster.get_valence_from_lexicon(key)
            if valence > 0:
                self.positive_valence_words[key] = (value, valence)
            else:
                self.negative_valence_words[key] = (value, valence)

    def get_emoticon_valence(self, emoticon_data):
        sorted_emoticon_data = sorted(emoticon_data.items(), key=lambda kv: kv[1], reverse=True)
        for key, value in sorted_emoticon_data:
            valence = self.data_adjuster.get_valence_from_lexicon(key)
            if valence > 0:
                self.positive_valence_emoticons[key] = (value, valence)
            else:
                self.positive_valence_emoticons[key] = (value, valence)


class BarChartGenerator(object):

    def __init__(self):
        self.csv_files = []
        self.total_word_count = {'0-50': 0, '51-100': 0, '101-200': 0, '201-300': 0, '301-400': 0, '400+': 0}
        self.per_file_word_count = []
        self.bar_charts = []
        self.data_adjuster = DataAdjustment()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.file_valence_data = []
        self.divergent_bar_charts = []

        self.file_rating_data = []

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def categorize_ratings(self):
        for file in self.csv_files:
            data = pd.read_csv(file)
            token_data = data['Rating'].value_counts().to_dict()
            positive_compound_data = data[data['Compound'] >= 0.5].groupby('Rating')['Compound'].count().to_dict()
            self.check_dict_for_key(positive_compound_data, token_data.keys())
            negative_compound_data = data[data['Compound'] <= -0.5].groupby('Rating')['Compound'].count().to_dict()
            self.check_dict_for_key(negative_compound_data, token_data.keys())
            neutral_compound_data = data[(data['Compound'] > -0.50) & (data['Compound'] < 0.5)].groupby('Rating')[
                'Compound'].count().to_dict()
            self.check_dict_for_key(neutral_compound_data, token_data.keys())
            year = self.data_adjuster.get_year_from_string(file)
            file_name = self.data_adjuster.get_file_name_from_path(file)
            self.file_rating_data.append(
                (file_name, year, token_data, positive_compound_data, negative_compound_data, neutral_compound_data))

    # classify each review in the csv_files into different categories based on the number of words contained
    def categorize_text_by_word_count(self):
        for file in self.csv_files:
            current_word_count = {'0-50': 0, '51-100': 0, '101-200': 0, '201-300': 0, '301-400': 0, '400+': 0}
            data = pd.read_csv(file)
            for text in data['Text']:
                text = str(text)
                text = self.data_adjuster.remove_string_punctuation(text)
                self.allocate_review_to_count_category(len(text), current_word_count)
            self.per_file_word_count.append((self.data_adjuster.get_year_from_string(file), current_word_count))
        self.calculate_total_word_count_of_files()

    def calculate_total_word_count_of_files(self):
        self.total_word_count['0-50'] = sum([item['0-50'] for year, item in self.per_file_word_count])
        self.total_word_count['51-100'] = sum([item['51-100'] for year, item in self.per_file_word_count])
        self.total_word_count['101-200'] = sum([item['51-100'] for year, item in self.per_file_word_count])
        self.total_word_count['201-300'] = sum([item['51-100'] for year, item in self.per_file_word_count])
        self.total_word_count['301-400'] = sum([item['51-100'] for year, item in self.per_file_word_count])
        self.total_word_count['400+'] = sum([item['400+'] for year, item in self.per_file_word_count])

    def categorize_words_by_valence(self):
        for file in self.csv_files:
            csv_data = pd.read_csv(file)
            word_usage = {}
            emoticon_usage = {}
            for index, row in csv_data.iterrows():
                text, score, negation_usage, word_and_emoticon_usage = self.sentiment_analyzer.calculate_text_score_and_word_appearance(
                    str(row['Text']))
                separated_emoticon_usage, separated_word_usage = self.data_adjuster.separate_emoticons_and_words(
                    word_and_emoticon_usage)
                word_usage = self.data_adjuster.merge_dictionaries(word_usage, separated_word_usage)
                emoticon_usage = self.data_adjuster.merge_dictionaries(emoticon_usage, separated_emoticon_usage)

            temp_file_valence = FileValenceData()
            temp_file_valence.get_word_valence(word_usage)
            temp_file_valence.get_emoticon_valence(emoticon_usage)
            self.file_valence_data.append((self.data_adjuster.get_year_from_string(file), temp_file_valence))

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
        self.create_horizontal_bar_chart(self.total_word_count, "Total Word Usage 2012-2016")
        # create a bar chart for each file
        for year, dictionary in self.per_file_word_count:
            self.create_horizontal_bar_chart(dictionary, "Total Word Usage for " + str(year))

    def display_bar_charts(self):
        for chart in self.bar_charts:
            chart.show()

    def display_divergent_bar_charts(self):
        for chart in self.divergent_bar_charts:
            chart.show()

    def create_horizontal_bar_chart(self, dictionary, title):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

        # data
        bar_height = []
        bar_name = []
        for key, value in dictionary.items():
            # bar height is the value of the key, in this case the total count
            bar_height.append(value)
            # and the name of the bar is the key itself
            bar_name.append(key)

        y_pos = np.arange(len(bar_name))  # is the number of items in the dict

        # bar charts are created on a 2D plane , X and Y
        # I am creating horizontal bar charts here
        ax.barh(y_pos, bar_height, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_name)
        ax.invert_yaxis()
        ax.set_ylabel('Words Per Rating')
        ax.set_xlabel('Number of reviews')
        ax.set_title(title)

        # create a line in the bar chart to show the location of the average word count
        ax.axvline(np.mean(list(dictionary.values())), color='red',
                   label='Average of categories: ' + str(int(np.mean(list(dictionary.values())))))
        print(str(int(np.mean(list(dictionary.values())))))

        # create a legend explaining the average
        # loc 0 places the legend in the 'best' location where there would be minimum overlap with the chart
        plt.legend(loc=0)

        self.bar_charts.append(fig)

    def create_rating_bar_charts(self):
        for file_name, year, token_data, positive_compound, negative_compound, neutral_compound in self.file_rating_data:
            st = plt.suptitle(file_name + ' ' + year)
            st.set_y(0.95)
            fig = plt.figure(1)
            fig.set_figheight(20)
            fig.set_figwidth(15)
            plt.subplot(121)
            x_labels = []
            x_height = []
            total_ratings = 0

            for key in sorted(token_data, reverse=True):
                x_labels.append(int(key))
                x_height.append(token_data[key])
                total_ratings += token_data[key]

            bar_chart = plt.bar(x_labels, x_height)
            plt.xlabel('Ratings')
            plt.ylabel('Number of Ratings', rotation=90)

            for rect in bar_chart:
                height = rect.get_height()
                percentage = str(int(height) / total_ratings * 100)
                plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                         percentage[:4] + '%(' + '%d' % int(height) + ')',
                         ha='center', va='bottom')

            plt.subplot(122)
            positive_values = sorted(list(positive_compound.values()), reverse=True)
            negative_values = sorted(list(negative_compound.values()), reverse=True)
            neutral_values = sorted(list(neutral_compound.values()), reverse=True)
            positive_stacked_chart = plt.bar(x_labels, positive_values, color='g')
            negative_stacked_chart = plt.bar(x_labels, negative_values, color='r', bottom=positive_values)
            neutral_stacked_chart = plt.bar(x_labels, neutral_values,
                                            bottom=np.array(positive_values) + np.array(negative_values), color='k')
            plt.xlabel('Ratings')
            plt.subplots_adjust(top=0.85)
            plt.show()

    # these are bar charts that go in separate directions
    # they are still horizontal
    def create_divergent_valence_bar_chart(self):
        # a list of (year[0], file_valence_data[1])
        for valence_data in self.file_valence_data:
            fig = plt.figure()
            ax = fig.add_subplot()

            file_year = valence_data[0]
            current_valence_data = valence_data[1]

            positive_section = []
            for key, (freq, valence) in current_valence_data.positive_valence_words.items():
                if len(positive_section) >= 10:
                    break
                else:
                    if valence > 0:
                        # it is freq * valence as I also want to focus on
                        # the impact the word has due to it's valence score
                        positive_section.append((key, freq * valence))

            negative_section = []
            for key, (freq, valence) in current_valence_data.negative_valence_words.items():
                if len(negative_section) >= 10:
                    break
                else:
                    if valence < 0:
                        # it is freq * valence as I also want to focus on the
                        # impact the word has due to it's valence score
                        negative_section.append((key, (freq * valence)))

            # I am sorting the sections of the bar charts
            # around their freq * valence score in order the positive section in descending order
            # and the negative section in ascending order
            sorted_positive_section = sorted(positive_section, key=lambda kv: kv[1])
            sorted_negative_section = sorted(negative_section, key=lambda kv: kv[1])

            positive_labels = []
            positive_bar_height = []
            # labels are the key value( the word )
            # height is the value of freq * valence
            for key, value in sorted_positive_section:
                positive_labels.append(key)
                positive_bar_height.append(value)
            negative_labels = []
            negative_bar_height = []
            for key, value in sorted_negative_section:
                negative_labels.append(key)
                negative_bar_height.append(value)

            # get all labels
            y_labels = negative_labels + positive_labels
            # Y position in 2D based on the number of labels
            y_pos = np.arange(len(y_labels))
            bar_height = negative_bar_height + positive_bar_height

            positive_colour, negative_colour = [plt.cm.Greens, plt.cm.Reds]

            ax.barh(y_pos, bar_height, align='center',
                    color=self.divergent_bar_chart_colors(positive_colour, negative_colour, len(positive_labels),
                                                          len(negative_labels)))
            ax.set_yticks(y_pos)
            ax.set_title("Most popular positive/negative words in reviews for " + file_year)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel('Word frequency * Valence Value')
            self.create_legend(['Positive', 'Negative'], positive_colour, negative_colour)
            self.divergent_bar_charts.append(fig)

    @staticmethod
    # for each element in the positive and negative list we give a colour
    def divergent_bar_chart_colors(positive_color, negative_color, positive_count, negative_count):
        positive_colour_list = []
        negative_colour_list = []
        for i in range(0, positive_count):
            positive_colour_list.append(positive_color(0.6))
        for i in range(0, negative_count):
            negative_colour_list.append(negative_color(0.6))
        return negative_colour_list + positive_colour_list

    @staticmethod
    def create_legend(labels, positive_color, negative_color):
        positive_patch = mpatches.Patch(color=positive_color(0.6), label='Positive')
        negative_patch = mpatches.Patch(color=negative_color(0.6), label='Negative')
        plt.legend(handles=[positive_patch, negative_patch])

    def save_bar_charts(self, output_folder_path):
        count = 0
        self.bar_charts[0].savefig(
            output_folder_path + "\\" + "fraud_apps_640_review_info_final_total_word_usage_bar_chart.png")
        for chart in self.bar_charts[1:]:
            head, tail = os.path.split(self.csv_files[count])
            chart.savefig(output_folder_path + "\\" + tail[:-4] + "_bar_chart.png")
            count += 1

    def save_divergent_bar_charts(self, output_folder_path):
        count = 0
        for chart in self.divergent_bar_charts:
            head, tail = os.path.split(self.csv_files[count])
            chart.savefig(output_folder_path + "\\" + tail[:-4] + "_divergent_bar_chart.png")
            count += 1

    @staticmethod
    def check_dict_for_key(data_dict, key_list):
        for key in key_list:
            if key not in data_dict:
                data_dict[key] = 0
