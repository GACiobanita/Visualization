import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .data_adjustments import DataAdjustment
import os

class ChartSection(object):

    def __init__(self):
        self.total_section_count = 0
        self.section_review_star_count = {5.0: 0, 4.0: 0, 3.0: 0, 2.0: 0, 1.0: 0}

    def update_section(self, count, key, value):
        self.total_section_count += count
        self.section_review_star_count[key] += value

class PieChartGenerator(object):

    def __init__(self):
        self.csv_files = []
        self.positive_review_count = 0
        self.negative_review_count = 0
        self.neutral_review_count = 0

    def acquire_csv_files(self, csv_files):
        self.csv_files = csv_files

    def get_counts_from_csv_files(self):
        for file in self.csv_files:
            data = pd.DataFrame(file)

    def sentiment_classification(self, compound):
        if compound >= 0.05:
            self.positive_review_count += 1
        elif compound <= -0.05:
            self.negative_review_count += 1
        else:
            self.neutral_review_count += 1

    def review_score_classification(self, score):
        self.star_count[score] += 1

