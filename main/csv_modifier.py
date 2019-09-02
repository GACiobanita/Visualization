from implementation.sentiment_analyzer import SentimentAnalyzer
from implementation.file_receiver import FileReceiver
from implementation.data_adjustments import DataAdjustment
import os.path
import pandas as pd


def main():
    file_receiver = FileReceiver()
    sentiment_analyzer = SentimentAnalyzer()
    data_adjuster = DataAdjustment()

    input_text = input("Calculate Sentiment for text in csv(S) or Add column from existing csv file to another(C):")

    if input_text == 'S':

        file_receiver.acquire_input_path()
        file_receiver.acquire_output_path()
        sentiment_analyzer.acquire_csv_files(file_receiver.csv_files)
        sentiment_analyzer.create_data_frames_with_result_columns()
        sentiment_analyzer.save_sentiment_csv_file(file_receiver.output_folder_path)

    elif input_text == "C":

        print("Input path to initial set:")
        file_receiver.acquire_input_path()
        first_set = file_receiver.csv_files
        print("Input path to second set:")
        file_receiver.acquire_input_path()
        second_set = file_receiver.csv_files

        file_receiver.acquire_output_path()

        first_set_files = []
        second_set_files = []
        first_folder = []
        for i in range(0, len(first_set)):
            first_head, first_tail = os.path.split(first_set[i])
            first_folder = first_head
            second_head, second_tail = os.path.split(second_set[i])
            first_set_files.append(first_tail)
            second_set_files.append(second_tail)

        for file in first_set_files:
            second_set_index = second_set_files.index(file)
            first_data = pd.read_csv(first_folder + '\\' + file)
            second_data = pd.read_csv(second_set[second_set_index])
            data_adjuster.concatenate_csv_data(file_receiver.output_folder_path, file, first_data, second_data,
                                               "Main Topic", "Main Topic")


if __name__ == "__main__":
    main()
