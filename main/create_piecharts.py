from implementation.pie_chart_generator import PieChartGenerator
from implementation.file_receiver import FileReceiver
import os
import pandas as pd


def get_csv_files_from_directories(csv_directory):
    file_extension = ".csv"
    filenames = os.listdir(csv_directory)
    for filename in filenames:
        if filename.endswith(file_extension):
            yield csv_directory + "\\" + filename


def main():
    file_receiver = FileReceiver()
    pie_chart_generator = PieChartGenerator()

    pie_type = input("Basic Pie Chart(B), Nested Pie Chart(Sentiment-Rating=SR,Sentiment-Topic=ST, Rating-Topic=RT): ")

    if pie_type == 'B':
        file_receiver.acquire_input_path()
        file_receiver.acquire_output_path()

        pie_chart_generator.acquire_csv_files(file_receiver.csv_files)
        pie_chart_generator.get_chart_data_from_csv_files()
        pie_chart_generator.create_basic_pie_chart()
        pie_chart_generator.save_basic_pie_charts(file_receiver.output_folder_path)
    elif pie_type == 'SR':
        file_receiver.acquire_input_path()
        file_receiver.acquire_output_path()

        pie_chart_generator.acquire_csv_files(file_receiver.csv_files)
        pie_chart_generator.get_chart_data_from_csv_files()
        pie_chart_generator.create_nested_pie_chart_sentiment_and_rating()
        pie_chart_generator.save_nested_pie_chart(file_receiver.output_folder_path)
    elif pie_type == 'ST':
        input_folders = [
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2012"#,
            #"D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2013",
            #"D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2014",
            #"D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2015",
            #"D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2016"
        ]
        output_folders = [
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012\\Sentiment-Topic Nested Pie Charts"#,
            #"D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2013\\Sentiment-Topic Nested Pie Charts",
            #"D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2014\\Sentiment-Topic Nested Pie Charts",
            #"D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2015\\Sentiment-Topic Nested Pie Charts",
            #"D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2016\\Sentiment-Topic Nested Pie Charts"
        ]

        for i in range(0, len(input_folders)):
            filenames = get_csv_files_from_directories(input_folders[i])
            file_list = []
            for filename in filenames:
                file_list.append(filename)
            pie_chart_generator.acquire_csv_files(file_list)
            pie_chart_generator.get_chart_data_from_csv_files(True)
            pie_chart_generator.create_nested_pie_chart_sentiment_and_topic()
            pie_chart_generator.save_nested_pie_chart(output_folders[i])

    elif pie_type == 'RT':
        input_folders = [
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2012",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2013",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2014",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2015",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\Sentiment & Topic\\2016"
        ]
        output_folders = [
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012\\Rating-Topic Nested Pie Charts",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2013\\Rating-Topic Nested Pie Charts",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2014\\Rating-Topic Nested Pie Charts",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2015\\Rating-Topic Nested Pie Charts",
            "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2016\\Rating-Topic Nested Pie Charts"
        ]

        for i in range(0, len(input_folders)):
            filenames = get_csv_files_from_directories(input_folders[i])
            file_list = []
            for filename in filenames:
                file_list.append(filename)
            pie_chart_generator.acquire_csv_files(file_list)
            pie_chart_generator.get_chart_data_from_csv_files(True)
            pie_chart_generator.create_nested_pie_chart_rating_and_topic()
            pie_chart_generator.save_nested_pie_chart(output_folders[i])


if __name__ == "__main__":
    main()
