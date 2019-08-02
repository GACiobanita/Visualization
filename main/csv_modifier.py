from implementation.sentiment_analyzer import SentimentAnalyzer
from implementation.file_receiver import FileReceiver


def main():
    file_receiver = FileReceiver()
    sentiment_analyzer = SentimentAnalyzer()

    file_receiver.acquire_input_path()
    file_receiver.acquire_output_path()

    sentiment_analyzer.acquire_csv_files(file_receiver.csv_files)
    sentiment_analyzer.create_data_frames_with_result_columns()
    sentiment_analyzer.save_sentiment_csv_file(file_receiver.output_folder_path)


if __name__ == "__main__":
    main()
