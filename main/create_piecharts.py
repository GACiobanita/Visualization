from implementation.pie_chart_generator import PieChartGenerator
from implementation.file_receiver import FileReceiver


def main():
    file_receiver = FileReceiver()
    pie_chart_generator = PieChartGenerator()

    file_receiver.acquire_input_path()
    file_receiver.acquire_output_path()

    pie_chart_generator.acquire_csv_files(file_receiver.csv_files)
    pie_chart_generator.get_counts_from_csv_files()
    pie_chart_generator.create_basic_pie_chart()
    pie_chart_generator.create_nested_pie_chart_sentiment_and_rating()

    pie_chart_generator.save_basic_pie_charts(file_receiver.output_folder_path)
    pie_chart_generator.save_nested_pie_chart(file_receiver.output_folder_path)

    # display line charts
    while True:
        input_text = input("Would you like to see the basic pie charts? (Y/N) :")
        if input_text == 'Y':
            pie_chart_generator.display_basic_pie_chart()
            pie_chart_generator.display_nested_pie_chart()
            break
        else:
            if input_text == 'N':
                print("Shutting down.")
                break
            else:
                print("I didn't catch that, try again.")


if __name__ == "__main__":
    main()
