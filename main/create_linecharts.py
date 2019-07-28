from implementation.line_chart_generator import LineChartGenerator
from implementation.file_receiver import FileReceiver

def main():
    lc_generator = LineChartGenerator()
    file_receiver = FileReceiver()

    file_receiver.acquire_input_path()
    file_receiver.acquire_output_path()

    lc_generator.acquire_csv_files(file_receiver.csv_files)
    lc_generator.calculate_yearly_reviews()
    lc_generator.calculate_app_review()

    lc_generator.save_bar_charts(file_receiver.output_folder_path)

    # display word clouds
    while True:
        input_text = input("Would you like to see the word clouds? (Y/N) :")
        if input_text == 'Y':
            lc_generator.display_bar_charts()
            break
        else:
            if input_text == 'N':
                print("Shutting down.")
                break
            else:
                print("I didn't catch that, try again.")


if __name__ == "__main__":
    main()