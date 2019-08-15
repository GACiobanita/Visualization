from implementation.line_chart_generator import LineChartGenerator
from implementation.file_receiver import FileReceiver

def main():
    lc_generator = LineChartGenerator()
    file_receiver = FileReceiver()

    file_receiver.acquire_input_path()
    file_receiver.acquire_output_path()

    lc_generator.acquire_csv_files(file_receiver.csv_files)
    lc_generator.calculate_overall_reviews_by_identifiable_individuals()
    lc_generator.calculate_reviews_by_identifiable_individuals_per_app()
    lc_generator.create_all_year_data_chart()
    lc_generator.create_yearly_app_data_charts()

    lc_generator.save_yearly_app_charts(file_receiver.output_folder_path)
    lc_generator.save_all_year_charts(file_receiver.output_folder_path)

    # display line charts
    while True:
        input_text = input("Would you like to see the line charts? (Y/N) :")
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