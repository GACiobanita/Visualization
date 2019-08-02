from implementation.bar_chart_generator import BarChartGenerator
from implementation.file_receiver import FileReceiver

def main():
    bc_generator = BarChartGenerator()
    file_receiver = FileReceiver()

    file_receiver.acquire_input_path()
    file_receiver.acquire_output_path()

    bc_generator.acquire_csv_files(file_receiver.csv_files)
    bc_generator.calculate_text_length()
    bc_generator.create_overall_bar_charts()

    bc_generator.save_bar_charts(file_receiver.output_folder_path)

    # display bar charts
    while True:
        input_text = input("Would you like to see the bar charts? (Y/N) :")
        if input_text == 'Y':
            bc_generator.display_bar_charts()
            break
        else:
            if input_text == 'N':
                print("Shutting down.")
                break
            else:
                print("I didn't catch that, try again.")


if __name__ == "__main__":
    main()