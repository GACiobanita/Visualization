from implementation.word_cloud_generator import WordCloudGenerator
from implementation.file_receiver import FileReceiver


def main():
    wc_generator = WordCloudGenerator()
    file_receiver = FileReceiver()

    # acquire paths
    if len(wc_generator.csv_files) == 0:
        file_receiver.acquire_input_path()
    file_receiver.acquire_output_path()

    # create word clouds
    wc_generator.acquire_csv_files(file_receiver.csv_files)
    figure_choice = input("Single figure-multiple word clouds(M) or each word clouds separate(S)?\n")
    if figure_choice == 'M':
        wc_generator.create_dictionaries_from_topics()
        wc_generator.create_figure_with_multiple_word_clouds()
        #save word clouds
        wc_generator.save_figure(file_receiver.output_folder_path)
    else:
        column_name = input("What is the column name from which you want to extract text frequency?\n")
        wc_generator.create_dictionaries(column_name)
        wc_generator.create_word_cloud()
        # save word clouds
        wc_generator.save_word_cloud(file_receiver.output_folder_path)

    # display word clouds
    while True:
        input_text = input("Would you like to see the word clouds? (Y/N) :")
        if input_text == 'Y':
            wc_generator.display_word_cloud()
            break
        else:
            if input_text == 'N':
                print("Shutting down.")
                break
            else:
                print("I didn't catch that, try again.")


if __name__ == "__main__":
    main()
