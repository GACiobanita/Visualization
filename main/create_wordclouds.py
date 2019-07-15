from implementation.wordcloud_generator import WordCloudGenerator

def main():
    wc_generator = WordCloudGenerator()

    #acquire paths
    wc_generator.acquire_input_path()
    wc_generator.acquire_output_path()

    #create word clouds
    wc_generator.create_dictionaries()
    wc_generator.create_wordcloud()

    #save word clouds
    wc_generator.save_word_cloud()

    #display word clouds
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