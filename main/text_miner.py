from implementation.text_mining import TextMining
from implementation.data_adjustments import DataAdjustment
import pandas as pd
import os
import matplotlib.pyplot as plt

N_TOPICS = [5, 8, 10, 12, 15, 20]
LEARNING_DECAY = [0.5, 0.7, 0.9]
TOP_WORDS = 10


def create_lda_model_plot(models, output_folder_path, file_name):
    mean_test_scores = models.cv_results_.get('mean_test_score')
    ld_scores = []
    for value in LEARNING_DECAY:
        ld_scores.append((value, mean_test_scores[:len(N_TOPICS)].tolist()))
        mean_test_scores = mean_test_scores[len(N_TOPICS):]
    plt.figure(figsize=(12, 8))
    for value in ld_scores:
        plt.plot(N_TOPICS, value[1], label=value[0])
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Topic Count")
    plt.ylabel("Mean Test Scores")
    plt.legend(title='Learning decay', loc='best')
    save_location = output_folder_path + "\\clusters_of_" + file_name[:-4] + ".png"
    plt.savefig(save_location)


def create_cluster_plot(x, y, clusters, output_folder_path, file_name):
    print("Creating cluster plot for:" + str(file_name))
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, c=clusters)
    print(clusters)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Segregation of Topic clusters")
    save_location = output_folder_path + "\\clusters_of_" + file_name[:-4] + ".png"
    plt.savefig(save_location)


def get_csv_files_from_directories():
    csv_directories = [
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2012\\sentiment",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2013\\sentiment",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2014\\sentiment",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2015\\sentiment",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\2016\\sentiment"]
    file_extension = ".csv"
    for directory_path in csv_directories:
        filenames = os.listdir(directory_path)
        for filename in filenames:
            if filename.endswith(file_extension):
                yield (directory_path, filename)


def get_output_directory(year):
    csv_output_directories = [
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2012",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2013",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2014",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2015",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2016"]
    for directory in csv_output_directories:
        if year in str(directory):
            return directory


def save_best_model_details(output_folder, file, score, params, estimator):
    file = open(output_folder + "\\" + file[:-4] + "_model_details.txt", "w")
    file.write("Score:" + str(score) + "\n")
    file.write("Params:" + str(params) + "\n")
    file.write("Estimator:" + str(estimator) + "\n")
    file.close()


def main():
    text_miner = TextMining()
    csv_column = "Text"
    csv_files = get_csv_files_from_directories()
    for (directory, file) in csv_files:
        year = text_miner.data_adjuster.get_year_from_string(file)
        output_folder = get_output_directory(year)
        csv_df = pd.read_csv(directory + "\\" + file)
        corpus = text_miner.chunk_iterator(csv_df[csv_column])
        text_length = len(csv_df[csv_column])
        term_frequency, tf_feature_names = text_miner.create_term_frequency_from_text(corpus, text_length)
        print("Searching for the best lda model for:" + str(file))
        models = text_miner.search_for_best_lda_model(term_frequency, N_TOPICS, LEARNING_DECAY)
        # save best model details
        save_best_model_details(output_folder, file, models.best_score_, models.best_params_, models.best_estimator_)
        # get data for plot to display differences between models
        create_lda_model_plot(models, output_folder, file)
        # take the best lda model for use
        best_lda_model = models.best_estimator_
        best_lda_matrix = best_lda_model.transform(term_frequency)
        best_topic_count = models.best_params_.get("n_components", 0)
        print("Creating topic data for:" + str(file))
        dominant_lda_topics = text_miner.get_dominant_topics_from_text(best_lda_model, term_frequency,
                                                                       len(csv_df['Text']),
                                                                       best_topic_count)
        topic_distribution = text_miner.get_topic_distribution_across_texts(dominant_lda_topics)
        topic_keywords = text_miner.get_top_keywords_per_topic(tf_feature_names, best_lda_model.components_, TOP_WORDS)
        # get cluster data using K-Means
        x_pos, y_pos, clusters = text_miner.get_cluster_data_based_on_topic_similarity(best_topic_count,
                                                                                       best_lda_matrix)
        create_cluster_plot(x_pos, y_pos, clusters, output_folder, file)
        dominant_lda_topics.to_csv(output_folder + "\\dominant_topics_" + file)
        topic_distribution.to_csv(output_folder + "\\topic_distribution_" + file)
        topic_keywords.to_csv(output_folder + "\\topic_keywords_" + file)
        print("Finished for file:" + str(file))


if __name__ == "__main__":
    main()
