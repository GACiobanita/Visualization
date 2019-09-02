from implementation.dendrogram_generator import DendrogramGenerator

import os
import pandas as pd


def get_csv_files_from_directories(csv_directory):
    file_extension = ".csv"
    filenames = os.listdir(csv_directory)
    for filename in filenames:
        if filename.endswith(file_extension):
            yield filename


def main():
    dendrogram_generator = DendrogramGenerator()

    input_folders = [
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2015\\Keywords",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2013\\Keywords",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2014\\Keywords",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2015\\Keywords",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\All Data\\LATENT DIRICHLET ALLOCATION\\2016\\Keywords"]
    output_folders = [
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\All Data\\LATENT DIRICHLET ALLOCATION\\2015\\Dendrograms",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\All Data\\LATENT DIRICHLET ALLOCATION\\2013\\Dendrograms",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\All Data\\LATENT DIRICHLET ALLOCATION\\2014\\Dendrograms",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\All Data\\LATENT DIRICHLET ALLOCATION\\2015\\Dendrograms",
        "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\All Data\\LATENT DIRICHLET ALLOCATION\\2016\\Dendrograms"]

    for i in range(0, len(input_folders)):
        filenames = get_csv_files_from_directories(input_folders[i])
        for filename in filenames:
            print(filename)
            csv_data = pd.read_csv(input_folders[i] + '\\' + filename)
            topics = dendrogram_generator.construct_topics_from_file(csv_data)
            topic_dicts = dendrogram_generator.construct_topic_vocabulary_from_file(csv_data, topics)
            tree_structure = dendrogram_generator.construct_tree_structure(topic_dicts)
            linkage_matrix, leaf_names = dendrogram_generator.construct_linkage_matrix(tree_structure)
            dendrogram = dendrogram_generator.construct_dendrogram(linkage_matrix, leaf_names, tree_structure['Topics'])
            dendrogram_generator.save_dendrogram("Topic distribution for " + filename, dendrogram, output_folders[i],
                                                 filename)


if __name__ == "__main__":
    main()
