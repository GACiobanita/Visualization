import os

class FileReceiver:

    def __init__(self):
        self.INPUT_ROOT_FOLDER = "D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage"
        self.OUTPUT_ROOT_FOLDER = "D:\\Google_Play_Fraud_Benign_Malware\\Visualizations"
        self.input_folder_path = self.INPUT_ROOT_FOLDER
        self.output_folder_path = self.OUTPUT_ROOT_FOLDER
        self.csv_files = []

    def acquire_input_path(self):
        print("Default file path to input folder is: D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage")
        filepath = input("Input a new file path to a folder if you wish to change it: ")
        if os.path.exists(filepath) is True:
            print("New file path selected.")
            self.input_folder_path = filepath
            for r, d, f in os.walk(self.input_folder_path):
                for file in f:
                    self.csv_files.append(r + "\\" + file)
        else:
            print(
                "Input path is not correct, default will be used: D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage")
            for r, d, f in os.walk(self.input_folder_path):
                for file in f:
                    self.csv_files.append(r + "\\" + file)


    def acquire_output_path(self):
        print("Default file path to output folder is: D:\\Google_Play_Fraud_Benign_Malware\\Visualizations")
        filepath = input("Input a new file path to a folder if you wish to change it: ")
        if os.path.exists(filepath) is True:
            print("New file path selected.")
            self.output_folder_path = filepath
        else:
            print("Output path is not correct, default will be used: D:\\Google_Play_Fraud_Benign_Malware\\Visualizations")

    def pass_csv_files(self):
        return self.csv_files