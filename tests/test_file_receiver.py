import unittest
from implementation.file_receiver import FileReceiver


class TestFileReceiver(unittest.TestCase):

    def setUp(self):
        self.file_receiver = FileReceiver()

    def test_acquire_input_path_valid(self):
        self.file_receiver.acquire_input_path()
        self.assertNotEqual("D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage",
                            self.file_receiver.input_folder_path)

    def test_acquire_input_path_invalid(self):
        self.file_receiver.acquire_input_path()
        self.assertEqual("D:\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage",
                         self.file_receiver.input_folder_path)

    def test_acquire_output_path_valid(self):
        self.file_receiver.acquire_output_path()
        self.assertNotEqual("D:\\Google_Play_Fraud_Benign_Malware\\Visualizations",
                            self.file_receiver.output_folder_path)

    def test_acquire_output_path_invalid(self):
        self.file_receiver.acquire_output_path()
        self.assertEqual("D:\\Google_Play_Fraud_Benign_Malware\\Visualizations",
                         self.file_receiver.output_folder_path)


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
