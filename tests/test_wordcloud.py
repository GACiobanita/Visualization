import unittest
import pandas as pd
from wordcloud import WordCloud
from implementation.wordcloud_generator import WordCloudGenerator

class Test_WordCloud(unittest.TestCase):

    def setUp(self):
        self.wordcloud=WordCloudGenerator()

    def test_acquire_input_path_valid(self):
        self.wordcloud.acquire_input_path()
        self.assertNotEqual("C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage",
                            self.wordcloud.input_folder_path)

    def test_acquire_input_path_invalid(self):
        self.wordcloud.acquire_input_path()
        self.assertEqual("C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Fraud\\PythonUsage", self.wordcloud.input_folder_path)

    def test_acquire_output_path_valid(self):
        self.wordcloud.acquire_output_path()
        self.assertNotEqual("C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Visualizations",
                            self.wordcloud.output_folder_path)

    def test_acquire_output_path_invalid(self):
        self.wordcloud.acquire_output_path()
        self.assertEqual("C:\\Users\\Alex\\Google_Play_Fraud_Benign_Malware\\Visualizations", self.wordcloud.output_folder_path)

    def test_removeStopWords_valid(self):
        result = self.wordcloud.remove_string_stopwords(["apple", "an", "the"])
        self.assertEqual(1, len(result))
        print(result)

    def test_removeStopWords_invalid(self):
        result = self.wordcloud.remove_string_stopwords(["an", "the"])
        self.assertEqual(1, len(result))
        print(result)

    def test_remove_string_punctuation_valid(self):
        test = "this string. has ! a bunch , of ? punctuation "
        test = self.wordcloud.remove_string_punctuation(test)
        self.assertEqual(-1, test.find('.'))
        print(test)

    def test_remove_string_punctuation_invalid(self):
        test = "this string has no punctuation"
        test = self.wordcloud.remove_string_punctuation(test)
        self.assertNotEqual(-1, test.find(','))
        print(test)

    def test_display_wordcloud_valid(self):
        wc = WordCloud(background_color="white")
        wc.generate_from_text("this is a word cloud text lmao lmao lmao lmao lo lo lo lol lol lol lol")
        self.wordcloud.word_clouds.append(wc)
        self.wordcloud.display_word_cloud()

    def test_display_wordcloud_invalid(self):
        self.wordcloud.display_word_cloud()

    def test_create_dict_from_tuple_valid(self):
        test = (("lol", 5), ("test", 4), ("trash", 3))
        result = self.wordcloud.create_dict_from_tuple(test)
        self.assertNotEqual(0, len(result))

    def test_create_dict_from_tuple_invalid(self):
        test = (("lol", 5), ("test", 4), ("trash", 3))
        result = self.wordcloud.create_dict_from_tuple(test)
        self.assertNotEqual(3, len(result))

    def test_create_dictionaries_valid(self):
        self.wordcloud.acquire_input_path()
        self.wordcloud.create_dictionaries()
        self.assertNotEqual(0, len(self.wordcloud.csv_files))
        self.assertNotEqual(0, len(self.wordcloud.word_freqs))

    def test_create_dictionaries_invalid(self):
        self.wordcloud.acquire_input_path()
        self.wordcloud.create_dictionaries()
        self.assertEqual(0, len(self.wordcloud.csv_files))
        self.assertEqual(0, len(self.wordcloud.word_freqs))

    def test_create_word_cloud_valid(self):
        test_data = {'Word': ["nice", "very nice"], 'Frequency': [100, 200]}
        test_frame = pd.DataFrame(test_data, columns=['Word', 'Frequency'])
        self.wordcloud.word_freqs.append(test_frame)
        self.wordcloud.create_wordcloud()
        self.assertNotEqual(0, len(self.wordcloud.word_clouds))
        self.wordcloud.display_word_cloud()

    def test_create_word_cloud_invalid(self):
        test_data = {'Word': ["nice", "very nice"], 'Frequency': [100, 200]}
        test_frame = pd.DataFrame(test_data, columns=['Word', 'Frequency'])
        self.wordcloud.word_freqs.append(test_frame)
        self.wordcloud.create_wordcloud()
        self.assertNotEqual(1, len(self.wordcloud.word_clouds))
        print(len(self.wordcloud.word_clouds))
        self.wordcloud.display_word_cloud()

    def test_save_word_cloud_valid(self):
        self.wordcloud.acquire_input_path()
        test_data = {'Word': ["nice", "very nice"], 'Frequency': [100, 200]}
        test_frame = pd.DataFrame(test_data, columns=['Word', 'Frequency'])
        self.wordcloud.word_freqs.append(test_frame)
        self.wordcloud.create_wordcloud()
        self.assertNotEqual(0, len(self.wordcloud.word_clouds))
        self.wordcloud.save_word_cloud()

    def test_save_word_cloud_invalid(self):
        self.wordcloud.acquire_input_path()
        test_data = {'Word': ["nice", "very nice"], 'Frequency': [100, 200]}
        test_frame = pd.DataFrame(test_data, columns=['Word', 'Frequency'])
        self.wordcloud.word_freqs.append(test_frame)
        self.wordcloud.create_wordcloud()
        self.assertEqual(0, len(self.wordcloud.word_clouds))
        self.wordcloud.save_word_cloud()

if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
