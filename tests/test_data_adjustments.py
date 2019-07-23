import unittest
from implementation.data_adjustments import DataAdjustment


class Test_DataAdjustment(unittest.TestCase):

    def setUp(self):
        self.data_adjuster = DataAdjustment()

    def test_removeStopWords_valid(self):
        result = self.data_adjuster.remove_string_stopwords(["apple", "an", "the"])
        self.assertEqual(1, len(result))
        print(result)

    def test_removeStopWords_invalid(self):
        result = self.data_adjuster.remove_string_stopwords(["an", "the"])
        self.assertEqual(1, len(result))
        print(result)

    def test_remove_string_punctuation_valid(self):
        test = "this string. has ! a bunch , of ? punctuation "
        test = self.data_adjuster.remove_string_punctuation(test)
        self.assertEqual(-1, test.find('.'))
        print(test)

    def test_remove_string_punctuation_invalid(self):
        test = "this string has no punctuation"
        test = self.data_adjuster.remove_string_punctuation(test)
        self.assertNotEqual(-1, test.find(','))
        print(test)

    def test_create_dict_from_tuple_valid(self):
        test = (("lol", 5), ("test", 4), ("trash", 3))
        result = self.data_adjuster.create_dict_from_tuple(test)
        self.assertNotEqual(0, len(result))

    def test_create_dict_from_tuple_invalid(self):
        test = (("lol", 5), ("test", 4), ("trash", 3))
        result = self.data_adjuster.create_dict_from_tuple(test)
        self.assertNotEqual(3, len(result))


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
