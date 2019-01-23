import pandas
import nltk


class DataLoader:
    @staticmethod
    def load_file(file_name):
        data_frame = pandas.read_csv(file_name, sep='\t', lineterminator='\n')
        data_frame = DataLoader.extract_labels_and_sentences(data_frame)
        data_frame, removed = DataLoader.remove_undefined_labels(data_frame)
        print('-------------------------------------------')
        print('Removed entries without labels: ')
        print(removed.head(10))
        print('-------------------------------------------')
        data_frame = DataLoader.tokenize(data_frame)
        DataLoader.print_without_duplicates(data_frame)
        data_frame = DataLoader.add_metadata_tokens(data_frame)
        return data_frame

    @staticmethod
    def replace_and_remove_unknown_words(data_frame, result_map):
        result_list = []
        data_frame['sentence1'] = data_frame['sentence1'].apply(
            lambda row: DataLoader.replace_unknown(row, result_map, result_list))
        data_frame['sentence2'] = data_frame['sentence2'].apply(
            lambda row: DataLoader.replace_unknown(row, result_map, result_list))
        print('Number of unknown words: ' + str(len(result_list)))
        result_map = {k: v for k, v in result_map.items() if DataLoader.data_frame_contains_word(data_frame, k)}
        return data_frame, result_map

    @staticmethod
    def data_frame_contains_word(data_frame, word):
        for words in data_frame['sentence1']:
            if word in words:
                return True
        for words in data_frame['sentence2']:
            if word in words:
                return True
        return False

    @staticmethod
    def replace_unknown(array, result_map, replaced_words):
        result_array = []
        for word in array:
            if word is '<bos>' or word is '<eos>' or word in result_map.keys():
                result_array.append(word)
            else:
                result_array.append('<unk>')
                replaced_words.append(word)
        return result_array

    @staticmethod
    def extract_labels_and_sentences(data_frame):
        data_frame = data_frame.drop('sentence1_binary_parse', 1)
        data_frame = data_frame.drop('sentence2_binary_parse', 1)
        data_frame = data_frame.drop('sentence1_parse', 1)
        data_frame = data_frame.drop('sentence2_parse', 1)
        data_frame = data_frame.drop('captionID', 1)
        data_frame = data_frame.drop('pairID', 1)
        data_frame = data_frame.drop('label1', 1)
        data_frame = data_frame.drop('label2', 1)
        data_frame = data_frame.drop('label3', 1)
        data_frame = data_frame.drop('label4', 1)
        data_frame = data_frame.drop('label5', 1)
        return data_frame

    @staticmethod
    def remove_undefined_labels(data_frame):
        removed = data_frame[data_frame.gold_label == '-']
        data_frame = data_frame[data_frame.gold_label != '-']
        return data_frame, removed

    @staticmethod
    def tokenize(data_frame):
        data_frame['sentence1'] = data_frame['sentence1'].apply(lambda x: x.lower())
        data_frame['sentence1'] = data_frame['sentence1'].apply(nltk.word_tokenize)
        data_frame['sentence2'] = data_frame['sentence2'].apply(lambda x: x.lower())
        data_frame['sentence2'] = data_frame['sentence2'].apply(nltk.word_tokenize)
        return data_frame

    @staticmethod
    def add_metadata_tokens(data_frame):
        data_frame['sentence1'] = data_frame['sentence1'].apply(lambda array: ['<bos>'] + array)
        data_frame['sentence1'] = data_frame['sentence1'].apply(lambda array: array + ['<eos>'])

        data_frame['sentence2'] = data_frame['sentence2'].apply(lambda array: ['<bos>'] + array)
        data_frame['sentence2'] = data_frame['sentence2'].apply(lambda array: array + ['<eos>'])
        return data_frame

    @staticmethod
    def print_without_duplicates(data_frame):
        result_set = set()
        data_frame['sentence1'].apply(lambda array: result_set.update(array))
        print('Amount of words without duplicates: ' + len(result_set).__str__())
