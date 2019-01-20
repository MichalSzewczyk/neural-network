import pandas


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
        return data_frame

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
