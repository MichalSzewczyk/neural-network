from preprocessing.data_loader import DataLoader
from preprocessing.words_encoder import WordsEncoder

data_loader = DataLoader()
embedding_generator = WordsEncoder()
data = data_loader.load_file('project/dev.txt')
embedding_generator.encode_words(data, 'data/test.txt')
print(data)
