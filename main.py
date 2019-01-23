from preprocessing.data_loader import DataLoader
from preprocessing.statements_encoder import StatementsEncoder
from preprocessing.words_encoder import WordsEncoder

data_loader = DataLoader()
words_encoder = WordsEncoder()
statements_encoder = StatementsEncoder()
data = data_loader.load_file('project/dev.txt')
data = words_encoder.encode_words(data, 'data/test.txt')
statements_encoder.encode_statements(data)
print(data)
