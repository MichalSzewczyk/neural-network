from preprocessing.data_loader import DataLoader
from preprocessing.embedding import EmbeddingGenerator

data_loader = DataLoader()
embedding_generator = EmbeddingGenerator()
data = data_loader.load_file('project/dev.txt')
embedding_generator.get_embedding(data, 'data/test.txt')
print(data)
