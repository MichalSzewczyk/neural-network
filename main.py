from preprocessing.data_loader import DataLoader
from preprocessing.embedding_layer import EmbeddingLayerCreator
from preprocessing.statements_encoder import StatementsEncoder
from preprocessing.words_encoder import WordsEncoder

data_loader = DataLoader()
words_encoder = WordsEncoder()
statements_encoder = StatementsEncoder()
prepared_for_embedding = data_loader.load_file('project/dev.txt')
words_to_vectors_mapping = words_encoder.prepare_embedding('data/test.txt')
print('prepared for embedding: ' + str(prepared_for_embedding))
prepared_for_embedding, words_to_vectors_mapping = data_loader.replace_and_remove_unknown_words(prepared_for_embedding,
                                                                                                words_to_vectors_mapping)
embedding = EmbeddingLayerCreator.create_layer(prepared_for_embedding, words_to_vectors_mapping)
