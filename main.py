from preprocessing.data_loader import DataLoader
from preprocessing.embedding_layer import EmbeddingLayerCreator
from preprocessing.statements_encoder import StatementsEncoder
from preprocessing.words_encoder import WordsEncoder

statements_encoder = StatementsEncoder()
prepared_for_embedding = DataLoader.load_file('project/dev.txt')
words_to_vectors_mapping = WordsEncoder.prepare_embedding('data/glove.6B.200d.txt')
print('prepared for embedding: ' + str(prepared_for_embedding))
prepared_for_embedding, words_to_vectors_mapping = DataLoader.replace_and_remove_unknown_words(prepared_for_embedding,
                                                                                               words_to_vectors_mapping)
embedding = EmbeddingLayerCreator.create_layer(prepared_for_embedding, words_to_vectors_mapping)
