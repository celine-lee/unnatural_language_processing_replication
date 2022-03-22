# load seq2seq model
# collect all synthetic utterances in training(?) data
# create a function that loads in sentence embedding  models
#     function should take in a NL sentence and return the most similar sentence in synthetics


import torch
from seq2seq import Encoder, Decoder, Seq2seq
import json
from data import Calendar

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Projector(): 

    def __init__(self, model_file, config_file, train_datafile, test_datafile, embedder_name='all-MiniLM-L12-v2', batch_size=8):
        self.model = self.load_model(model_file, config_file)
        self.data = Calendar(train_datafile, test_datafile, batch_size=batch_size)
        self.embedder = SentenceTransformer(embedder_name)

        self.synth_utterances, self.synth_encodings = self.collect_all_synth_utterances(self.data.train_dataloader, self.data)


    def load_model(self, model_file, config_file):
        """ Loads the seq2seq model from the file specified.
        Args:
            model_file: The file containing the model.
            config_file: The file containing the config.
        Returns:
            seq2seq: The seq2seq model.
        """
        config = json.load(open(config_file))
        encoder = Encoder(config['src_vocab_size'], config['embedding_size'], config['hidden_size'])
        decoder = Decoder(config['tgt_vocab_size'], config['embedding_size'], config['hidden_size'])
        seq2seq = Seq2seq(encoder, decoder)
        seq2seq.load_state_dict(torch.load(model_file))
        return seq2seq

    def collect_all_synth_utterances(self, dataloader, data):
        """ Collects all synthetic utterances from the training data.
        Args:
            dataloader: The training data.
            data: Calendar object
            embedder: SentenceTransformer object
        Returns:
            synth_utterances (dict): A dict of all synthetic utterances and their embeddings
        """
        synth_utterances = []
        for input, _, _, _ in dataloader:
            for j in range(input.size(0)):
                synth_utterance = data.tensorized_to_synth_utterance(input[j])
                synth_utterances.append(synth_utterance)
        synth_encodings = self.embedder.encode(synth_utterances)
        return synth_utterances, synth_encodings
        
    def get_most_similar_synth_utterance(self, input_sentence):
        """ Return the most similar synthetic utterance, based on cosine distance in the embedding space.
        Args:
            input_sentence (str): input sentence in NL to map to synth sentence
            synth_encodings (array): array of all synthetic utterances' encodings
            synthetic_utterances (list): synth sentence corresp to encodings by idx
            tokenizer (SentenceTransformer): embedding model
        Returns:
            (str) synth utterance closest in embedding space
        """
        embedding = self.embedder.encode([input_sentence])
        similarities = cosine_similarity(embedding, self.synth_encodings)
        idx = similarities.argmax()
        return self.synth_utterances[idx]
