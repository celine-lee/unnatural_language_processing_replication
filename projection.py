# load seq2seq model
# collect all synthetic utterances in training(?) data
# create a function that loads in sentence embedding  models
#     function should take in a NL sentence and return the most similar sentence in synthetics


import torch
from seq2seq import Encoder, Decoder, Seq2seq
import json
import numpy as np
from data import Calendar

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Projector(): 

    def __init__(self, model_file, config_file, train_datafile, test_datafile, embedder_name='all-MiniLM-L12-v2', batch_size=8, get_synth_from_test_too=False, augment=False, data_option=0):
        self.model = self.load_model(model_file, config_file)
        self.data = Calendar(train_datafile, test_datafile, batch_size=batch_size, data_option=data_option)
        self.embedder = SentenceTransformer(embedder_name)

        self.synth_utterances, self.synth_encodings = self.collect_all_synth_utterances(self.data.train_dataloader, self.data, augment)
        if get_synth_from_test_too:
            synth_utterances_test, synth_encodings_test = self.collect_all_synth_utterances(self.data.test_dataloader, self.data, augment)
            self.synth_utterances = np.concatenate((self.synth_utterances, synth_utterances_test))
            self.synth_encodings = np.concatenate((self.synth_encodings, synth_encodings_test))


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

    def collect_all_synth_utterances(self, dataloader, data, augment):
        """ Collects all synthetic utterances from the training data.
        Args:
            dataloader: The training data.
            data: Calendar object
            embedder: SentenceTransformer object
            augment: whether to augment the training data
        Returns:
            synth_utterances (dict): A dict of all synthetic utterances and their embeddings
        """

        synth_utterances = []

        def augment_sentence(sentence, old, new):
            if old in sentence:
                if sentence.replace(old, new) not in synth_utterances:
                    synth_utterances.append(sentence.replace(old, new))

        for input, _, _, _ in dataloader:
            for j in range(input.size(0)):
                synth_utterance = data.tensorized_to_utterance(input[j])
                synth_utterance = synth_utterance.replace('<EOS>', '').replace('<SOS>', '').replace('<PAD>', '')
                synth_utterances.append(synth_utterance)
                if augment:
                    augment_sentence(synth_utterance, 'weekly standup', 'annual review')
                    augment_sentence(synth_utterance, 'annual review', 'weekly standup')
                    augment_sentence(synth_utterance, 'jan 2', 'jan 3')
                    augment_sentence(synth_utterance, 'jan 3', 'jan 2')
                    augment_sentence(synth_utterance, 'start time', 'end time')
                    augment_sentence(synth_utterance, 'end time', 'start time')
                    augment_sentence(synth_utterance, '10am', '3pm')
                    augment_sentence(synth_utterance, '3pm', '10am')
                    augment_sentence(synth_utterance, 'three hours', 'one hour')
                    augment_sentence(synth_utterance, 'one hour', 'three hours')
                    augment_sentence(synth_utterance, 'alice', 'bob')
                    augment_sentence(synth_utterance, 'bob', 'alice')
                    augment_sentence(synth_utterance, 'greenberg cafe', 'central office')
                    augment_sentence(synth_utterance, 'central office', 'greenberg cafe')
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
