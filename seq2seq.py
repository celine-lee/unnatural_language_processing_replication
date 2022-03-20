# python3 model.py --optimizer adadelta --learning_rate 0.001 --epochs 1200 --batch_size 16 --embedding_size 620 --hidden_size 1000 --a_hidden_units 1000 --mo_hidden_units 500 --teacher_forcing_ratio 1.0
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
# import wandb
import random
import json
import argparse

from data import Calendar


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

class Encoder(nn.Module):
    """ A biRNN encoder """
    def __init__(self, vocab_size, embedding_size=256, hidden_size=1024): 
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size 

        self.embedding = nn.Embedding(vocab_size, embedding_size) 
        self.birnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input, input_lens):
        """ Compute forward pass of encoder biRNN
        Args:
            input: (batch_size, seq_len) where values are indices in vocab
            input_lens: (batch_size) where values are original lengths of input sequences

        Returns:
            output: (batch_size, seq_len, hidden_size*2) annotations
        """
        batch_size = input.size(0)
        seq_len = input.size(1)
        embedded_input = self.embedding(input) # (batch_size, seq_len, embedding_size)
        assert embedded_input.size(0) == batch_size
        assert embedded_input.size(1) == seq_len
        assert embedded_input.size(2) == self.embedding_size
        packed_input = pack_padded_sequence(embedded_input, input_lens, batch_first=True, enforce_sorted=True)
        output, (hidden, cell) = self.birnn(packed_input) # output (batch_size, seq_len, 2*hidden_size)
        output, output_lengths = pad_packed_sequence(output, batch_first=True)
        # assert output.size(0) == batch_size
        # assert output.size(1) == seq_len
        # assert output.size(2) == 2*self.hidden_size
        return hidden, cell 
        # (batch_size, seq_len, 2*hidden_size)
    
class Decoder(nn.Module):
    """ A RNN decoder """
    def __init__(self, vocab_size, embedding_size=256, hidden_size=1024): 
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.decoder_rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True) # can we have it be bidirectional?
        self.embedder = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.linear_init_W = nn.Linear(hidden_size, hidden_size)


    def init_hidden(self, last_enc_hidden):
        """ Initialize s_0 = tanh(Ws h_1) 
        Args:
            last_enc_hidden: (batch_size, hidden_size)

        Returns:
            hidden: (batch_size, hidden_size)
        """
        return self.tanh(self.linear_init_W(last_enc_hidden))

    def forward(self, prev_tok, prev_hidden, prev_cell):
        """ Forward pass of the decoder
        Args:
            prev_tok: (batch_size)  token from previous time step
            prev_hidden: (1, batch_size, 2*hidden_size) s_{i-1}
            prev_cell: (1, batch_size, hidden_size) c_{i-1}

        Return:
            out: (batch_size, vocab_size) prob distr over possible next output tokens
            hidden: (batch_size, hidden_size) s_{i}
        """

        assert prev_hidden.shape[-1] == self.hidden_size, 'Hidden size mismatch {} vs {}'.format(prev_hidden.shape[-1], self.hidden_size)
        assert prev_cell.shape == prev_hidden.shape

        embedded_prev_tok = self.embedder(prev_tok) # (batch_size, embedding_size)
        embedded_prev_tok = embedded_prev_tok.unsqueeze(1)

        out, (hidden, cell) = self.decoder_rnn(embedded_prev_tok, (prev_hidden, prev_cell)) # (batch_size, hidden_size)
        out = self.linear(out) # (batch_size, vocab_size)
        return out, hidden, cell
        
class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder




def train_epoch(data, seq2seq, hidden_size, optimizer, criterion, force_teacher=True):
    """ Train seq2seq model.
    Args:
        data: IWSLT_Data object
        seq2seq: Seq2seq model
        hidden_size: size of hidden state
        optimizer:  optimizer
        criterion: loss function
        force_teacher: bool whether to force teacher forcing

    Returns:
        loss: scalar loss
        accuracy: scalar accuracy
        bleu: scalar bleu score
    """
    seq2seq.train()
    # iterator over the dataset. (padded) first element is input (batch_size, seq_len), second is target (batch_size, out_seq_len) 
    dataloader = data.train_dataloader

    total_loss = 0
    total_correct = 0
    total_num_items = 0

    for input, target, input_lens, _ in dataloader:
        print('INPUT: ', input.shape, '; TARGETS: ', target.shape)
        batch_size = input.size(0)
        # seq_len = input.size(1)
        out_seq_len = target.size(1)
        assert batch_size == target.size(0)

        # Send data to device
        input = input.to(device)
        target = target.to(device)
        # input_lens = input_lens.to(device)

        optimizer.zero_grad()
        # Encode inputs, get annotations
        decoder_state, cell = seq2seq.encoder(input, input_lens) # (2, batch_size, hidden_size)
        decoder_state = decoder_state.mean(dim=0, keepdim=True)
        cell = cell.mean(dim=0, keepdim=True)
        # Decode in time steps using annotations and decoder states
        out_token = torch.tensor([data.SOS_TOK]*batch_size).to(device) 
        loss = 0
        num_correct = 0
        num_in_this_batch = 0
        not_produced_eos = [b for b in range(batch_size)] # (batch_size)
        predictions = [[] for _ in range(batch_size)]
        for i in range(out_seq_len):
            out_token_prob_distr, decoder_state, cell = seq2seq.decoder(out_token, decoder_state, cell) # (batch_size, hidden_size)
            target_i = target[:,i]
            out_token_prob_distr = out_token_prob_distr[:,-1,:]
            predicted = torch.argmax(out_token_prob_distr, dim=-1)
            loss += criterion(out_token_prob_distr, target_i) 
            if force_teacher: out_token = target[:,i] # if teacher forcing
            else: out_token = predicted
            num_correct += torch.eq(predicted[not_produced_eos], target_i[not_produced_eos]).sum() 
            num_in_this_batch += len(not_produced_eos)
            not_produced_eos = [j for j in not_produced_eos if (out_token[j] != data.EOS_TOK)]
            for j in range(batch_size):
                predictions[j].append(predicted[j].item())
        

        # make predictions tensors
        for j in range(batch_size):
            predictions[j] = torch.tensor(predictions[j])
            print('\nSRC: ', data.tensorized_to_synth_utterance(input[j]), '\nTGT: ', data.tensorized_to_program(target[j]), '\n\t-->PRED: ', data.tensorized_to_program(predictions[j]))

        # update loss and accuracy
        total_loss += loss.item()
        total_correct += num_correct.item()
        total_num_items += num_in_this_batch

        loss.backward() # after loss.backwards, when we have the gradients, clip them, THEN step the optimizer otherwise lost
        # with torch.no_grad(): # loop over model parameters and clip to 1
        #     torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 1.0, norm_type=2)
        optimizer.step() 

    return total_loss / total_num_items, total_correct / total_num_items


def train_model(seq2seq, data, optimizer, hidden_size, num_epochs, learning_rate, teacher_forcing_ratio, patience):
    """ Train the model with the training set. 
    Args:
        seq2seq: the seq2seq model
        data: the data object
        optimizer: the optimizer
        hidden_size: the hidden size of the model
        num_epochs: the number of epochs to train for
        learning_rate: the learning rate
        teacher_forcing_ratio: the teacher forcing ratio
        patience: the patience for early stopping
    
    """
    seq2seq.train()

    if optimizer == "adadelta":
        optimizer = optim.Adadelta(seq2seq.parameters(), lr=learning_rate, rho=0.95)
    elif optimizer == "adam":
        optimizer = optim.Adam(seq2seq.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(seq2seq.parameters(), lr=learning_rate)
    else:
        print('Choose optimizer that is adadelta, adam, or sgd. Exiting...')
        exit()
    loss_fn = nn.CrossEntropyLoss(ignore_index=data.PAD_TOK)
    for epoch in range(num_epochs):
        force_teacher = (epoch < num_epochs/2) or (random.random() < teacher_forcing_ratio) 
        loss, accuracy = train_epoch(data, seq2seq, hidden_size, optimizer, loss_fn, force_teacher)
        
        # wandb.log({'loss': loss, 'accuracy': accuracy, 'bleu': bleu})
        print('epoch ', epoch, ': loss: ', loss, '; accuracy: ', accuracy)

def test_model(seq2seq, data, hidden_size):
    """ Test the model on the test set. 
    Args:
        seq2seq: the seq2seq model
        data: the data object
        hidden_size: the hidden size of the model
    
    Returns:
        loss: the loss on the test set
        accuracy: the accuracy on the test set
        bleu: the BLEU score on the test set
    """
    seq2seq.eval()
    dataloader = data.test_dataloader

    criterion = nn.CrossEntropyLoss(ignore_index=data.PAD_TOK)

    

    total_loss = 0
    total_correct = 0
    total_num_items = 0


    for input, target, input_lens, _ in dataloader:
        print('INPUT: ', input.shape, '; TARGETS: ', target.shape)
        batch_size = input.size(0)
        # seq_len = input.size(1)
        out_seq_len = target.size(1)
        assert batch_size == target.size(0)

        # Send data to device
        input = input.to(device)
        target = target.to(device)
        # input_lens = input_lens.to(device)

        # Encode inputs, get annotations
        decoder_state, cell = seq2seq.encoder(input, input_lens) # (2, batch_size, hidden_size)
        decoder_state = decoder_state.mean(dim=0, keepdim=True)
        cell = cell.mean(dim=0, keepdim=True)
        # Decode in time steps using annotations and decoder states
        out_token = torch.tensor([data.SOS_TOK]*batch_size).to(device) 
        loss = 0
        num_correct = 0
        num_in_this_batch = 0
        not_produced_eos = [b for b in range(batch_size)] # (batch_size)
        predictions = [[] for _ in range(batch_size)]
        for i in range(out_seq_len):
            out_token_prob_distr, decoder_state, cell = seq2seq.decoder(out_token, decoder_state, cell) # (batch_size, hidden_size)
            target_i = target[:,i]
            out_token_prob_distr = out_token_prob_distr[:,-1,:]
            predicted = torch.argmax(out_token_prob_distr, dim=-1)
            loss += criterion(out_token_prob_distr, target_i) 
            out_token = predicted
            num_correct += torch.eq(predicted[not_produced_eos], target_i[not_produced_eos]).sum() 
            num_in_this_batch += len(not_produced_eos)
            not_produced_eos = [j for j in not_produced_eos if (out_token[j] != data.EOS_TOK)]
            for j in range(batch_size):
                predictions[j].append(predicted[j].item())
        

        # make predictions tensors
        for j in range(batch_size):
            predictions[j] = torch.tensor(predictions[j])
            print('\nSRC: ', data.tensorized_to_synth_utterance(input[j]), '\nTGT: ', data.tensorized_to_program(target[j]), '\n\t-->PRED: ', data.tensorized_to_program(predictions[j]))

        # update loss and accuracy
        total_loss += loss.item()
        total_correct += num_correct.item()
        total_num_items += num_in_this_batch


    return total_loss / total_num_items, total_correct / total_num_items

def build_parser():
    """ Builds the parser for the command line arguments.
    Returns:
        parser: The parser for the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='adadelta', help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate to use')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size to use')
    parser.add_argument('--embedding_size', type=int, default=256, help='word embedding dimensionality')
    parser.add_argument('--hidden_size', type=int, default=1024, help='hidden size for encoder')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help='ratio of examples on which to do teacher forcing during training')
    parser.add_argument('--save_model', action='store_true', help='whether to save the model. Will overwrite loaded model, if set true.')
    parser.add_argument('--load_model', action='store_true', help='whether to load a model for evaluation.')
    parser.add_argument('--train_model', action='store_true', help='whether to train the model')
    parser.add_argument('--model_filename', type=str, default='seq2seq_model', help='filename of saved model. It will be saved in models/ with .pt and configs/ with .json')
    parser.add_argument('--patience', type=int, default=5, help='number of epochs to wait before early stopping')
    return parser

def load_model(model_file, config_file):
    """ Loads the model from the file specified in the command line arguments.
    Args:
        model_file: The file containing the model.
        config_file: The file containing the config.
    Returns:
        seq2seq: The seq2seq model.
    """
    config = json.load(open(config_file))
    encoder = Encoder(config['src_vocab_size'], config['embedding_size'], config['hidden_size'])
    decoder = Decoder(config['tgt_vocab_size'], config['embedding_size'], config['hidden_size'], config['a_hidden_units'], config['mo_hidden_units'])
    seq2seq = Seq2seq(encoder, decoder)
    seq2seq.load_state_dict(torch.load(model_file))
    return seq2seq

if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    model_path = 'models/' + args.model_filename + '.pt'
    config_path = 'configs/' + args.model_filename + '.json'

    if args.train_model:
        # wandb.init(project="unnatural-seq2seq-translation", 
        #             entity="celinelee", 
        #             config = args
        # )
        # config = wandb.config
        config = args
        data = Calendar('overnight_data/calendar.paraphrases.train.examples', 'overnight_data/calendar.paraphrases.test.examples', batch_size=args.batch_size)


        if config.load_model:
            seq2seq = load_model('models/' + config.model_filename + '.pt', 'configs/' + config.model_filename + '.json')
        else:
            encoder = Encoder(len(data.synth_utterance_vocab), config.embedding_size, config.hidden_size)
            decoder = Decoder(len(data.program_vocab), config.embedding_size, config.hidden_size)
            seq2seq = Seq2seq(encoder, decoder)
        seq2seq.to(device)
        train_model(seq2seq, data, config.optimizer, config.hidden_size, config.epochs, config.learning_rate, config.teacher_forcing_ratio, config.patience)
        print("Finished training...")
    else:
        config = args
        if config.load_model:
            seq2seq = load_model('models/' + config.model_filename + '.pt', 'configs/' + config.model_filename + '.json')
            data = Calendar('overnight_data/calendar.paraphrases.train.examples', 'overnight_data/calendar.paraphrases.test.examples', batch_size=args.batch_size)
            seq2seq.to(device)
        else:
            print('Must either train or load a model. Exiting...')
            exit()

    loss, accuracy = test_model(seq2seq, data, config.hidden_size)
    print('Test results:\n\tLoss: ', loss, '\n\tAccuracy: ', accuracy)

    # if config.save_model:
    #     torch.save(seq2seq.state_dict(), model_path)
    #     with open(config_path, 'w') as f:
    #         model_config = {
    #             'src_vocab_size': data.src_vocab_size,
    #             'tgt_vocab_size': data.tgt_vocab_size,
    #             'embedding_size': config.embedding_size,
    #             'hidden_size': config.hidden_size,
    #         }
    #         json.dump(model_config, f, indent=4)
    #     print('Saved model in ', model_path, ' with config saved to ', config_path)
    # else:
    #     print('Model not saved.')

