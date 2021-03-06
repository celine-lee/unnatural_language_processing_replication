import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CalendarDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Calendar():
    def __init__(self, train_filename, test_filename, batch_size = 2, data_option=0, general_grammarfile=None, specific_grammarfile=None):
        # data_option: 0=synth only; 1=real only; 2=both
        # TODO eventually both of the following dicts can be built from the grammar. (self.calendar_vocab = self.obtain_vocab_from_grammar(general_grammarfile, specific_grammarfile))
        self.utterance_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.program_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.PAD_TOK = self.utterance_vocab['<PAD>']
        self.SOS_TOK = self.utterance_vocab['<SOS>']
        self.EOS_TOK = self.utterance_vocab['<EOS>']
        assert self.PAD_TOK == self.program_vocab['<PAD>']
        assert self.SOS_TOK == self.program_vocab['<SOS>']
        assert self.EOS_TOK == self.program_vocab['<EOS>']
        self.idx_to_utterance_vocab = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
        self.idx_to_program_vocab = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}

        self.data_option = data_option
        
        self.train_dataset = CalendarDataset(self.process_datafile(train_filename))
        self.test_dataset = CalendarDataset(self.process_datafile(test_filename))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)


    def process_datafile(self, filename):
        examples = []
        current_example = {}
        for line in open(filename):
            if line.startswith('(example'):
                if current_example != {}:
                    if self.data_option == 0: # only synthetics
                        current_example['input'] = current_example['synthetic']
                    elif self.data_option == 1: # only paraphrases
                        current_example['input'] = current_example['paraphrase']
                    else: # add both paraphrases and synthetics
                        current_example['input'] = current_example['synthetic']
                        examples.append(current_example)
                        current_example['input'] = current_example['paraphrase']
                    examples.append(current_example)
                current_example = {}
            elif '(utterance "' in line:
                current_example['paraphrase'] = line.split('"')[1]
                if self.data_option in {1,2}:
                    for token in current_example['paraphrase'].split():
                        if token not in self.utterance_vocab:
                            self.idx_to_utterance_vocab[len(self.utterance_vocab)] = token
                            self.utterance_vocab[token] = len(self.utterance_vocab)

            elif '(original "' in line:
                current_example['synthetic'] = line.split('"')[1]
                if self.data_option in {0,2}:
                    for token in current_example['synthetic'].split():
                        if token not in self.utterance_vocab:
                            self.idx_to_utterance_vocab[len(self.utterance_vocab)] = token
                            self.utterance_vocab[token] = len(self.utterance_vocab)
            elif '(call edu.stanford.nlp.sempre.' in line:
                current_example['program'] = line.replace('call edu.stanford.nlp.sempre.overnight.SimpleWorld.', '').replace('\n', '').replace(')', ') ')
                # current_example['program'] = line.replace('call edu.stanford.nlp.sempre.overnight.SimpleWorld.', '').replace('(', '( ').replace(')', ' )').replace('\n', '')
                for token in current_example['program'].split(' '):
                    if token == '':
                        continue
                    if token not in self.program_vocab:
                        self.idx_to_program_vocab[len(self.program_vocab)] = token
                        self.program_vocab[token] = len(self.program_vocab)
        if current_example != {}:
            if self.data_option == 0: # only synthetics
                current_example['input'] = current_example['synthetic']
            elif self.data_option == 1: # only paraphrases
                current_example['input'] = current_example['paraphrase']
            else: # add both paraphrases and synthetics
                current_example['input'] = current_example['synthetic']
                for token in current_example['input'].split():
                    if token not in self.utterance_vocab:
                        self.idx_to_utterance_vocab[len(self.utterance_vocab)] = token
                        self.utterance_vocab[token] = len(self.utterance_vocab)
                examples.append(current_example)
                current_example['input'] = current_example['paraphrase']
            for token in current_example['input'].split():
                if token not in self.utterance_vocab:
                    self.idx_to_utterance_vocab[len(self.utterance_vocab)] = token
                    self.utterance_vocab[token] = len(self.utterance_vocab)
            examples.append(current_example)
        return examples

    def collate_fn(self, batch):

        inputs = [example['input'] for example in batch]
        programs = [example['program'] for example in batch]

        src = [torch.cat((self.tensorize_utterance(input), torch.tensor([self.EOS_TOK]))) for input in inputs]
        tgt = [torch.cat((self.tensorize_program(program), torch.tensor([self.EOS_TOK]))) for program in programs]
        src, tgt = zip(*sorted(zip(src, tgt), key=lambda x: len(x[0]), reverse=True))
        src_lens = [len(src_i) for src_i in src]
        tgt_lens = [len(tgt_i) for tgt_i in tgt]
        src_batch = pad_sequence(src, batch_first=True, padding_value=self.PAD_TOK)
        tgt_batch = pad_sequence(tgt, batch_first=True, padding_value=self.PAD_TOK)
        return src_batch, tgt_batch, src_lens, tgt_lens

    def tensorize_utterance(self, utterance):
        translated = []
        for tok in utterance.split():
            assert tok in self.utterance_vocab, "Because we are working with synthetic sentences, all tokens passed in should be in the vocabulary. {} wasn't.".format(tok)
            translated.append(self.utterance_vocab[tok])
        return torch.tensor(translated)

    def tensorize_program(self, program):
        translated = []
        for tok in program.split():
            assert tok in self.program_vocab, "Because we are working with programs, all tokens passed in should be in the vocabulary. {} wasn't.".format(tok)
            translated.append(self.program_vocab[tok])
        return torch.tensor(translated)

    def tensorized_to_program(self, program):
        translated = []
        for idx in program:
            translated.append(self.idx_to_program_vocab[idx.item()])
        return ' '.join(translated)

    def tensorized_to_utterance(self, tensorized_utterance):
        translated = []
        for idx in tensorized_utterance:
            translated.append(self.idx_to_utterance_vocab[idx.item()])
        return ' '.join(translated)


if __name__ == '__main__':
    data = CalendarDataset('overnight_data/calendar.paraphrases.train.examples', 'overnight_data/calendar.paraphrases.test.examples')
    for example in data.test_examples:
        print(example)
    print(data.program_vocab)



