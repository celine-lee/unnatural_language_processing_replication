from torch.utils.data import Dataset, DataLoader

# example:original is synthetic; example:utterance is mturk paraphrase
# example:targetFormula is program. I want to translate it into a simpler text of the same formula

class CalendarDataset(Dataset):
    def __init__(self, filename):
        self.examples = self.process_datafile(filename)
        self.num_examples = len(self.examples)
        self.calendar_vocab = {}

    def process_datafile(self, filename):
        examples = []
        current_example = {}
        for line in open(filename):
            if line.startswith('(example'):
                if current_example != {}:
                    examples.append(current_example)
                current_example = {}
            elif '(utterance "' in line:
                current_example['paraphrase'] = line.split('"')[1]
            elif '(original "' in line:
                current_example['synthetic'] = line.split('"')[1]
            elif '(call edu.stanford.nlp.sempre."' in line:
                current_example['program'] = line.replace('call edu.stanford.nlp.sempre.overnight.SimpleWorld.', '')
        if current_example != {}:
            examples.append(current_example)
        return examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.examples[idx]

if __name__ == '__main__':
    data = CalendarDataset('overnight_data/calendar.paraphrases.test.examples')
    for example in data:
        print(example)



