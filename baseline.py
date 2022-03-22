import torch
import argparse
from projection import Projector


def build_parser():
    """ Builds the parser for the command line arguments.
    Returns:
        parser: The parser for the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch size to use')
    parser.add_argument('--model_filename', type=str, default='seq2seq', help='filename of saved model. It will be loaded in from models/ with .pt and configs/ with .json')
    parser.add_argument('--output_file', type=str, default='output/seq2seq_projection.txt', help='filename of file to write results to')
    parser.add_argument('--use_test_too', action='store_true', help='use test data to get synthetic utterances too')
    parser.add_argument('--augment', action='store_true', help='whether to flip data fields in synthetic utterances to generate more')
    parser.add_argument('--no_projection', action='store_true', help='whether to evaluate the model without projection')
    return parser

if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    modelname = 'models/' + args.model_filename + '.pt'
    configname = 'configs/' + args.model_filename + '.json'
    print('Loading model from ', modelname, configname)

    data_option = 0
    if args.no_projection:
        data_option = 1

    projector = Projector(modelname, configname, 'overnight_data/calendar.paraphrases.train.examples', 'overnight_data/calendar.paraphrases.test.examples', batch_size=args.batch_size, get_synth_from_test_too=args.use_test_too, augment=args.augment, data_option=data_option)
    test_ds = projector.data.test_dataset

    outputs = []
    num_em = 0
    num_syn_match = 0
    f = open(args.output_file, 'w')
    for example in test_ds:
        paraphrase = example['paraphrase']
        synthetic_utt = example['synthetic']
        program = example['program']
        if args.no_projection:
            tensorized_input = torch.cat((projector.data.tensorize_utterance(paraphrase), torch.tensor([projector.data.EOS_TOK])))
        else:
            closest_synth = projector.get_most_similar_synth_utterance(paraphrase)
            tensorized_input = torch.cat((projector.data.tensorize_utterance(closest_synth), torch.tensor([projector.data.EOS_TOK])))
        src_len = len(tensorized_input)
        predicted_prog = projector.model(tensorized_input, src_len, projector.data)
        if args.no_projection:
            f.write('\n-------------\n')
            f.write('NL praphrase: {}\nTrue program: {}\n'.format(paraphrase, program))
            f.write('-->Predicted program:{}\n'.format(predicted_prog))
            f.write('-------------\n')
        else:
            f.write('\n-------------\n')
            f.write('NL praphrase: {}\nTrue synthetic utterance: {}\nTrue program: {}\n'.format(paraphrase, synthetic_utt, program))
            f.write('-->Closest synth: {}\n-->Predicted program:{}\n'.format(closest_synth, predicted_prog))
            f.write('-------------\n')
        if program.split() == predicted_prog.split():
            num_em += 1
        if not args.no_projection:
            if closest_synth.split() == synthetic_utt.split():
                num_syn_match += 1

    f.write("\nProgram Accuracy: {}".format(1.0 * num_em/len(test_ds)))
    if not args.no_projection:
        f.write("\nSynthetic Accuracy: {}".format(1.0 * num_syn_match/len(test_ds)))
    f.close()