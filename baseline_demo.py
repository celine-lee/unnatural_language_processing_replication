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
    parser.add_argument('--model_filename', type=str, default='seq2seq_adam_8', help='filename of saved model. It will be loaded in from models/ with .pt and configs/ with .json')
    parser.add_argument('--output_file', type=str, default='output/interactive_projection.txt', help='filename of file to write results to')
    return parser

if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    modelname = 'models/' + args.model_filename + '.pt'
    configname = 'configs/' + args.model_filename + '.json'
    print('Loading model from ', modelname, configname)

    projector = Projector(modelname, configname, 'overnight_data/calendar.paraphrases.train.examples', 'overnight_data/calendar.paraphrases.test.examples', batch_size=args.batch_size, get_synth_from_test_too=True)

    outputs = []
    f = open(args.output_file, 'w')
    while True:
        nl_utterance = input('\nPlease enter a request for the Calendar. (type \'exit\' to exit): ')
        if nl_utterance == 'exit':
            break
        closest_synth = projector.get_most_similar_synth_utterance(nl_utterance)
        print('--> Closest synthetic utterance: {}'.format(closest_synth))
        tensorized_closest_synth = torch.cat((projector.data.tensorize_utterance(closest_synth), torch.tensor([projector.data.EOS_TOK])))
        src_len = len(tensorized_closest_synth)
        predicted_prog = projector.model(tensorized_closest_synth, src_len, projector.data)
        f.write('\n-------------\n')
        f.write('NL input: {}\n'.format(nl_utterance))
        f.write('-->Closest synth: {}\n-->Predicted program:{}\n'.format(closest_synth, predicted_prog))
        f.write('-------------\n')
        print('-->Predicted program: {}\n'.format(predicted_prog))

    f.close()