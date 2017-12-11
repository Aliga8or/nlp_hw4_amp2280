from optparse import OptionParser
from network import *
import pickle
from net_properties import *
from utils import *
from vocab import *
import sys

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train_file", metavar="FILE", default=None)
    parser.add_option("--train_data", dest="train_data_file", metavar="FILE", default=None)
    parser.add_option("--test", dest="test_file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", metavar="FILE", default=None)
    parser.add_option("--vocab", dest="vocab_path", metavar="FILE", default=None)
    parser.add_option("--we", type="int", dest="we", default=100)
    parser.add_option("--pe", type="int", dest="pe", default=50)
    parser.add_option("--hidden_1", type="int", dest="hidden_1", default=200)
    parser.add_option("--hidden_2", type="int", dest="hidden_2", default=200)
    parser.add_option("--minibatch", type="int", dest="minibatch", default=1000)
    parser.add_option("--epochs", type="int", dest="epochs", default=7)

    (options, args) = parser.parse_args()
    we = 64
    pe = 32
    le = 32
    minibatch = 1000
    data_path = "data/"
    train_data_file = "data/train.data"

    print ("train_data_file: " + options.train_data_file)
    print ("model_path: " + options.model_path)
    print ("vocab_path: " + options.vocab_path)
    print ("hidden_1: " + str(options.hidden_1))
    print ("hidden_2: " + str(options.hidden_2))
    print ("epochs: " + str(options.epochs))

    #if options.train_file and options.train_data_file and options.model_path and options.vocab_path:
    net_properties = NetProperties(we, pe, le, options.hidden_1, options.hidden_2, minibatch)

    # creating vocabulary file
    vocab = Vocab(data_path)

    # writing properties and vocabulary file into pickle
    pickle.dump((vocab, net_properties), open(options.vocab_path, 'w'))

    #sys.exit("Updated Vocab Pickled !!!")

    # constructing network
    network = Network(vocab, net_properties)

    # training
    network.train(train_data_file,options.epochs)

    # saving network
    network.save(options.model_path)
    
    '''
    if options.test_file and options.model_path and options.vocab_path and options.output_file:
        # loading vocab and net properties
        vocab, net_properties = pickle.load(open(options.vocab_path, 'r'))

        # constructing default network
        network = Network(vocab, net_properties)

        # loading network trained model
        network.load(options.model_path)

        writer = open(options.output_file, 'w')
        for sentence in open(options.test_file, 'r'):
            words = sentence.strip().split()
            tags = network.decode(words)
            output = [word + '\t' + tag for word, tag in zip(words, tags)]
            writer.write('\n'.join(output) + '\n\n')
        writer.close()
    '''
