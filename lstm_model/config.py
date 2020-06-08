import os
import argparse
from .general_utils import get_logger
from .data_utils import get_trimmed_wordvec_vectors, load_vocab, \
        get_processing_word


def Config(load=True):
    """Initialize hyperparameters and load vocabs

    Args:
        load_embeddings: (bool) if True, load embeddings into
            np array, else None

    """
    def load_(args):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        args.vocab_words, args.vocab_words_freq = load_vocab(args.filename_words)
        args.vocab_tags = load_vocab(args.filename_tags)
        # args.vocab_chars = load_vocab(args.filename_chars)

        args.nwords = len(args.vocab_words)
        # args.nchars     = len(args.vocab_chars)
        args.ntags = len(args.vocab_tags)

        # 2. get processing functions that map str -> id
        # args.use_chars = args.use_lstm_chars | args.use_cnn_chars
        args.processing_word = get_processing_word(args.vocab_words, lowercase=True, chars=False)
        args.processing_tag  = get_processing_word(args.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        args.embeddings = (get_trimmed_wordvec_vectors(args.filename_wordvec_trimmed, args.vocab_words)
                if args.use_pretrained else None)
        args.dim_word = args.embeddings.shape[1]

        return args

    ## parse args
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('--nepochs', default='100', type=int,
                help='number of epochs')
    parser.add_argument('--dropout', default='0.8', type=float,
                help='number of epochs')
    parser.add_argument('--batch_size', default='30', type=int,
                help='batch size')
    parser.add_argument('--batch_size_aug', default='30', type=int,
                        help='batch size for data augmentation')
    parser.add_argument('--lr', default='0.001', type=float,
                help='learning rate')
    parser.add_argument('--lr_method', default='adam', type=str,
                help='optimization method')
    parser.add_argument('--lr_decay', default='0.99', type=float,
                help='learning rate decay rate')
    parser.add_argument('--clip', default='2', type=float,
                help='gradient clipping')
    parser.add_argument('--nepoch_no_imprv', default='4', type=int,
                help='number of epoch patience')
    parser.add_argument('--l2_reg_lambda', default='1e-6', type=float,
                help='l2 regularization coefficient')

    # data and results paths
    parser.add_argument('--dir_output', default='test', type=str,
                help='directory for output')
    parser.add_argument('--data_keyname', default='nicta', type=str,
                help='directory for output')
    parser.add_argument('--filename_wordvec_trimmed', default='../data/word2vec_pubmed.trimmed.txt',
                type=str, help='directory for trimmed word embeddings file')
    parser.add_argument('--filename_wordvec', default='/data/medg/misc/jindi/nlp/embeddings/word2vec/wikipedia-pubmed-and-PMC-w2v.txt',
                type=str, help='directory for original word embeddings file')

    # model hyperparameters
    parser.add_argument('--hidden_size_char', default='50', type=int,
                help='hidden size of character level lstm')
    parser.add_argument('--hidden_size_lstm_sentence', default='100', type=int,
                help='hidden size of sentence level lstm')
    parser.add_argument('--hidden_size_lstm_document', default='100', type=int,
                help='hidden size of document level lstm')
    parser.add_argument('--attention_size', default='400', type=int,
                help='attention vector size')
    parser.add_argument('--cnn_filter_num', default='300', type=int,
                help='number of cnn filters for each window size')
    parser.add_argument('--dim_char', default='50', type=int,
                help='character embedding dimension')
    parser.add_argument('--cnn_filter_sizes', default='2,3,4', type=str,
                help='cnn filter window sizes')
    parser.add_argument('--cnn_char_windows', default='3', type=str,
                help='cnn filter window sizes')
    parser.add_argument('--adv_reg_coeff', default='0.2', type=float,
                help='Regularization coefficient of adversarial loss')
    parser.add_argument('--va_reg_coeff', default='0.05', type=float,
                help='Regularization coefficient of virtual adversarial loss')
    parser.add_argument('--adv_perturb_norm_length', default='8.0', type=float,
                help='Norm length of adversarial perturbation to be')
    parser.add_argument('--va_perturb_norm_length', default='4.0', type=float,
                help='Norm length of virtual adversarial perturbation to be')
    parser.add_argument('--embedding_dropout', default='0.8', type=float,
                help='Keep dropout for embeddings')
    parser.add_argument('--embedding_normalize', action='store_false',
                help='Whether normalize the embeddings')

    # misc
    parser.add_argument('--restore', action='store_true',
                help='whether restore from previous trained model')
    parser.add_argument('--use_crf', action='store_false',
                help='whether use crf optimization layer')
    parser.add_argument('--use_document_level', action='store_false',
                help='whether use document level lstm layer')
    parser.add_argument('--use_document_attention', action='store_true',
                        help='whether use document level attention')
    parser.add_argument('--use_attention', action='store_false',
                help='whether use attention based pooling')
    parser.add_argument('--use_cnn', action='store_false',
                help='whether use cnn or lstm for sentence representation')
    parser.add_argument('--train_embeddings', action='store_true',
                help='whether use cnn or lstm for sentence representation')
    parser.add_argument('--use_pretrained', action='store_false',
                help='whether use pre-trained word embeddings')
    parser.add_argument('--train_accuracy', action='store_true',
                help='whether report accuracy while training')
    parser.add_argument('--min_freq', default='20', type=int,
                        help='remove tokens with small frequency for vocab')
    parser.add_argument('--num_augmentation', default='0', type=int,
                        help='Number of abstracts for data augmentation for VADV')

    args = parser.parse_args()

    # args.filename_wordvec = os.path.join('/data/medg/misc/jindi/nlp/embeddings',
    #                                     args.filename_wordvec)
    args.dir_output = os.path.join('results', args.dir_output)
    if not os.path.exists(args.dir_output):
        os.makedirs(args.dir_output)
    args.dir_model = os.path.join(args.dir_output, "model.weights")
    args.path_log = os.path.join(args.dir_output, "log.txt")

    # dataset
    if args.data_keyname == 'PICO':
        args.data_root = '../data/pico'
        args.filename_dev = os.path.join(args.data_root, 'dev.txt')
        args.filename_test = os.path.join(args.data_root, 'test.txt')
        args.filename_train = os.path.join(args.data_root, 'train.txt')
    elif args.data_keyname == 'nicta':
        args.data_root = '../data/nicta_piboso'
        args.filename_dev = os.path.join(args.data_root, 'test_clean.txt')
        args.filename_test = os.path.join(args.data_root, 'test_clean.txt')
        args.filename_train = os.path.join(args.data_root, 'train_clean.txt')

    # data augmentation dataset
    args.filename_aug = '../data/unlabeled_corpus'

    # vocab (created from dataset with build_data.py)
    args.filename_words = os.path.join('data', args.data_keyname, 'words.txt')
    args.filename_tags = os.path.join('data', args.data_keyname, 'tags.txt')
    # args.filename_chars = os.path.join('data', args.data_keyname, 'chars.txt')

    args.cnn_filter_sizes = [int(i) for i in args.cnn_filter_sizes.split(',')]
    args.cnn_char_windows = [int(i) for i in args.cnn_char_windows.split(',')]

    # directory for training outputs
    if not os.path.exists(os.path.join('data', args.data_keyname)):
        os.makedirs(os.path.join('data', args.data_keyname))

    # directory for data output
    if not os.path.exists(args.dir_output):
        os.makedirs(args.dir_output)

    # create instance of logger
    args.logger = get_logger(args.path_log)

    # log the attributes
    msg = ', '.join(['{}: {}'.format(attr, getattr(args, attr)) for attr in dir(args) \
                    if not callable(getattr(args, attr)) and not attr.startswith("__")])
    args.logger.info(msg)

    # load if requested (default)
    if load:
        args = load_(args)

    return args
