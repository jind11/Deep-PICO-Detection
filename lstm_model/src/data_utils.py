import numpy as np
import random
import tensorflow as tf

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"
WORD_PAD = '$W_PAD$'
TAG_PAD = '$T_PAD$'

# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


def Dataset(filename, processing_word=None, processing_tag=None, max_iter=None):
    results = []
    with open(filename) as f:
        sentences, tags = [], []
        n_iter = 0
        for line in f:
            line = line.strip()
            if not line:
                if len(sentences) != 0:
                    n_iter += 1
                    if max_iter is not None and n_iter > max_iter:
                        break
                    results.append((sentences, tags))
                    sentences, tags = [], []
            elif not line.startswith("###"):
                ls = line.split('|')
                tag, sentence = ls[1], ls[2].split()
                # if tag != 'Others':
                if processing_word is not None:
                    try:
                        sentence = [processing_word(word) for word in sentence]
                    except:
                        pass
                if processing_tag is not None:
                    tag = processing_tag(tag)
                sentences += [sentence]
                tags += [tag]

    return results

# class Dataset(object):
#     """Class that iterates over CoNLL Dataset
#
#     __iter__ method yields a tuple (words, tags)
#         words: list of raw words
#         tags: list of raw tags
#
#     If processing_word and processing_tag are not None,
#     optional preprocessing is appplied
#
#     Example:
#         ```python
#         data = CoNLLDataset(filename)
#         for sentence, tags in data:
#             pass
#         ```
#
#     """
#     def __init__(self, filename, processing_word=None, processing_tag=None, max_iter=None):
#         """
#         Args:
#             filename: path to the file
#             processing_words: (optional) function that takes a word as input
#             processing_tags: (optional) function that takes a tag as input
#             max_iter: (optional) max number of sentences to yield
#
#         """
#         self.filename = filename
#         self.processing_word = processing_word
#         self.processing_tag = processing_tag
#         self.length = None
#         self.max_iter = max_iter
#
#
#     def __iter__(self):
#         with open(self.filename) as f:
#             sentences, tags = [], []
#             n_iter = 0
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     if len(sentences) != 0:
#                         n_iter += 1
#                         if self.max_iter is not None and n_iter > self.max_iter:
#                             break
#                         yield sentences, tags
#                         sentences, tags = [], []
#                 elif not line.startswith("###"):
#                     ls = line.split('|')
#                     tag, sentence = ls[1], ls[2].split()
#                     # if tag != 'Others':
#                     if self.processing_word is not None:
#                         sentence = [self.processing_word(word) for word in sentence]
#                     if self.processing_tag is not None:
#                         tag = self.processing_tag(tag)
#                     sentences += [sentence]
#                     tags += [tag]
#
#
#     def __len__(self):
#         """Iterates once over the corpus to set and store length"""
#         if self.length is None:
#             self.length = 0
#             for _ in self:
#                 self.length += 1
#
#         return self.length


class Embedding(object):
    """Embedding layer with frequency-based normalization and dropout."""
    def __init__(self, vocab_size=None,
                embedding_dim=None,
                embeddings=None,
                normalize=False,
                vocab_freqs=None,
                keep_prob=1.,
                trainable=False):
        # super(Embedding, self).__init__(**kwargs)
        with tf.variable_scope("words"):
            if embeddings is None:
                assert vocab_size is not None
                assert embedding_dim is not None
                self._word_embeddings = tf.get_variable(
                                name="_word_embeddings",
                                dtype=tf.float32,
                                shape=[vocab_size, embedding_dim])
            else:
                vocab_size = embeddings.shape[0]
                self._word_embeddings = tf.Variable(
                                embeddings,
                                name="_word_embeddings",
                                dtype=tf.float32,
                                trainable=trainable)

        self.keep_prob = keep_prob

        if normalize:
            assert vocab_freqs is not None
            vocab_freqs = tf.constant(
              vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))
            self._word_embeddings = self._normalize(self._word_embeddings, vocab_freqs)

    def embed(self, x):
        with tf.variable_scope("words"):
            embedded = tf.nn.embedding_lookup(self._word_embeddings, x)
            if self.keep_prob < 1.:
                # embedded = tf.nn.dropout(embedded, self.keep_prob)
                shape = embedded.get_shape().as_list()

                # Use same dropout masks at each timestep with specifying noise_shape.
                # This slightly improves performance.
                # Please see https://arxiv.org/abs/1512.05287 for the theoretical
                # explanation.
                if len(shape) == 3:
                    embedded = tf.nn.dropout(
                      embedded, self.keep_prob, noise_shape=(shape[0], 1, shape[2]))
                elif len(shape) == 4:
                    embedded = tf.nn.dropout(
                      embedded, self.keep_prob, noise_shape=(shape[0], 1, 1, shape[2]))
                else:
                    pass
        return embedded

    def _normalize(self, emb, vocab_freqs):
        weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keepdims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keepdims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_tags = set()
    vocab_words_freq = dict()
    for dataset in datasets:
        for sentences, tags in dataset:
            for sent in sentences:
                for token in sent:
                    vocab_words_freq[token] = vocab_words_freq.get(token, 0) + 1
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words_freq)))
    return vocab_words_freq, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for sents, _ in dataset:
        for sent in sents:
            for word in sent:
                vocab_char.update(word)

    return vocab_char


def get_wordvec_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        if isinstance(vocab, dict):
            for i, word in enumerate(vocab):
                if i != len(vocab) - 1:
                    f.write("{}\t{}\n".format(word, vocab[word]))
                else:
                    f.write('{}\t{}'.format(word, vocab[word]))
        else:
            for i, word in enumerate(vocab):
                if i != len(vocab) - 1:
                    f.write("{}\n".format(word))
                else:
                    f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        vocab_freq = []
        with open(filename) as f:
            for idx, line in enumerate(f):
                line = line.strip().split()
                if len(line) < 2:
                    word = line[0]
                    d[word] = idx
                else:
                    word, freq = line
                    d[word] = idx
                    try:
                        vocab_freq.append(int(freq))
                    except:
                        pass

    except IOError:
        raise MyIOError(filename)

    if len(vocab_freq) == 0:
        return d
    else:
        return d, vocab_freq


def export_trimmed_wordvec_vectors(vocab, wordvec_filename, trimmed_filename):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    num = 0
    with open(trimmed_filename, 'w') as outFile:
        with open(wordvec_filename, 'r') as inFile:
            for line in inFile:
                word = line.strip().split(' ')[0]
                if word in vocab:
                    outFile.write(line)
                    num += 1

    print('{} out of {} tokens can find pre-trained embeddings!'.format(num, len(vocab)))


def get_trimmed_wordvec_vectors(filename, vocab):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    f = open(filename, 'r')
    f.readline()
    dim = len(f.readline().strip().split()) - 1
    assert dim > 30
    embeddings = np.random.uniform(-0.1, 0.1, size=(len(vocab)+1, dim))
    with open(filename, 'r') as inFile:
        for line in inFile:
            line = line.strip().split()
            word = line[0]       
            if word in vocab:
                embeddings[vocab[word]] = np.array([float(item) for item in line[1:]])

    return embeddings


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=2):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length) 

    elif nlevels == 2:
        max_length_sentence = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_sentence)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_document = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_sentence, max_length_document)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_document)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size, shuffle=True):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    if shuffle:
        random.shuffle(data)

    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        # if type(x[0]) == tuple:
            # x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
