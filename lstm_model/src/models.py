import numpy as np
import os
import itertools
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix


from .data_utils import minibatches, pad_sequences, get_chunks, WORD_PAD, TAG_PAD, Embedding
from .general_utils import Progbar
from .base_model import BaseModel
from .adversarial_losses import adversarial_loss, virtual_adversarial_loss


class HANNModel(BaseModel):
    def __init__(self, config):
        super(HANNModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        # self.target_names = [self.idx_to_tag[i] for i in range(len(self.idx_to_tag)) if self.idx_to_tag[i]!=TAG_PAD]
        self.target_names = [self.idx_to_tag[i] for i in range(len(self.idx_to_tag))]

        self.idx_to_words = {idx: word for word, idx in
                           self.config.vocab_words.items()}
        # self.class_weights = [self.config.weight_tags[tag] for idx, tag in sorted(self.idx_to_tag.items())]
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.l2_reg_lambda)


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size)
        self.document_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="document_lengths")

        # shape = (batch size, max length of documents in batch (how many sentences in one abstract), max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="word_ids")

        # shape = (batch_size, max_length of sentence)
        self.sentence_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # for data augmentation
        self.word_aug_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                           name="word_aug_ids")

        self.document_lengths_aug = tf.placeholder(tf.int32, shape=[None],
                                               name="document_lengths")

        self.sentence_lengths_aug = tf.placeholder(tf.int32, shape=[None, None],
                                               name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None, data_aug=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) drop prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        word_pad_idx = self.config.vocab_words[WORD_PAD]
        # word_pad_idx = 0
        # tag_pad_idx = self.config.vocab_tags[TAG_PAD]

        _, document_lengths = pad_sequences(words, pad_tok=word_pad_idx, nlevels=1)
        word_ids, sentence_lengths = pad_sequences(words, pad_tok=word_pad_idx, nlevels=2)

        if data_aug is not None:
            _, document_lengths_aug = pad_sequences(data_aug, pad_tok=word_pad_idx, nlevels=1)
            word_aug_ids, sentence_lengths_aug = pad_sequences(data_aug, pad_tok=word_pad_idx, nlevels=2)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.document_lengths: document_lengths,
            self.sentence_lengths: sentence_lengths
        }

        if labels is not None:
            labels, _ = pad_sequences(labels, 0, nlevels=1)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        if data_aug is not None:
            feed[self.word_aug_ids] = word_aug_ids
            feed[self.document_lengths_aug] = document_lengths_aug
            feed[self.sentence_lengths_aug] = sentence_lengths_aug

        return feed, document_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        def _normalize(emb, vocab_freqs):
            weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
            mean = tf.reduce_sum(weights * emb, 0, keepdims=True)
            var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keepdims=True)
            stddev = tf.sqrt(1e-6 + var)
            return (emb - mean) / stddev

        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            if self.config.embedding_normalize:
                assert self.config.vocab_words_freq is not None
                vocab_freqs = tf.constant(
                 self.config.vocab_words_freq, dtype=tf.float32, shape=(self.config.nwords+1, 1))
                _word_embeddings = _normalize(_word_embeddings, vocab_freqs)

        #     word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
        #             self.word_ids, name="word_embeddings")
        #
        # word_embeddings = tf.nn.dropout(word_embeddings, self.config.embedding_dropout)

        return _word_embeddings


    def forward(self, word_embeddings, sentence_lengths, document_lengths):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        s = tf.shape(word_embeddings)

        word_embeddings_dim = self.config.dim_word

        sentence_lengths = tf.reshape(sentence_lengths, shape=[s[0]*s[1]])
        
        if self.config.use_cnn:
            word_embeddings = tf.reshape(word_embeddings, 
                            shape=[s[0]*s[1], s[-2], word_embeddings_dim, 1])

            if self.config.use_attention:
                with tf.variable_scope("conv", reuse=tf.AUTO_REUSE):
                    W_word = tf.get_variable("weight", dtype=tf.float32, 
                        initializer=self.initializer, regularizer=self.regularizer,
                        shape=[self.config.cnn_filter_num, self.config.attention_size])
                    b_word = tf.get_variable("bias", shape=[self.config.attention_size],
                        dtype=tf.float32, initializer=tf.zeros_initializer())
                    U_word = tf.get_variable("U-noreg", dtype=tf.float32, 
                            initializer=self.initializer, 
                            shape=[self.config.attention_size, 1])

            pooled_outputs = []
            for i, size in enumerate(self.config.cnn_filter_sizes):
                with tf.variable_scope("conv-%d" % size, reuse=tf.AUTO_REUSE):# , reuse=False
                    W_conv = tf.get_variable(name='weight', initializer=self.initializer, 
                                            shape=[size, word_embeddings_dim, 1, self.config.cnn_filter_num], 
                                            regularizer=self.regularizer)
                    b_conv = tf.get_variable(name='bias', initializer=self.initializer, 
                                            shape=[self.config.cnn_filter_num])
                    conv = tf.nn.conv2d(word_embeddings, W_conv, strides=[1, 1, word_embeddings_dim, 1],
                                        padding="SAME")

                    h = tf.nn.tanh(tf.nn.bias_add(conv, b_conv), name="h") # bz, n, 1, dc
                    h = tf.squeeze(h, axis=2) # bz, n, dc

                    if self.config.use_attention:
                        h = tf.reshape(h, shape=[-1, self.config.cnn_filter_num])
                        U_sent = tf.tanh(tf.matmul(h, W_word) + b_word)
                        A = tf.nn.softmax(tf.reshape(tf.squeeze(tf.matmul(U_sent, U_word)), shape=[-1, s[2]]))
                        h = tf.reshape(h, shape=[-1, s[2], self.config.cnn_filter_num])
                        pooled = tf.reduce_sum(tf.multiply(h, tf.tile(tf.expand_dims(A, axis=-1),
                                                [1, 1, self.config.cnn_filter_num])), axis=1) # bz, dc
                    else:
                        # max pooling
                        pooled = tf.reduce_max(h, axis=1) # bz, dc
                    
                    pooled_outputs.append(pooled)

            output = tf.concat(pooled_outputs, axis=-1) 
            # dropout
            output = tf.nn.dropout(output, self.dropout)

            cnn_filter_tot_num = len(self.config.cnn_filter_sizes) * self.config.cnn_filter_num

            if self.config.use_document_level == True:
                output = tf.reshape(output, 
                            shape=[-1, s[1], cnn_filter_tot_num])
        else:
            word_embeddings = tf.reshape(word_embeddings, 
                                shape=[s[0]*s[1], s[-2], word_embeddings_dim])

            if self.config.use_attention:
                with tf.variable_scope("bi-lstm-sentence", reuse=tf.AUTO_REUSE):
                    cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)
                    cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)

                    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw, cell_bw, word_embeddings,
                            sequence_length=sentence_lengths, dtype=tf.float32)
                    output = tf.concat([output_fw, output_bw], axis=-1)

                    W_word = tf.get_variable("weight", dtype=tf.float32, 
                            initializer=self.initializer, regularizer=self.regularizer,
                            shape=[2*self.config.hidden_size_lstm_sentence, self.config.attention_size])
                    b_word = tf.get_variable("bias", shape=[self.config.attention_size],
                        dtype=tf.float32, initializer=tf.zeros_initializer())
                    U_word = tf.get_variable("U-noreg", dtype=tf.float32, 
                            initializer=self.initializer, 
                            shape=[self.config.attention_size, 1])

                    output = tf.reshape(output, shape=[-1, 2*self.config.hidden_size_lstm_sentence])
                    U_sent = tf.tanh(tf.matmul(output, W_word) + b_word)
                    A = tf.nn.softmax(tf.reshape(tf.squeeze(tf.matmul(U_sent, U_word)), shape=[-1, s[2]]))
                    output = tf.reshape(output, shape=[-1, s[2], 2*self.config.hidden_size_lstm_sentence])
                    output = tf.reduce_sum(tf.multiply(output, tf.tile(tf.expand_dims(A, axis=-1), 
                                            [1, 1, 2*self.config.hidden_size_lstm_sentence])), axis=1)

            else:
                with tf.variable_scope("bi-lstm-sentence", reuse=tf.AUTO_REUSE):
                    cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)
                    cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)
                    _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw, cell_bw, word_embeddings,
                            sequence_length=sentence_lengths, dtype=tf.float32)
                    output = tf.concat([output_fw, output_bw], axis=-1)

            # dropout
            output = tf.nn.dropout(output, self.dropout)
            if self.config.use_document_level == True:
                output = tf.reshape(output, [-1, s[1], 2*self.config.hidden_size_lstm_sentence])

        if self.config.use_document_level == True:
            with tf.variable_scope("bi-lstm-document", reuse=tf.AUTO_REUSE):
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_document)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_document)

                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, output,
                    sequence_length=document_lengths, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
                # dropout
                output = tf.nn.dropout(output, self.dropout)
                output = tf.reshape(output, shape=[-1, 2*self.config.hidden_size_lstm_document])

        if self.config.use_document_level == True:
            hidden_size = 2 * self.config.hidden_size_lstm_document
        else:
            if self.config.use_cnn:
                hidden_size = cnn_filter_tot_num
            else:
                hidden_size = 2 * self.config.hidden_size_lstm_sentence

        with tf.variable_scope("proj", reuse=tf.AUTO_REUSE):
            W_infer = tf.get_variable("weight", dtype=tf.float32, 
                    initializer=self.initializer, regularizer=self.regularizer,
                    shape=[hidden_size, self.config.ntags])

            b_infer = tf.get_variable("bias", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            pred = tf.matmul(output, W_infer) + b_infer
            logits = tf.reshape(pred, [-1, s[1], self.config.ntags])

        return logits


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def original_loss(self, logits, labels, document_lengths):
        """Defines the loss"""
        if self.config.use_crf:
            with tf.variable_scope("crf_loss", reuse=tf.AUTO_REUSE):
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                        logits, labels, document_lengths)
                self.trans_params = trans_params # need to evaluate it for decoding
                loss = tf.reduce_mean(-log_likelihood)
        else:
            with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels)
                mask = tf.sequence_mask(document_lengths)
                losses = tf.boolean_mask(losses, mask)
                loss = tf.reduce_mean(losses)

        # add l2 regularization
        l2 = self.config.l2_reg_lambda * sum([
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "bias" in tf_var.name)])
        loss += l2

        # for tensorboard
        tf.summary.scalar("loss", loss)

        return loss


    def embedd_to_loss(self, embedd):
        logits = self.forward(embedd, self.sentence_lengths, self.document_lengths)
        loss = self.original_loss(logits, self.labels, self.document_lengths)
        return loss


    def build(self):
        # NER specific functions
        tf.reset_default_graph()
        self.add_placeholders()

        word_embeddings_table = self.add_word_embeddings_op()
        word_embeddings = tf.nn.embedding_lookup(word_embeddings_table,
                                                self.word_ids, name="word_embeddings")
        word_embeddings = tf.nn.dropout(word_embeddings, self.config.embedding_dropout)

        self.logits = self.forward(word_embeddings, self.sentence_lengths, self.document_lengths)
        self.loss = self.original_loss(self.logits, self.labels, self.document_lengths)

        if self.config.adv_reg_coeff:
            adv_loss = adversarial_loss(word_embeddings, self.loss, self.embedd_to_loss, self.config.adv_perturb_norm_length)
            self.loss += self.config.adv_reg_coeff * adv_loss
        if self.config.va_reg_coeff:
            va_loss = virtual_adversarial_loss(self.logits, word_embeddings, self.config.ntags, self.sentence_lengths,
                                               self.document_lengths,
                                                self.forward, self.config.va_perturb_norm_length)
            self.loss += self.config.va_reg_coeff * va_loss

        if self.config.va_reg_coeff and self.config.num_augmentation:
            word_embeddings_aug = tf.nn.embedding_lookup(word_embeddings_table,
                                                     self.word_aug_ids, name="word_embeddings")
            word_embeddings_aug = tf.nn.dropout(word_embeddings_aug, self.config.embedding_dropout)
            self.logits_aug = self.forward(word_embeddings_aug, self.sentence_lengths_aug, self.document_lengths_aug)
            va_loss_aug = virtual_adversarial_loss(self.logits_aug, word_embeddings_aug, self.config.ntags,
                                                    self.sentence_lengths_aug, self.document_lengths_aug,
                                                   self.forward, self.config.va_perturb_norm_length)
            self.loss += self.config.va_reg_coeff * va_loss_aug

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            document_length

        """
        fd, document_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, document_length in zip(logits, document_lengths):
                logit = logit[:document_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, document_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, document_lengths


    def run_epoch(self, train, dev, data_aug=None):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        if data_aug is None:
            for i, (words, labels) in enumerate(minibatches(train, batch_size)):
                fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                                           self.config.dropout)

                _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

                if not self.config.train_accuracy:
                    prog.update(i + 1, [("train loss", train_loss)])
                else:
                    labels_pred, document_lengths = self.predict_batch(words)
                    accs = []
                    for lab, lab_pred, length in zip(labels, labels_pred,
                                                     document_lengths):
                        lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs += [a == b for (a, b) in zip(lab, lab_pred)]
                    acc = np.mean(accs)
                    prog.update(i + 1, [("train loss", train_loss), ("accuracy", acc)])
        else:
            # for i, (a, b) in enumerate(zip(minibatches(train, batch_size), minibatches(data_aug, self.config.batch_size_aug))):
            # for a, b in minibatches(data_aug, self.config.batch_size_aug):
            #     print(i, len(a), len(b))
            #     break
            for i, ((words, labels), (words_aug, _)) in enumerate(zip(minibatches(train, batch_size),
                                                                minibatches(data_aug, self.config.batch_size_aug))):
                fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                        self.config.dropout, data_aug=words_aug)

                _, train_loss, summary = self.sess.run(
                        [self.train_op, self.loss, self.merged], feed_dict=fd)

                if not self.config.train_accuracy:
                    prog.update(i + 1, [("train loss", train_loss)])
                else:
                    labels_pred, document_lengths = self.predict_batch(words)
                    accs = []
                    for lab, lab_pred, length in zip(labels, labels_pred,
                                                     document_lengths):
                        lab      = lab[:length]
                        lab_pred = lab_pred[:length]
                        accs    += [a==b for (a, b) in zip(lab, lab_pred)]
                    acc = np.mean(accs)
                    prog.update(i + 1, [("train loss", train_loss), ("accuracy", acc)])

            # tensorboard
            # if i % 10 == 0:
                # self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev, report=True)
        msg = " - ".join(["{} {:04.4f}".format(k, v)
                    if k == 'acc' else '{} {}'.format(k, ', '.join(['{}: {:04.4f}'.format(a, b) \
                    for a, b in v.items()])) for k, v in metrics.items()])
        self.logger.info(msg)

        return np.mean(list(metrics["f1"].values()))


    def run_evaluate(self, test, report=False):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        labs = []
        labs_pred = []
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, document_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             document_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                # lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                # lab_pred_chunks = set(get_chunks(lab_pred,
                                                 # self.config.vocab_tags))

                # correct_preds += len(accs)
                # total_preds   += len(lab_pred)
                # total_correct += len(lab)

                labs.extend(lab)
                labs_pred.extend(lab_pred)

            labs_orig = [self.idx_to_tag[jj] for ii in labels for jj in ii]
            labs_pred_orig = [self.idx_to_tag[jj] for ii in labels_pred for jj in ii]
            sents_orig = [[self.idx_to_words[word] for word in sent] for abstract in words for sent in abstract]
            # print('\n'.join(['{}|{}|{}'.format(lab, pred, ' '.join(sent)) for lab, pred, sent in zip(labs_orig, labs_pred_orig, sents_orig)]))

        # p   = correct_preds / total_preds if correct_preds > 0 else 0
        # r   = correct_preds / total_correct if correct_preds > 0 else 0
        # f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        precision, recall, f1, _ = precision_recall_fscore_support(labs, labs_pred)
        acc = np.mean(accs)

        if report:
            print(classification_report(labs, labs_pred, target_names=self.target_names, digits=5))
            print(self.idx_to_tag)
            print(confusion_matrix(labs, labs_pred))

        return {"acc": 100*acc, 
                'precision': {tag: precision[self.config.vocab_tags[tag]] for tag in ['P', 'I', 'O']},
                'recall': {tag: recall[self.config.vocab_tags[tag]] for tag in ['P', 'I', 'O']},
                'f1': {tag: f1[self.config.vocab_tags[tag]] for tag in ['P', 'I', 'O']},
                'precision_all': {tag: precision[self.config.vocab_tags[tag]] for tag in self.config.vocab_tags.keys()},
                'recall_all': {tag: recall[self.config.vocab_tags[tag]] for tag in self.config.vocab_tags.keys()},
                'f1_all': {tag: f1[self.config.vocab_tags[tag]] for tag in self.config.vocab_tags.keys()}
                }


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
