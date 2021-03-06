"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from bert_model import BertForSequentialClassification
from adversarial_losses import adversarial_loss, virtual_adversarial_loss

import torch
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, document, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.document = document
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, document_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.document_mask = document_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_corpus(cls, input_file):
        sentences, tags = [], []
        documents = []
        n_iters = 0
        with open(input_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    n_iters += 1
                    if sentences:
                        # if n_iters > 10:
                            # break
                        documents.append((sentences, tags))
                        sentences, tags = [], []
                elif not line.startswith("###"):
                    ls = line.split('|')
                    tag, sentence = ls[1], ls[2]
                    sentences.append(sentence)
                    tags.append(tag)

        return documents


class PICOProcessor(DataProcessor):
    """Processor for the PICO dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_corpus(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_corpus(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_corpus(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["A", "M", "P", "I", "O", "R", "C"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, document=line[0], label=line[1]))
        return examples


class NICTAProcessor(DataProcessor):
    """Processor for the PICO dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_corpus(os.path.join(data_dir, "train.txt")), "train")

    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_corpus(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_corpus(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["B", "P", "I", "O", "S", "OT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, document=line[0], label=line[1]))
        return examples


def batch_convert_examples_to_features(examples, label_map, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    input_ids, label_ids = [], []
    document_lens = [len(example.document) for example in examples]
    max_doc_len = max(document_lens)
    document_mask = [[1] * document_len + [0] * (max_doc_len - document_len) for document_len in document_lens]

    sent_lens = []
    for (ex_index, example) in enumerate(examples):
        tokens = [["[CLS]"] + tokenizer.tokenize(sent)[:max_seq_length] + ["[SEP]"] for sent in example.document]

        input_id = [tokenizer.convert_tokens_to_ids(seq) for seq in tokens]
        input_id += [[] for _ in range(max_doc_len - len(input_id))]

        label_id = [label_map[lab] for lab in example.label] + [0] * (max_doc_len - len(example.label))

        sent_lens.extend(list(map(len, input_id)))

        input_ids.extend(input_id)
        label_ids.append(label_id)

    max_sent_len = max(sent_lens)
    # print(max_doc_len, max_sent_len, input_ids)
    segment_ids = [[0] * max_sent_len for _ in range(max_doc_len * len(examples))]
    input_mask = [[1] * sent_len + [0] * (max_sent_len - sent_len) for sent_len in sent_lens]
    input_ids = [input_id + [0] * (max_sent_len - sent_len) for input_id, sent_len in zip(input_ids, sent_lens)]

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_ids=label_ids,
                         document_mask=document_mask)


def minibatches(data, label_map, tokenizer, minibatch_size, max_seq_length, shuffle=False):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    if shuffle:
        random.shuffle(data)

    batch = []
    for item in data:
        if len(batch) == minibatch_size:
            batch_features = batch_convert_examples_to_features(batch, label_map, tokenizer, max_seq_length)
            yield batch_features
            batch = []

        batch += [item]

    if batch:
        batch_features = batch_convert_examples_to_features(batch, label_map, tokenizer, max_seq_length)
        yield batch_features


def accuracy(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = np.mean(labels == preds)

    return acc, precision, recall, f1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=32,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--tag_space",
                        default=128,
                        type=int,
                        help="dimension of linear transformation.")
    parser.add_argument("--rnn_hidden_size",
                        default=None,
                        type=int,
                        help="dimension of document level rnn layer.")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="dropout for outputs other than the original bert model")
    parser.add_argument("--use_crf",
                        action='store_true',
                        help="Whether to use crf layer.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_resume",
                        action='store_true',
                        help="Whether to run eval on the resumed pretrained model.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument('--adv_reg_coeff', default='0.0', type=float,
                        help='Regularization coefficient of adversarial loss')
    parser.add_argument('--va_reg_coeff', default='0.0', type=float,
                        help='Regularization coefficient of virtual adversarial loss')
    parser.add_argument('--adv_perturb_norm_length', default='8.0', type=float,
                        help='Norm length of adversarial perturbation to be')
    parser.add_argument('--va_perturb_norm_length', default='4.0', type=float,
                        help='Norm length of virtual adversarial perturbation to be')
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "pico": PICOProcessor,
        "nicta": NICTAProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_map)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        num_train_optimization_steps_epoch = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps)
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = BertForSequentialClassification.from_pretrained(args.bert_model,
                                                            cache_dir=cache_dir,
                                                            num_labels=num_labels,
                                                            tag_space=args.tag_space,
                                                            use_crf=args.use_crf,
                                                            rnn_hidden_size=args.rnn_hidden_size,
                                                            dropout=args.dropout)
    # print(count_parameters(model))
    # if args.fp16:
    #     model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        # if args.fp16:
        #     try:
        #         from apex.optimizers import FP16_Optimizer
        #         from apex.optimizers import FusedAdam
        #     except ImportError:
        #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        #
        #     optimizer = FusedAdam(optimizer_grouped_parameters,
        #                           lr=args.learning_rate,
        #                           bias_correction=False,
        #                           max_grad_norm=1.0)
        #     if args.loss_scale == 0:
        #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        #     else:
        #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        #
        # else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(minibatches(train_examples, label_map, tokenizer,
                                                          args.train_batch_size,
                                                          args.max_seq_length,
                                                          shuffle=True), desc="Iteration",
                                              total=num_train_optimization_steps_epoch)):
                input_ids = torch.tensor(batch.input_ids, dtype=torch.long).to(device)
                segment_ids = torch.tensor(batch.segment_ids, dtype=torch.long).to(device)
                input_mask = torch.tensor(batch.input_mask, dtype=torch.long).to(device)
                label_ids = torch.tensor(batch.label_ids, dtype=torch.long).to(device)
                document_mask = torch.tensor(batch.document_mask, dtype=torch.float).to(device)
                loss, logits, embeddings = model(input_ids, segment_ids, input_mask, document_mask, label_ids)
                if args.adv_reg_coeff:
                    adv_loss = adversarial_loss(embeddings, segment_ids, input_mask, document_mask, label_ids,
                                                loss, model, args.adv_perturb_norm_length)[0]
                    loss += args.adv_reg_coeff * adv_loss
                if args.va_reg_coeff:
                    va_loss = virtual_adversarial_loss(logits, embeddings, segment_ids, input_mask, document_mask,
                                                       num_labels, model, args.va_perturb_norm_length)
                    loss += args.va_reg_coeff * va_loss
                # if n_gpu > 1:
                #     loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # if args.fp16:
                    #     # modify learning rate with special warm up BERT uses
                    #     # if args.fp16 is False, BertAdam is used that handles this automatically
                    #     lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps,
                    #                                                       args.warmup_proportion)
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForSequentialClassification(config, num_labels=num_labels,
                                                tag_space=args.tag_space,
                                                use_crf=args.use_crf,
                                                rnn_hidden_size=args.rnn_hidden_size,
                                                dropout=args.dropout)
        model.load_state_dict(torch.load(output_model_file))
    elif args.do_resume:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForSequentialClassification(config, num_labels=num_labels,
                                                tag_space=args.tag_space,
                                                use_crf=args.use_crf,
                                                rnn_hidden_size=args.rnn_hidden_size,
                                                dropout=args.dropout)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForSequentialClassification.from_pretrained(args.bert_model, num_labels=num_labels,
                                                                tag_space=args.tag_space,
                                                                use_crf=args.use_crf,
                                                                rnn_hidden_size=args.rnn_hidden_size,
                                                                dropout=args.dropout)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir) ## eval on dev/test sets

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        preds_all = []
        labels_all = []

        for step, batch in enumerate(tqdm(minibatches(eval_examples, label_map, tokenizer,
                                                      args.eval_batch_size,
                                                      args.max_seq_length,
                                                      shuffle=False), desc="Evaluating")):
            input_ids = torch.tensor(batch.input_ids, dtype=torch.long).to(device)
            segment_ids = torch.tensor(batch.segment_ids, dtype=torch.long).to(device)
            input_mask = torch.tensor(batch.input_mask, dtype=torch.long).to(device)
            document_mask = torch.tensor(batch.document_mask, dtype=torch.float).to(device)

            with torch.no_grad():
                preds, _, _ = model(input_ids, segment_ids, input_mask, document_mask)

            preds = preds.cpu().tolist()
            document_lens = np.sum(batch.document_mask, axis=1)
            for pred, label, document_len in zip(preds, batch.label_ids, document_lens):
                preds_all += pred[:document_len]
                labels_all += label[:document_len]

        eval_acc, eval_prec, eval_recall, eval_f1 = accuracy(preds_all, labels_all)
        print(confusion_matrix(labels_all, preds_all))
        eval_sents = [sent for example in eval_examples for sent in example.document]
        with open(os.path.join(args.output_dir, 'eval_text'), 'w') as ofile:
            for sent, pred, label in zip(eval_sents, preds_all, labels_all):
                ofile.write('{}\t{}\t{}\n'.format(label_list[label], label_list[pred], sent))

        loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {
                  'global_step': global_step,
                  'loss': loss}
        for tag in processor.get_labels():
            result.update({tag: {"precision": eval_prec[label_map[tag]],
                                 "recall": eval_recall[label_map[tag]],
                                 "f1": eval_f1[label_map[tag]]}})

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        fold_num = args.output_dir.split('/')[-1]
        params_log = ', '.join(['{}: {}'.format(attr, getattr(args, attr)) for attr in dir(args) \
                   if not callable(getattr(args, attr)) and not attr.startswith("__")])
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results for Fold {}*****".format(fold_num))
            writer.write("\n***** Eval results for Fold {}*****\n".format(fold_num))
            writer.write(params_log+'\n')
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()