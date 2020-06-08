from src.data_utils import Dataset
from src.models import HANNModel
from src.config import Config
import argparse
import os
import numpy as np
from collections import defaultdict


def main():
    # create instance of config
    config = Config()
    assert config.data_keyname == 'nicta'
    config.num_augmentation = 200000
    config.batch_size = 20
    config.batch_size_aug = 20
    config.attention_size = 50
    config.hidden_size_lstm_document = 200
    config.dropout = 0.8
    config.cnn_filter_num = 150
    config.adv_perturb_norm_length = 4
    config.va_perturb_norm_length = 4
    config.adv_reg_coeff = 0.3
    config.va_reg_coeff = 0.3
    config.data_root = '../data/nicta_piboso/10_folds'
    config.dir_output = 'results/nicta/test-num_augmentation-{}-va_coeff-{}-adv-coeff-{}'.format(config.num_augmentation,
                                                                                                 config.va_reg_coeff,
                                                                                                 config.adv_reg_coeff)
    config.dir_model = os.path.join(config.dir_output, "model.weights")

    result_file_path = os.path.join(config.dir_output, 'cross_validate_results')

    precisions = defaultdict(list)
    recalls = defaultdict(list)
    f1s = defaultdict(list)
    tag_ls = ['P', 'I', 'O', 'S', 'B', 'OT']

    for fold in range(1, 11):
        # build model
        # tf.reset_default_graph()
        print('Fold {}'.format(fold))

        # build model
        model = HANNModel(config)
        model.build()
        # if config.restore:
        # model.restore_session("results/test/model.weights/") # optional, restore weights
        # model.reinitialize_weights("proj")

        # create datasets
        train = Dataset(os.path.join(config.data_root, str(fold), 'train.txt'), config.processing_word,
                        config.processing_tag)
        dev = Dataset(os.path.join(config.data_root, str(fold), 'test.txt'), config.processing_word,
                      config.processing_tag)
        test = Dataset(os.path.join(config.data_root, str(fold), 'test.txt'), config.processing_word,
                       config.processing_tag)
        if config.num_augmentation:
            data_aug = Dataset(config.filename_aug, config.processing_word, max_iter=config.num_augmentation)
        else:
            data_aug = None

        # train model
        model.train(train, dev, data_aug)

        # evaluate model
        model.restore_session(config.dir_model)
        metrics = model.evaluate(test)

        [precisions[tag].append(metrics['precision_all'][tag]) for tag in tag_ls]
        [recalls[tag].append(metrics['recall_all'][tag]) for tag in tag_ls]
        [f1s[tag].append(metrics['f1_all'][tag]) for tag in tag_ls]
        msg = 'fold: {}\tprecision: {}\trecall: {}\tf1: {}\n'.format(fold, metrics['precision_all'],
                                                                     metrics['recall_all'], metrics['f1_all'])
        print(msg)
        with open(result_file_path, 'a') as ofile:
            ofile.write(msg)

    msg = 'Average Precision: {}'.format('\t'.join(['{}: {}'.format(tag, np.mean(precisions[tag])) for tag in tag_ls]))
    print(msg)
    with open(result_file_path, 'a') as ofile:
        ofile.write(msg)
    msg = 'Average Recall: {}'.format('\t'.join(['{}: {}'.format(tag, np.mean(recalls[tag])) for tag in tag_ls]))
    print(msg)
    with open(result_file_path, 'a') as ofile:
        ofile.write(msg)
    res = np.mean([np.mean(values) for values in f1s.values()])
    msg = 'Average F1: {}'.format('\t'.join(['{}: {}'.format(tag, np.mean(f1s[tag])) for tag in tag_ls]))
    print(msg)
    with open(result_file_path, 'a') as ofile:
        ofile.write(msg)
        ofile.write('\n\n\n')

if __name__ == "__main__":
    main()
