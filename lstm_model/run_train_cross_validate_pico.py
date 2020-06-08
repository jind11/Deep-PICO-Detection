from src.data_utils import Dataset
from src.models import HANNModel
from src.config import Config
import argparse
import os
import numpy as np


def main():
    # create instance of config
    config = Config()
    assert config.data_keyname == 'pico'
    config.num_augmentation = 0
    config.batch_size = 20
    config.batch_size_aug = 20
    config.dir_output = 'test-num_augmentation-{}'.format(config.num_augmentation)
    config.dir_model = os.path.join(config.dir_output, "model.weights")
    config.data_root = '../data/{}/10_folds'.format(config.data_keyname)

    result_file_path = os.path.join(config.dir_output, 'cross_validate_results')

    precisions = {'P': [], 'I': [], 'O': []}
    recalls = {'P': [], 'I': [], 'O': []}
    f1s = {'P': [], 'I': [], 'O': []}

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
        dev = Dataset(os.path.join(config.data_root, str(fold), 'dev.txt'), config.processing_word,
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

        [precisions[tag].append(metrics['precision'][tag]) for tag in ['P', 'I', 'O']]
        [recalls[tag].append(metrics['recall'][tag]) for tag in ['P', 'I', 'O']]
        [f1s[tag].append(metrics['f1'][tag]) for tag in ['P', 'I', 'O']]
        msg = 'fold: {}\tprecision: {}\trecall: {}\tf1: {}\n'.format(fold, metrics['precision'], metrics['recall'], metrics['f1'])
        print(msg)
        with open(result_file_path, 'a') as ofile:
            ofile.write(msg)


    # print('Precision: ', 'P: ', (precisions['P']), 'I: ', (precisions['I']), 'O: ', (precisions['O']))
    # print('Recall: ', 'P: ', (recalls['P']), 'I: ', (recalls['I']), 'O: ', (recalls['O']))
    # print('F1: ', 'P: ', (f1s['P']), 'I: ', (f1s['I']), 'O: ', (f1s['O']))
    # print('Precision: ', 'P: ', np.mean(precisions['P']), 'I: ', np.mean(precisions['I']), 'O: ', np.mean(precisions['O']))
    # print('Recall: ', 'P: ', np.mean(recalls['P']), 'I: ', np.mean(recalls['I']), 'O: ', np.mean(recalls['O']))
    # res = np.mean([np.mean(values) for values in f1s.values()])
    # print('F1: ', 'P: ', np.mean(f1s['P']), 'I: ', np.mean(f1s['I']), 'O: ', np.mean(f1s['O']), 'all avg: ', res)
    msg = 'Average Precision: P: {}\tI: {}\tO: {}\n'.format(np.mean(precisions['P']), np.mean(precisions['I']), np.mean(precisions['O']))
    print(msg)
    with open(result_file_path, 'a') as ofile:
        ofile.write(msg)
    msg = 'Average Recall: P: {}\tI: {}\tO: {}\n'.format(np.mean(recalls['P']), np.mean(recalls['I']), np.mean(recalls['O']))
    print(msg)
    with open(result_file_path, 'a') as ofile:
        ofile.write(msg)
    res = np.mean([np.mean(values) for values in f1s.values()])
    msg = 'Average F1: P: {}\tI: {}\tO: {}\tall: {}\n'.format(np.mean(f1s['P']), np.mean(f1s['I']), np.mean(f1s['O']), res)
    print(msg)
    with open(result_file_path, 'a') as ofile:
        ofile.write(msg)
        ofile.write('\n\n\n')

if __name__ == "__main__":
    main()
