from src.data_utils import Dataset
from src.models import HANNModel
from src.config import Config
import argparse
import os

parser = argparse.ArgumentParser()

def main():
    # create instance of config
    config = Config(parser)
    assert config.data_keyname == 'pico'

    # build model
    model = HANNModel(config)
    model.build()

    # create datasets
    dev = Dataset(config.filename_dev, config.processing_word,
                         config.processing_tag)
    train = Dataset(config.filename_train, config.processing_word,
                         config.processing_tag)
    test = Dataset(config.filename_test, config.processing_word,
                         config.processing_tag)
    if config.num_augmentation:
        data_aug = Dataset(config.filename_aug, config.processing_word, max_iter=config.num_augmentation)
    else:
        data_aug = None

    # train model
    model.train(train, dev, data_aug)

    # evaluate model
    model.restore_session(config.dir_model)
    model.evaluate(test)

if __name__ == "__main__":
    main()
