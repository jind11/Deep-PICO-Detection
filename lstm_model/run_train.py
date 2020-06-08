from model.data_utils import Dataset
from model.models import HANNModel
from model.config import Config
import argparse
import os

parser = argparse.ArgumentParser()

def main():
    # create instance of config
    config = Config(parser)
    config.num_augmentation = 20000
    config.batch_size = 20
    config.batch_size_aug = 20
    config.dir_output = 'test-num_augmentation-{}-2'.format(config.num_augmentation)
    config.dir_model = os.path.join(config.dir_output, "model.weights")

    # build model
    model = HANNModel(config)
    model.build()
    # if config.restore:
        # model.restore_session("results/test/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = Dataset(config.filename_dev, config.processing_word,
                         config.processing_tag)
    train = Dataset(config.filename_train, config.processing_word,
                         config.processing_tag)
    test  = Dataset(config.filename_test, config.processing_word,
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
