import os
import sys

MODEL_PATH = sys.argv[1]

for fold in range(1, 11):
    # # use bio-bert
    command = 'python bert_classifier.py --data_dir ../data/pico/10_folds/{fold} ' \
              '--bert_model {MODEL_PATH} ' \
              '--task_name pico --output_dir results/PICO/biobert_crf_{fold} ' \
              '--train_batch_size 2 --tag_space 0 --max_seq_length 60 --use_crf ' \
              '--do_train --do_eval --do_lower_case --num_train_epochs 3 ' \
              '--rnn_hidden_size 512 --dropout 0.2 '.format(fold=fold,
                                                            MODEL_PATH=MODEL_PATH)

    os.system(command)