# Deep-PICO-Detection
A model for identifying PICO elements in a given biomedical/clinical text.

This is the source code for the paper: [Di Jin, Peter Szolovits, Advancing PICO Element Detection in Biomedical Text via Deep Neural Networks, Bioinformatics, , btaa256](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaa256/5822877?guestAccessKey=7f54ea86-4ec0-4080-9d5c-1251b730aa42). If you use the code, please cite the paper:

```
@article{10.1093/bioinformatics/btaa256,
    author = {Jin, Di and Szolovits, Peter},
    title = "{Advancing PICO element detection in biomedical text via deep neural networks}",
    journal = {Bioinformatics},
    year = {2020},
    month = {04},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa256},
    url = {https://doi.org/10.1093/bioinformatics/btaa256},
    note = {btaa256},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btaa256/33363807/btaa256.pdf},
}
```

## Prerequisites:
Run the following command to install the prerequisite packages:
```
pip install -r requirements.txt
```

## Data:
Please download the data including PICO and NICTA-PIBOSO from the [Google Drive](https://drive.google.com/file/d/1M9QCgrRjERZnD9LM2FeK-3jjvXJbjRTl/view?usp=sharing) and unzip it to the main directory of this repository so that the folder layout is like this:
```
./BERT
./lstm_model
./data
```

## How to use
### For LSTM based models
* The code for the LSTM based models is in the folder of "lstm_model", so run the following command to enter it:
```
cd lstm_model
```

* First we need to process the data to get vocabulary and trim the embedding file. The embeddings we used in experiments are from [here](http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin). Please download it and convert it to "txt" format. Of course, you can also try other kinds of embeddings such as fasttext. Then run the following command:
```
python build_data.py --data_keyname DATA_KEYNAME --filename_wordvec PATH_TO_EMBEDDING_FILE
```
DATA_KEYNAME can be "pico" for the PICO dataset and "nicta" for the NICTA-PIBOSO dataset; PATH_TO_EMBEDDING_FILE specifies where you store the embedding file.

* Then we can start training the model for the PICO dataset by running the following command:
```
python run_train_pico.py --data_keyname pico
```
And the following command is for the NICTA-PIBOSO dataset:
```
python run_train_nicta.py --data_keyname nicta
```

* If we want to implement the 10 fold cross-validation, we run the following commands:
```
python run_train_cross_validate_pico.py --data_keyname pico
python run_train_cross_validate_nicta.py --data_keyname nicta
```

There are several important arguments in the file of "src/config.py" that configures the model architecture and they are explains here:

* --adv_reg_coeff: The coefficient for the adversarial loss regularization. Setting it to zero means we do not conduct the adversarial training.
* --va_reg_coeff: The coefficient for the virtual adversarial loss regularization. Setting it to zero means we do not conduct the virtual adversarial training.
* --num_augmentation: The number of samples we use for the virtual adversarial training.

### For the BERT Models
* Code for the BERT models is in the folder of "BERT" and please enter this folder.

* The best BERT model we found is the [BioBERT model](https://github.com/dmis-lab/biobert). The pretrained model parameter files available in the original repository only have the tensorflow version, and if you want the pytorch version, you can download from [here](https://drive.google.com/file/d/1H6DTBXlXDZ6tJYcJWdZnZ3UCoY16p19m/view?usp=sharing). Once you obtain the pretrained BERT model file, run the following commands for training:
```
python run_classifier_pico.py PATH_TO_BERT_MODEL
python run_classifier_nicta.py PATH_TO_BERT_MODEL
```
In this command, PATH_TO_BERT_MODEL specifies the directory where you put your downloaded BERT model files. 

* The following commands are for the 10-fold cross-validation:
```
python run_classifier_pico_cross_validate.py PATH_TO_BERT_MODEL
python run_classifier_nicta_cross_validate.py PATH_TO_BERT_MODEL
```
