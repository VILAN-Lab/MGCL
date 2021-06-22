# MGCL: Image-guided Story Ending Generation

##### Code for paper "IgSEG: Image-guided Story Ending Generation", Findings of ACL 2021.

## Prerequisites

- Python 3.6
- PyTorch == 1.14.0
- l stanfordcorenlp 3.9.1.1
- l six 1.14.0
- l numpy 1.18.5
- l tensorboard 2.2.1
- 

## Quick start:

- Dataset

All dataset under the directory of `data`, download the SIS-with-labels.tar.gz (https://visionandlanguage.net/VIST/dataset.html),  download the image features (https://vist-arel.s3.amazonaws.com/resnet_features.zip) and put then in directory of `data`. We utilize the glove embedding, please download the *glove.6b.300d.txt* and put it in directory of `data/embedding`.

- Data preprocess

Run following command:

1. python data/annotations.py 
2. `python data/img_feat_path.py `
3. python data/pro_label.py 
4. python data/embed_vocab.py 

Then we will get three files

1. data/`data_res.json`
2. data/data_adj_label.h5
3. data/data_src_label.h5
4. data/data_tgt_label.h5
5. `data/embedding/embedding_enc.pt`
6. `data/embedding/embedding_dec.pt`

- Training

Run command:

`python train.py -gpu 0 `

- Inference

Run command:

`python eval.py -gpu 0`

Then the output file will be save in directory of data/gen
