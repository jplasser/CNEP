![CNEP Logo](cneplogo.png)

# Contrastive Notes Events Pre-training (CNEP),

Code to reproduce the experiments in the Master's Thesis "Multimodal Contrastive Pre-Training on a Medical Benchmark Dataset (MIMIC-III)".

## What is CNEP?
CNEP is a variant of CLIP, the OpenAI model, and it operates on multimodal data,
similar to CLIP, but on text and
chart events data, rather than image/text-pairs.

CNEP has been trained on the MIMIC-III dataset.

## Master's Thesis
[Master's Thesis, Multimodal Contrastive Pre-Training on a Medical Benchmark Dataset (MIMIC-III), Jürgen R. Plasser, 2022](Master'sThesis_JürgenRichardPlasser_k8956888.pdf)

### Sample Code Training

```bash
nohup python -u src/training/main.py \
--save-frequency=1 \
--report-to=all \
--wandb-notes="<notes>" \
--train-data="<data pickle file>" \
--val-data="<data pickle file>" \
--dataset-type=mimic-emb \
--warmup=1500 \
--batch-size=128 \
--lr=1.4142e-2 \
--wd=1.e-3 \
--epochs=45 \
--gpu=0 \
--workers=1 \
--model=<LSTMCNN | LSTMCNN-SE | LSTMCNN-EMB> \
--lr-scheduler=cosine \
--batch-size-eval=128 \
--text-embedding-dimension=<700 | 768 | 1280>
```

#### Working Example
The parameter ```model``` valued with ```LSTMCNN-EMB``` ensures that the training
procedure utilizes the CNEP models with pre-trained representations.
To get the right embedding model the model-specific Pickle file hast to be applied
to the parameters ```train-data``` and ```val-data```.

```bash
nohup python -u src/training/main.py \
--save-frequency=1 \
--report-to=all \
--wandb-notes="Sent2Vec model" \
--train-data="./data/mimic3/new_extended_data_unique_embed_s2v.pickle" \
--val-data="./data/mimic3/new_test_data_unique_embed_s2v.pickle" \
--dataset-type=mimic-emb \
--warmup=1500 \
--batch-size=128 \
--lr=1.4142e-2 \
--wd=1.e-3 \
--epochs=45 \
--gpu=0 \
--workers=1 \
--model=LSTMCNN-EMB \
--lr-scheduler=cosine \
--batch-size-eval=128 \
--text-embedding-dimension=700
```

### Data Preparation
To prepare the datasets accordingly,
the following notebooks have to be run in consecutive order:

1. ```notebooks/mimic_data_preparation_1.ipynb```
2. ```notebooks/mimic_data_remove_duplicates_2.ipynb```
3. ```notebooks/mimic_data_preprocessing_2-1.ipynb```
4. ```notebooks/mimic_data_compute_embeddings_3.ipynb```
5. ```notebooks/mimic_data_preparation_no_discharge_notes_4.ipynb```

#### Data sets from the precursor work
Data sets from the following paper were used for this work:
Yu Wei Lin et al. “Analysis and prediction of unplanned intensive care unit readmission using recurrent neural networks with long shortterm memory”. In: PLoS ONE 14.7 (2019), p. 22. ISSN: 19326203. DOI: 10 . 1371 / journal . pone . 0218942

Original code from the paper: https://github.com/Jeffreylin0925/MIMIC-III_ICU_Readmission_Analysis

Adopted code for PyTorch: https://github.com/jplasser/MIMIC-III_ICU_Readmission_Analysis

### Launch tensorboard:
```bash
tensorboard --logdir=logs/tensorboard/ --port=7777
```

## Acknowledgments
Many thanks to OpenAI for building the CLIP model
and the great people involved in building an open source implementation
of CLIP, [Open CLIP](https://github.com/mlfoundations/open_clip) and making it publicly available.

I also want to thank Hugging Face for granting access to the pre-trained models used in this work.

## Citation
I recommend the following citation (unpublished, will be updated when published)

```bib
@mastersthesis{plasser2022cnep,
  title   = {Multimodal Contrastive Pre-Training on a Medical Benchmark Dataset (MIMIC-III)},
  author  = {Plasser, Jürgen Richard},
  school  = {Johannes Kepler University},
  year    = {2022},
  month   = aug,
  note    = {[Unpublished master's thesis]},
  eprint  = {unpublished}
}
```
