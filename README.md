![CNEP Logo](cneplogo.png)

# Contrastive Notes Events Pre-training (CNEP)

Code to reproduce the experiments in the master's thesis
"Multimodal Contrastive Pre-Training on a Medical Benchmark Dataset (MIMIC-III)".

## What is CNEP?
CNEP is a variant of CLIP, the OpenAI model, and it operates on multimodal data,
similar to CLIP, but on text and
chart events data rather than image/text-pairs.

CNEP has been trained on the MIMIC-III dataset.

## Master's Thesis
![Master's Thesis, Multimodal Contrastive Pre-Training on a Medical Benchmark Dataset (MIMIC-III), Jürgen R. Plasser, 2022](Master'sThesis_JürgenRichardPlasser_k8956888.pdf-001.png)

#### Download PDF
[Master's Thesis, Multimodal Contrastive Pre-Training on a Medical Benchmark Dataset (MIMIC-III), Jürgen R. Plasser, 2022](Master'sThesis_JürgenRichardPlasser_k8956888.pdf)

<details><summary>Usage</summary>
<p>

### Usage
```
usage: main.py [-h] [--train-data TRAIN_DATA] [--val-data VAL_DATA]
               [--dataset-type {webdataset,csv,auto,mimic,mimic-emb}]
               [--csv-separator CSV_SEPARATOR] [--csv-img-key CSV_IMG_KEY]
               [--csv-caption-key CSV_CAPTION_KEY]
               [--imagenet-val IMAGENET_VAL] [--imagenet-v2 IMAGENET_V2]
               [--mimic3-val MIMIC3_VAL] [--logs LOGS] [--name NAME]
               [--workers WORKERS] [--batch-size BATCH_SIZE]
               [--batch-size-eval BATCH_SIZE_EVAL] [--epochs EPOCHS] [--lr LR]
               [--beta1 BETA1] [--beta2 BETA2] [--eps EPS] [--wd WD]
               [--warmup WARMUP] [--lr-scheduler {cosine,cosine-restarts}]
               [--restart-cycles RESTART_CYCLES] [--use-bn-sync] [--gpu GPU]
               [--skip-scheduler] [--save-frequency SAVE_FREQUENCY]
               [--save-most-recent] [--zeroshot-frequency ZEROSHOT_FREQUENCY]
               [--regression-frequency REGRESSION_FREQUENCY] [--resume RESUME]
               [--resume-pretrained RESUME_PRETRAINED]
               [--precision {amp,fp16,fp32}]
               [--model {RN50,RN101,RN50x4,ViT-B/32,LSTMCNN,LSTMCNN-SE,LSTMCNN-EMB}]
               [--openai-pretrained] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--skip-aggregate]
               [--report-to REPORT_TO] [--wandb-notes WANDB_NOTES] [--C C]
               [--debug] [--debug-run] [--copy-codebase] [--dp]
               [--multigpu MULTIGPU]
               [--text-embedding-dimension TEXT_EMBEDDING_DIMENSION]
               [--omit-embeddings] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --train-data TRAIN_DATA
                        Path to csv filewith training data
  --val-data VAL_DATA   Path to csv file with validation data
  --dataset-type {webdataset,csv,auto,mimic,mimic-emb}
                        Which type of dataset to process.
  --csv-separator CSV_SEPARATOR
                        For csv-like datasets, which separator to use.
  --csv-img-key CSV_IMG_KEY
                        For csv-like datasets, the name of the key for the
                        image paths.
  --csv-caption-key CSV_CAPTION_KEY
                        For csv-like datasets, the name of the key for the
                        captions.
  --imagenet-val IMAGENET_VAL
                        Path to imagenet val set for conducting zero shot
                        evaluation.
  --imagenet-v2 IMAGENET_V2
                        Path to imagenet v2 for conducting zero shot
                        evaluation.
  --mimic3-val MIMIC3_VAL
                        Path to MIMIC3 val or test set for conducting zero
                        shot evaluation.
  --logs LOGS           Where to store tensorboard logs. Use None to avoid
                        storing logs.
  --name NAME           Optional identifier for the experiment when storing
                        logs. Otherwise use current time.
  --workers WORKERS     Number of workers per GPU.
  --batch-size BATCH_SIZE
                        Batch size per GPU.
  --batch-size-eval BATCH_SIZE_EVAL
                        Batch size during evaluation (on one GPU).
  --epochs EPOCHS       Number of epochs to train for.
  --lr LR               Learning rate.
  --beta1 BETA1         Adam beta 1.
  --beta2 BETA2         Adam beta 2.
  --eps EPS             Adam epsilon.
  --wd WD               Weight decay.
  --warmup WARMUP       Number of steps to warmup for.
  --lr-scheduler {cosine,cosine-restarts}
                        LR scheduler
  --restart-cycles RESTART_CYCLES
                        Number of restarts when using LR scheduler with
                        restarts
  --use-bn-sync         Whether to use batch norm sync.
  --gpu GPU             Specify a single GPU to run the code on for
                        debugging.Leave at None to use all available GPUs.
  --skip-scheduler      Use this flag to skip the learning rate decay.
  --save-frequency SAVE_FREQUENCY
                        How often to save checkpoints.
  --save-most-recent    Always save the most recent model trained to
                        epoch_latest.pt.
  --zeroshot-frequency ZEROSHOT_FREQUENCY
                        How often to run zero shot.
  --regression-frequency REGRESSION_FREQUENCY
                        How often to run zero shot.
  --resume RESUME       path to latest checkpoint (default: none)
  --resume-pretrained RESUME_PRETRAINED
                        resume from pretrained checkpoint, path to latest
                        checkpoint (default: none)
  --precision {amp,fp16,fp32}
                        Floating point precition.
  --model {RN50,RN101,RN50x4,ViT-B/32,LSTMCNN,LSTMCNN-SE,LSTMCNN-EMB}
                        Name of the vision backbone to use.
  --openai-pretrained   Use the openai pretrained models.
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --skip-aggregate      whether to aggregate features across gpus before
                        computing the loss
  --report-to REPORT_TO
                        Options are ['wandb', 'tensorboard',
                        'wandb,tensorboard']
  --wandb-notes WANDB_NOTES
                        Notes if logging with wandb
  --C C                 inverse regularizer for logistic reg.
  --debug               If true, more information is logged.
  --debug-run           If true, only subset of data is used.
  --copy-codebase       If true, we copy the entire base on the log diretory,
                        and execute from there.
  --dp                  Use DP instead of DDP.
  --multigpu MULTIGPU   In DP, which GPUs to use for multigpu training
  --text-embedding-dimension TEXT_EMBEDDING_DIMENSION
                        Dimension of the pre-computed text embeddings.
  --omit-embeddings     omit text embeddings for the EventsEncoder model
  --seed SEED           Seed for reproducibility
```

</p>
</details>

<details><summary>Sample Code Training</summary>
<p>

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
</p>
</details>

<details><summary>Working Example</summary>
<p>

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

##### Launch tensorboard:
```bash
tensorboard --logdir=logs/tensorboard/ --port=7777
```

</p>
</details>

<details><summary>Data Preparation</summary>
<p>

### Data Preparation

**Note: Due to MIMIC-III's restrictive access policy,
datasets used to train CNEP are not available online and may not be shared.**

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

</p>
</details>

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
