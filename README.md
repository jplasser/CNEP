![CNEP Logo](cneplogo.png)

# CNEP
CNEP (Contrastive Notes Events Pretraining), Contrastive Learning with Clinical Notes and Events Data
Pretraining from MIMIC-III.

## What is CNEP?
CNEP is a descendant of CLIP, the OpenAI model, and it operates on multimodal data, same as CLIP, but on text and
event data rather than images and text.
Otherwise, CNEP differs mainly in the encoders used and, of course, in the training data.

## Master's Thesis
Currently, this is being worked on as part of a master's thesis.
Find out more in the weeks coming.

### Sample running code:

```bash
nohup python -u src/training/main.py \
--save-frequency 1 \
--zeroshot-frequency 1 \
--report-to tensorboard \
--train-data="./data/mimic3/full_train_data.pickle" \
--val-data="./data/mimic3/full_val_data.pickle" \
--dataset-type mimic \
--warmup 1000 \
--batch-size=32 \
--lr=1e-3 \
--wd=0.1 \
--epochs=3 \
--workers=1 \
--model LSTMCNN
```

### Launch tensorboard:
```bash
tensorboard --logdir=logs/tensorboard/ --port=7777
```
