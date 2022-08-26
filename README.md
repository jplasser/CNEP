![CNEP Logo](cneplogo.png)

# Contrastive Notes Events Pretraining (CNEP),

Code to reproduce the experiments in the Master's Thesis "Multimodal Contrastive Pre-Training on a Medical Benchmark Dataset (MIMIC-III)".

## What is CNEP?
CNEP is a variant of CLIP, the OpenAI model, and it operates on multimodal data, same as CLIP, but on text and
chart events data, rather than image and text-pairs.

CNEP ist trained on the MIMIC-III dataset.

## Master's Thesis
tbd

### Sample Code Training

```bash
nohup python -u src/training/main.py \
--save-frequency 1 \
--zeroshot-frequency 1 \
--report-to all \
--train-data="./data/mimic3/full_train_data.pickle" \
--val-data="./data/mimic3/full_val_data.pickle" \
--dataset-type mimic \
--warmup 1500 \
--batch-size=128 \
--lr=1e-3 \
--wd=0.1 \
--epochs=3 \
--workers=1 \
--model <model>
```

### Launch tensorboard:
```bash
tensorboard --logdir=logs/tensorboard/ --port=7777
```

## Citation

I recommend the following citation:

```bib
@misc{plasser2022cnep,
  title={Multimodal Contrastive Pre-Training on a Medical Benchmark Dataset (MIMIC-III)},
  author={Plasser, Jürgen Richard, and Roland, Theresa and Hochreiter, Sepp},
  journal={tbd},
  year={2022},
  eprint={tbd}
}
```
