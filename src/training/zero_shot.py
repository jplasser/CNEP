from collections import Counter

from tqdm import tqdm
import torch
# import clip.clip as clip
from sklearn.metrics import roc_auc_score, RocCurveDisplay, accuracy_score,\
    f1_score, confusion_matrix, classification_report, auc, precision_recall_curve
import matplotlib.pyplot as plt
# from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template

import logging
import numpy as np


def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print ("confusion matrix:")
        print (cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = precision_recall_curve(y_true, predictions[:, 1])
    auprc = auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print ("accuracy =", acc)
        print ("precision class 0 =", prec0)
        print ("precision class 1 =", prec1)
        print ("recall class 0 =", rec0)
        print ("recall class 1 =", rec1)
        print ("AUC of ROC =", auroc)
        print ("AUC of PRC =", auprc)

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse}

# def zero_shot_classifier(model, classnames, templates, args):
def zero_shot_classifier(model, classnames, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            # texts = [template(classname) for template in templates] #format with class
            # texts = clip.tokenize(texts).to(args.gpu) #tokenize
            texts = classname
            texts = texts.to(args.gpu)
            if args.distributed:
                class_embeddings = model.module.encode_text(texts)
            elif args.dp:
                class_embeddings = model(None, texts)
            else:
                class_embeddings = model.encode_text(texts)
            # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            ## class_embedding = class_embeddings.mean(dim=0)
            ## class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embeddings)
        # FIXME: hardcoded dim
        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).reshape(-1, 1024).T.to(args.gpu)
    return zeroshot_weights


def accuracy(output, target, labels, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    # pred = output.topk(1, 1, True, True)[1].t()
    save_shape = pred.shape
    pred = labels[pred.reshape(-1)]
    pred = pred.reshape(save_shape)
    # pred = pred.count_nonzero(axis=0) > 0
    # pred = pred.type(torch.uint8)
    pred = pred.reshape(1, -1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # roc_auc = roc_auc_score(target.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy())
    # print("ROC-AUC score = ", roc_auc)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk], \
           pred.view(-1).cpu().numpy(), \
           target.view(-1).cpu().numpy()


def run(model, classifier, dataloader, labels, args):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        preds_ = []
        targets_ = []
        for images, target, _, embeds in tqdm(dataloader):
            images = images.to(args.gpu)
            target = target.to(args.gpu)
            # texts = texts.to(args.gpu)
            embeds = embeds.to(args.gpu)

            # predict
            if args.distributed:
                image_features = model.module.encode_image(images, embeds)
            elif args.dp:
                image_features = model(images, None, embeds)
            else:
                image_features = model.encode_image(images, embeds)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ classifier

            # measure accuracy
            # acc1, acc5 = accuracy(logits, target, labels, topk=(1, 5))
            acc1, predictions, targets = accuracy(logits, target, labels, topk=(1,))
            preds_.extend(predictions)
            targets_.extend(targets)
            top1 += acc1[0]
            # top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    # top5 = (top5 / n)
    return top1, preds_, targets_  # , top5


def zero_shot_eval(model, data, epoch, args):
    # if 'imagenet-val' not in data and 'imagenet-v2' not in data and \
    if 'mimic3-val' not in data:
        return {}

    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot MIMIC3.')

    logging.info('Building zero-shot classifier')

    # classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)
    # get embeddings from train data set
    train_dataloader = data['train'].dataloader
    # embeddings = []
    # for _, _, texts, _ in train_dataloader:
    #     embeddings.extend(texts)
    embeddings = [texts for _, _, texts, _ in train_dataloader]
    labels = [label.reshape(-1) for _, label, _, _ in train_dataloader]
    # labels = torch.stack(labels, dim=1).reshape(-1).to(args.gpu)
    labels = torch.cat(labels).reshape(-1).to(args.gpu)
    classifier = zero_shot_classifier(model, embeddings, args)

    logging.info(f'Using classifier of shape {classifier.shape}')
    results = {}

    if 'mimic3-val' in data:
        # top1, top5 = run(model, classifier, data['mimic3-val'].dataloader, labels, args)
        top1, preds, targets = run(model, classifier, data['mimic3-val'].dataloader, labels, args)
        results['mimic3-zeroshot-val-top1'] = top1

        print(classification_report(targets, preds))

        print_metrics_binary(targets, preds)

        print(f"Accuracy is {top1}.")
        acc_score = accuracy_score(targets, preds)
        print("Acc score = ", acc_score)
        fone_score = f1_score(targets, preds)
        print("F1 score = ", fone_score)
        cfm = confusion_matrix(targets, preds)
        print("confusion matrix = \n", cfm)
        roc_auc = roc_auc_score(targets, preds)
        print("ROC-AUC score = ", roc_auc)
        print(Counter(targets))
        RocCurveDisplay.from_predictions(targets, preds)
        plt.show()
        # results['mimic3-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot MIMIC3.')

    return results
