import os
import time
import json
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast
import torch.distributed as dist

from .zero_shot import zero_shot_eval
from torchvision.utils import save_image

import sys
import pdb
import wandb

import logging

def is_master(args):
    return (not args.distributed) or args.gpu == 0

def get_loss(model, images, texts, embeds, loss_img, loss_txt, epoch, idx, args):
    image_features, text_features, logit_scale = model(images, texts, embeds)
    if isinstance(texts, list):
        text_features = text_features.cuda(args.gpu, non_blocking=True)
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    if idx<1:
        # img1 = torch.nn.functional.normalize(logits_per_image)
        index = logits_per_image.argmax(dim=1).detach().cpu().numpy()
        max_ = logits_per_image.max()
        c = logits_per_image / max_
        c[np.arange(len(c)), index] = max_

        img1 = torch.sigmoid(c)
        min_, max_ = img1.min(), img1.max()
        img2 = 1. / (max_ - min_) * img1 + 1. * min_ / (min_ - max_)
        save_image(img2, f'{args.checkpoint_path}/train_logits_per_image_{epoch}_{idx}.png')
        if args.wandb:
            wandb.log({'train/images': wandb.Image(img2, caption=f"train_lpi_{epoch}_{idx}.png")})

    ground_truth = torch.arange(len(logits_per_image)).long()
    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2
    return total_loss


def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    
    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        images, _, texts, embeds = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            if not isinstance(texts, list):
                texts = texts.cuda(args.gpu, non_blocking=True)
            embeds = embeds.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = get_loss(model, images, texts, embeds, loss_img, loss_txt, epoch, i, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss = get_loss(model, images, texts, embeds, loss_img, loss_txt, epoch, i, args)
            total_loss.backward()
            optimizer.step()

        #if not args.skip_scheduler:
        scheduler.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        # lower bound is zero, why?
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            timestep = epoch * num_batches_per_epoch + i
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})


def evaluate(model, data, epoch, args, tb_writer=None, steps=None):
    if not is_master(args):
        return

    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

    if 'mimic3-val' in data:
        dataloader = data['mimic3-val'].dataloader
    else:
        dataloader = data['val'].dataloader

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        ctr = 0
        for batch in dataloader:
            # we do not apply the labels now, so we can omit them from the data loader:
            # images = features, _, texts = notes
            # TODO: refactor
            images, _, texts, embeds = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                if not isinstance(texts, list):
                    texts = texts.cuda(args.gpu, non_blocking=True)
                embeds = embeds.cuda(args.gpu, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts, embeds)
            if isinstance(texts, list):
                text_features = text_features.cuda(args.gpu, non_blocking=True)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            if ctr < 1:
                #img1 = torch.nn.functional.normalize(logits_per_image)
                c = logits_per_image.detach().cpu()
                idx = c.argmax(dim=1).numpy()
                max_ = c.max()
                c = c / max_
                c[np.arange(len(c)), idx] = max_

                img1 = torch.sigmoid(c)
                min_, max_ = img1.min(), img1.max()
                img2 = 1. / (max_ - min_) * img1 + 1. * min_ / (min_ - max_)
                save_image(img2, f'{args.checkpoint_path}/eval_logits_per_image_{epoch}_{ctr}.png')
                if args.wandb:
                    wandb.log({'val/images': wandb.Image(img2, caption=f"eval_lpi_{epoch}_{ctr}.png")})
                ctr += 1

            ground_truth = torch.arange(len(images)).long()
            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

        metrics = get_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale,
            epoch=epoch,
            args=args
        )
        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )
        metrics.update(zero_shot_metrics)

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)
        if args.wandb:
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch})

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics

def get_metrics(image_features, text_features, logit_scale, epoch=0, args=None):
    print(f"Get metrics for: {image_features.shape=}, {text_features.shape=}, {logit_scale=}")

    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    #img1 = torch.nn.functional.normalize(logits_per_image)
    c = logits_per_image.detach().cpu()
    idx = c.argmax(dim=1).numpy()
    max_ = c.max()
    c = c / max_
    c[np.arange(len(c)), idx] = 1000.

    img1 = torch.sigmoid(c)
    min_, max_ = img1.min(), img1.max()
    img2 = 1. / (max_ - min_) * img1 + 1. * min_ / (min_ - max_)
    save_image(img2, f'{args.checkpoint_path}/logits_per_image_{epoch}.png')
    if args.wandb and ((epoch + 1) == args.epochs):
        logging.info("logged final metric image to wandb.")
        wandb.log({'final_image': wandb.Image(img2, caption=f"lpi_{epoch}.png")})

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
