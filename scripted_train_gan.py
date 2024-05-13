# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
from dataset import CUDAPrefetcher, BaseImageDataset, PairedImageDataset
from imgproc import random_crop_torch, random_rotate_torch, random_vertically_flip_torch, random_horizontally_flip_torch
from test import test
from utils import build_iqa_model, load_resume_state_dict, load_pretrained_state_dict, make_directory, save_checkpoint, \
    Summary, AverageMeter, ProgressMeter


def main():
    # Read parameters from configuration file

    # Fixed random number seed
    random.seed(5203)
    np.random.seed(5203)
    torch.manual_seed(5203)
    torch.cuda.manual_seed_all(5023)

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Default to start training from scratch
    start_epoch = 0

    # Initialize the image clarity evaluation index
    best_psnr = 0.0
    best_ssim = 0.0

    # Define the running device number
    device = torch.device("cuda", 0)

    train_dirs = ["/home/jfeggerd/turbo/density1_splices", "/home/jfeggerd/turbo/density5_splices", "/home/jfeggerd/turbo/density10_splices", "/home/jfeggerd/turbo/density50_splices"]
    names = ["density1", "density5", "density10", "density50"]
    test_gt_dir = "/home/jfeggerd/turbo/hr_test"
    test_lr_dir = "/home/jfeggerd/turbo/lr_test"

    epochs = [5, 10, 20, 30, 50]

    for i in range(len(train_dirs)):

        print("Training with data from " + names[i])

        train_data_prefetcher, paired_test_data_prefetcher = load_dataset_man(train_dirs[i], test_gt_dir, test_lr_dir, 4, 16, 12, device)
        g_model, ema_g_model, d_model = build_model_man(1, 1, 64, 16, device)
        pixel_criterion, adversarial_criterion = define_loss_man(device)
        g_optimizer, d_optimizer = define_optimizer_man(g_model, d_model)
        g_scheduler, d_scheduler = define_scheduler_man(g_optimizer, d_optimizer)

        g_model = load_pretrained_state_dict(g_model,
                                            True,
                                            "./results/pretrained_models/SRGAN_x4-SRGAN_ImageNet.pth.tar.1")
        print(f"Loaded pretrained model weights successfully.")

        d_model = load_pretrained_state_dict(d_model,
                                            False,
                                            "./results/pretrained_models/DiscriminatorForVGG_x4-SRGAN_ImageNet.pth.tar")
        print(f"Loaded pretrained model weights successfully.")

        psnr_model, ssim_model = build_iqa_model(4, True, device)

        # Create the folder where the model weights are saved
        samples_dir = os.path.join("samples", names[i])
        results_dir = os.path.join("results", names[i])
        make_directory(samples_dir)
        make_directory(results_dir)

        writer = SummaryWriter(os.path.join("samples", "logs", names[i]))

        #Train the model from each dataset for 50 epochs
        for epoch in range(start_epoch, 50):
            train(g_model,
                ema_g_model,
                d_model,
                train_data_prefetcher,
                pixel_criterion,
                None,
                adversarial_criterion,
                g_optimizer,
                d_optimizer,
                epoch,
                scaler,
                writer,device)

            # Update LR
            g_scheduler.step()
            d_scheduler.step()

            print("Calling test now!!")

            psnr, ssim = test(g_model,
                            paired_test_data_prefetcher,
                            psnr_model,
                            ssim_model,
                            device, None)
            print("\n")

            # Write the evaluation indicators of each round of Epoch to the log
            writer.add_scalar(f"Test/PSNR", psnr, epoch + 1)
            writer.add_scalar(f"Test/SSIM", ssim, epoch + 1)


            #I don't need to save the checkpoint for every epoch, let's do every 5
            if(epoch % 5 == 0):
                # Automatically save model weights
                is_best = psnr > best_psnr and ssim > best_ssim
                is_last = (epoch + 1) == 50
                best_psnr = max(psnr, best_psnr)
                best_ssim = max(ssim, best_ssim)
                save_checkpoint({"epoch": epoch + 1,
                                "psnr": psnr,
                                "ssim": ssim,
                                "state_dict": g_model.state_dict(),
                                "ema_state_dict": ema_g_model.state_dict() if ema_g_model is not None else None,
                                "optimizer": g_optimizer.state_dict()},
                                f"epoch_{epoch + 1}.pth.tar",
                                samples_dir,
                                results_dir,
                                "g_best.pth.tar",
                                "g_last.pth.tar",
                                is_best,
                                is_last)
                save_checkpoint({"epoch": epoch + 1,
                                "psnr": psnr,
                                "ssim": ssim,
                                "state_dict": d_model.state_dict(),
                                "optimizer": d_optimizer.state_dict()},
                                f"epoch_{epoch + 1}.pth.tar",
                                samples_dir,
                                results_dir,
                                "d_best.pth.tar",
                                "d_last.pth.tar",
                                is_best,
                                is_last)


def load_dataset_man(train_gt_dir, test_gt_dir, test_lr_dir, scale_factor, batch_size, num_workers, device) -> [CUDAPrefetcher, CUDAPrefetcher]:
    degenerated_train_datasets = BaseImageDataset(
        train_gt_dir,
        None,
        scale_factor,
    )

    # Load the registration test dataset
    paired_test_datasets = PairedImageDataset(test_gt_dir, test_lr_dir)

    # generate dataset iterator
    degenerated_train_dataloader = DataLoader(degenerated_train_datasets,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              drop_last=True,
                                              persistent_workers=True)
    paired_test_dataloader = DataLoader(paired_test_datasets,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=True,
                                        drop_last=True,
                                        persistent_workers=True)

    # Replace the data set iterator with CUDA to speed up
    train_data_prefetcher = CUDAPrefetcher(degenerated_train_dataloader, device)
    paired_test_data_prefetcher = CUDAPrefetcher(paired_test_dataloader, device)

    return train_data_prefetcher, paired_test_data_prefetcher

def build_model_man(in_channels, out_channels, channels, num_rcb, device) -> [nn.Module, nn.Module or Any, nn.Module]:
    g_model = model.__dict__["srresnet_x4"](in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           channels=channels,
                                                           num_rcb=num_rcb)
    d_model = model.__dict__["discriminator_for_vgg"](in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           channels=channels)

    g_model = g_model.to(device)
    d_model = d_model.to(device)

    # Generate an exponential average model based on a generator to stabilize model training
    ema_decay = 0.999
    ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
        (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
    ema_g_model = AveragedModel(g_model, device=device, avg_fn=ema_avg_fn)


    g_model = torch.compile(g_model)

    ema_g_model = torch.compile(ema_g_model)

    return g_model, ema_g_model, d_model


def define_loss_man(device) -> [nn.MSELoss, nn.BCEWithLogitsLoss]:
    pixel_criterion = nn.MSELoss()

    adversarial_criterion = nn.BCEWithLogitsLoss()

    pixel_criterion = pixel_criterion.to(device)
    adversarial_criterion = adversarial_criterion.to(device)

    return pixel_criterion, adversarial_criterion


def define_optimizer_man(g_model: nn.Module, d_model: nn.Module) -> [optim.Adam, optim.Adam]:
    g_optimizer = optim.Adam(g_model.parameters(),
                                0.0001,
                                [ 0.9, 0.999 ],
                                0.0001,
                                0.0)
    d_optimizer = optim.Adam(d_model.parameters(),
                                0.0001,
                                [ 0.9, 0.999 ],
                                0.0001,
                                0.0)

    return g_optimizer, d_optimizer


def define_scheduler_man(g_optimizer: optim.Adam, d_optimizer: optim.Adam) -> [lr_scheduler.MultiStepLR, lr_scheduler.MultiStepLR]:
    g_scheduler = lr_scheduler.MultiStepLR(g_optimizer,
                                            [9],
                                            0.5)
    d_scheduler = lr_scheduler.MultiStepLR(d_optimizer,
                                            [9],
                                            0.5)
    
    return g_scheduler, d_scheduler


def train(
        g_model: nn.Module,
        ema_g_model: nn.Module,
        d_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.L1Loss,
        content_criterion: model.ContentLoss,
        adversarial_criterion: nn.BCEWithLogitsLoss,
        g_optimizer: optim.Adam,
        d_optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device
) -> None:
    # Calculate how many batches of data there are under a dataset iterator
    batches = len(train_data_prefetcher)

    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    g_losses = AverageMeter("G Loss", ":6.6f", Summary.NONE)
    d_losses = AverageMeter("D Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, g_losses, d_losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Set the model to training mode
    g_model.train()
    d_model.train()

    # Define loss function weights
    pixel_weight = torch.Tensor([1]).to(device)
    feature_weight = torch.Tensor([0]).to(device)
    adversarial_weight = torch.Tensor([0.001]).to(device)

    # Initialize data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    train_data_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = train_data_prefetcher.next()

    # Used for discriminator binary classification output, the input sample comes from the data set (real sample) is marked as 1, and the input sample comes from the generator (generated sample) is marked as 0
    batch_size = batch_data["gt"].shape[0]

    real_label = torch.full([batch_size, 1], 1.0, dtype=torch.float, device=device)
    fake_label = torch.full([batch_size, 1], 0.0, dtype=torch.float, device=device)

    while batch_data is not None:
        # Load batches of data
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        # image data augmentation
        # gt, lr = random_crop_torch(gt,
        #                            lr,
        #                            config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"],
        #                            config["SCALE"])
        # gt, lr = random_rotate_torch(gt, lr, config["SCALE"], [0, 90, 180, 270])
        # gt, lr = random_vertically_flip_torch(gt, lr)
        # gt, lr = random_horizontally_flip_torch(gt, lr)

        # Record the time to load a batch of data
        data_time.update(time.time() - end)

        # start training the generator model
        # Disable discriminator backpropagation during generator training
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize the generator model gradient
        g_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and confrontation loss
        with amp.autocast():
            sr = g_model(lr)

            pixel_loss = pixel_criterion(sr, gt)
            # feature_loss = content_criterion(sr, gt)
            adversarial_loss = adversarial_criterion(d_model(sr), real_label)
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_loss))
            # feature_loss = torch.sum(torch.mul(feature_weight, feature_loss))
            adversarial_loss = torch.sum(torch.mul(adversarial_weight, adversarial_loss))
            # Compute generator total loss
            g_loss = pixel_loss + adversarial_loss # + feature_loss
        # Backpropagation generator loss on generated samples
        scaler.scale(g_loss).backward()
        # update generator model weights
        scaler.step(g_optimizer)
        scaler.update()
        # end training generator model

        # start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradient
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model on real samples
        with amp.autocast():
            gt_output = d_model(gt)
            d_loss_gt = adversarial_criterion(gt_output, real_label)

        # backpropagate discriminator's loss on real samples
        scaler.scale(d_loss_gt).backward()

        # Calculate the classification score of the generated samples by the discriminator model
        with amp.autocast():
            sr_output = d_model(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
        # backpropagate discriminator loss on generated samples
        scaler.scale(d_loss_sr).backward()

        # Compute the discriminator total loss value
        d_loss = d_loss_gt + d_loss_sr
        # Update discriminator model weights
        scaler.step(d_optimizer)
        scaler.update()
        # end training discriminator model

        if True:
            # update exponentially averaged model weights
            ema_g_model.update_parameters(g_model)

        # record the loss value
        d_losses.update(d_loss.item(), batch_size)
        g_losses.update(g_loss.item(), batch_size)

        # Record the total time of training a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output training log information once
        if batch_index % 100 == 0:
            # write training log
            iters = batch_index + epoch * batches
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Loss", d_loss_gt.item(), iters)
            writer.add_scalar("Train/D(SR)_Loss", d_loss_sr.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Feature_Loss", 0, iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", torch.sigmoid_(torch.mean(gt_output.detach())).item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", torch.sigmoid_(torch.mean(sr_output.detach())).item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


if __name__ == "__main__":
    main()
