import argparse
import logging
import os
import sys
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from diceloss import dice_coef_9cat_loss
from classcount import classcount
from utils.logging.wandb_logging import Wandblogger, generate_run_name

torch.autograd.set_detect_anomaly(True)

# Comment/Uncomment to toggle subset for training
dir_img = 'data/img_subset/'
dir_mask = 'data/masks_subset/'

## Comment/Uncomment to toggle subset for training
# dir_img = 'data/training_data/images'
# dir_mask = 'data/training_data/masks'

dir_checkpoint = 'checkpoints/'

tags = ['train/loss','validation/loss']
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)

    ## Uncomment to use an exponential scheduler
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer= optimizer, gamma= 0.96) 

    ## Uncomment the below lines if optimal learning rate technique is to be found as explained in the blog
    # lambda1 = lambda epoch: 1.04 ** epoch
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda= lambda1)

    weights_classes = torch.from_numpy(classcount(train_loader))
    weights_classes = weights_classes.to(device=device, dtype=torch.float32)

    print("Class Distribution", weights_classes)

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss(weight=weights_classes)
    else:
        criterion = nn.BCEWithLogitsLoss()
    run_name = generate_run_name()
    wandb_logger = Wandblogger(name=run_name)


    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        pseudo_batch_loss = 0  ##remove when not pruning for lr
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # Use half precision model for training
                # net.half()

                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # imgs = imgs.to(device=device, dtype=torch.float16)
                imgs = imgs.to(device=device)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long  ## For cross entropy loss
                # mask_type = torch.float32 if net.n_classes == 1 else torch.float ## For Dice Loss
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                # convert the prediction to float32 for avoiding nan in loss calculation
                masks_pred = masks_pred.type(torch.float32)

                ## Cross Entropy Loss
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                ## Dice Loss
                # loss=dice_coef_9cat_loss(true_masks,masks_pred)
                # epoch_loss += loss.item()
                mean_epoch_loss= epoch_loss / n_train
                pbar.set_postfix(**{'Epoch Loss': epoch_loss / n_train})

                # convert model to full precision for optimization of weights
                net.float()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                pseudo_batch_loss += loss.item()

                if (global_step) % 16 == 0:
                    writer.add_scalar('Batch Loss/train', pseudo_batch_loss, global_step)
                    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    # scheduler.step()
                    pseudo_batch_loss = 0

            #end of batch
            wandb_logger.log({"train/batch_loss":loss})
            wandb_logger.end_batch()

        #end of epoch

        tags = ['train/loss', 'validation/loss']

        writer.add_scalar('Loss/train', epoch_loss / n_train, epoch + 1)

        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch + 1)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch + 1)

        val_score = eval_net(net, val_loader, device)

        # if (epoch+1) % 10 == 0:
        # scheduler.step()

        # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)

        if net.n_classes > 1:
            logging.info('Validation CE Loss: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, epoch + 1)
        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))
            writer.add_scalar('Dice/test', val_score, epoch + 1)

        for x, tag in zip(list(mean_epoch_loss) + list(val_score),tags):
            wandb_logger.log({tag: x})
        wandb_logger.end_epoch()


        writer.add_images('images', imgs, epoch + 1)
        if net.n_classes == 1:
            writer.add_images('masks/true', true_masks, epoch + 1)
            writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, epoch + 1)

        if (epoch + 1) % 5 == 0:
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=4e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=7, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
