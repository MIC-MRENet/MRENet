import argparse
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import SimpleITK as sitk

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
# from hist_match import histogram_matching
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = '/data/zk/1/data/imgs/TMJ_imgs/'
dir_mask = '/data/zk/1/data/masks/TMJ_masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=300,
              batch_size=1,
              lr=0.1,
              val_percent=0.5,
              save_cp=True,
              img_scale=0.366):

    #histogram_matching()
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train, val = random_split(dataset, [n_train, n_val])
    n_val = 1
    n_train = 27
    #train = dataset
    val = list(dataset)[0:1]
    train = list(dataset)[0:27]
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

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

    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    optimizer = torch.optim.SGD(net.parameters(),lr = args.lr,momentum = 0.9)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                # logging.info(f'img.size{imgs.shape}')
                # logging.info(f'mask.size{true_masks.shape}')
                #true_mask_pro = true_masks.squeeze().numpy().astype(np.uint8)
                #img_show = sitk.GetImageFromArray(true_mask_pro)
                #sitk.WriteImage(img_show, './data/pred/scale{}_mask.nii'.format(img_scale))
                '''
                imgs_pnp = imgs.squeeze().cpu().detach().numpy()
                imgs_show = sitk.GetImageFromArray(imgs_pnp)
                sitk.WriteImage(imgs_show, f'./data/debug/epoch300_41Img/input_round1/{epoch+1}_{global_step+1}.nii.gz')
                img_pro = (imgs*32768).squeeze().numpy().astype(np.int16)
                true_masks_pnp = true_masks.squeeze().cpu().detach().numpy()
                true_masks_show = sitk.GetImageFromArray(true_masks_pnp)
                sitk.WriteImage(true_masks_show, f'./data/debug/epoch300_41Img/gt_round1/{epoch+1}_{global_step+1}.nii.gz')
                img_show = sitk.GetImageFromArray(img_pro)
                #sitk.WriteImage(img_show, './data/pred/scale{}_input.nii'.format(img_scale))
                '''
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
 

                masks_pred = net(imgs)
                # if (epoch+1) % 100 == 0:
                #     masks_pnp = masks_pred.squeeze().cpu().detach().numpy().astype(np.int8)
                #     masks_show = sitk.GetImageFromArray(masks_pnp)
                #     sitk.WriteImage(masks_show, f'./data/debug/epoch500_20Img_UNetV2/model_out/{epoch+1}_{global_step+1}.nii')
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (batch_size)) == 0:
                #if global_step % (1) == 0:
                    val_score = eval_net(net, val_loader, device, n_val)
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)

                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                #writer.add_images('images', imgs, global_step)
                #if net.n_classes == 1:
                    #writer.add_images('masks/true', true_masks, global_step)
                    #writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            if (epoch+1) % 10 == 0:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'72Img_80160_5/' +f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.02,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.333,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.2 ** (epoch // 30))
    # lr = 0.05 * (0.5 ** (epoch // 200))
    for param_group in optimizer.param_groups:  
        param_group['lr'] = lr

if __name__ == '__main__':

    device = torch.device('cuda:0')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # device = torch.device('cuda')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Trilinear" if net.trilinear else "Dilated conv"} upscaling')

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

