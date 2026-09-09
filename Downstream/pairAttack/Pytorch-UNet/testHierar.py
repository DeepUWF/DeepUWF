import os 
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
#import wandb
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

from utils.data_loading_fixed_ids import CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from torchvision.utils import save_image,make_grid

from skimage.metrics import peak_signal_noise_ratio, structural_similarity,mean_squared_error
import numpy as np
from trainHierar import build_merged_dataset

nnname="SHDR_random_attack_results_robust_subspace_correcttrainset_pixcons"
basep = "/path/to/workspace/program_output/DeepSteg"


test_save_dir = Path(f'{basep}/{nnname}/test_unet_reocver')
nae_dir =  Path("/path/to/workspace/program_output/DeepUWF/random_attack_results_robust_subspace_correcttrainset_pixcons")


dir_img = Path("/path/to/workspace/program_output/DeepUWF/random_attack_results_robust_subspace_correcttrainset_pixcons")

transName = False

dir_mask = Path('/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/train') 


dir_checkpoint = Path(f'{basep}/{nnname}/unet_hierar')

print("dir_img:{}".format(dir_img))
print("dir_mask:{}".format(dir_mask))
print("nae_dir:{}".format(nae_dir))
print("dir_checkpoint:{}".format(dir_checkpoint))
print(f"test_save_dir:{test_save_dir}")
def convert1(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return img

def val_net(net,
              device,          
              batch_size: int = 1,
              val_percent: float = 0.1,
              img_scale: float = 0.5,
              nae: bool=False,
              useotherattack:bool=False):

    class_names = ["class_0", "class_1"]

    test_datasets = []

    for class_name in class_names:
        test_datasets.append(
            CarvanaDataset(
                images_dir=Path(dir_img) / class_name,
                masks_dir=Path(dir_mask) / class_name,
                nae_dir=Path(nae_dir) / class_name,
                scale=1,
                split="test",
                transName=transName,
                train_ratio=0.75,
                val_ratio=0.125,
                test_ratio=0.125,
                split_seed=42,

            )
        )



    test_set = ConcatDataset(test_datasets)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)


    net.eval()
    idx = 0

    
    if not os.path.exists(test_save_dir):
        os.mkdir(test_save_dir)

    psnr,ssim = [],[]
    for batch in test_loader:
        if idx>500:
            break
        images = batch['image']
        true_masks = batch['mask']
        nae_img = batch['nae']

        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        nae_img = nae_img.to(device=device, dtype=torch.float32)

        if useotherattack==True:
            #input_img = RICNet()
            input_img = images
        else:
            input_img = images
        if nae==True:
            input_img = torch.cat((input_img,nae_img),dim=1)

        masks_pred = net(input_img)
        #save_image(masks_pred,'/path/to/workspace/DeepSteg/data/val_img/1.png',normalize=True)
        shownum = 16
        # toshow = make_grid(torch.cat((images[:shownum],true_masks[:shownum],\
        #     masks_pred[:shownum]),dim=0),nrow=shownum)

        true_masks_norm = convert1(true_masks*0.5+0.5)
        masks_pred_norm = convert1(masks_pred*0.5+0.5)

        for ij in range(0,len(images)):
           
            save_image(images[ij].unsqueeze(0)*0.5+0.5,\
                '{}/{}_miximg.png'.format(test_save_dir,ij+idx*32),normalize=False)
            save_image(true_masks[ij].unsqueeze(0)*0.5+0.5,\
                '{}/{}_oriimg.png'.format(test_save_dir,ij+idx*32),normalize=False)
            save_image(masks_pred[ij].unsqueeze(0)*0.5+0.5,\
                '{}/{}_recovered.png'.format(test_save_dir,ij+idx*32),normalize=False)
            psnr.append(peak_signal_noise_ratio(true_masks_norm[ij],masks_pred_norm[ij],data_range=255))
            #ssim.append(structural_similarity(true_masks_norm[ij],masks_pred_norm[ij],win_size=11, data_range=255.0, multichannel=True))
            try:
                ssim.append(
                    structural_similarity(
                       true_masks_norm[ij],masks_pred_norm[ij],
                        win_size=11,
                        data_range=255.0,
                        channel_axis=-1
                    )
                )
            except TypeError:
                ssim.append(
                    structural_similarity(
                        true_masks_norm[ij],masks_pred_norm[ij],
                        win_size=11,
                        data_range=255.0,
                        multichannel=True
                    )
                )
            

        idx += 1
    print(f"avg psnr:{np.mean(np.asarray(psnr))}")
    print(f"avg ssim:{np.mean(np.asarray(ssim))}")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--nae', action='store_true', default=False, help='input nae')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    args.nae=False
    print("args.nae:{}".format(args.nae))
    if args.nae ==True or args.nae =="True":
        n_channels = 6
    else:
        n_channels = 3
    net = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device)["model_state_dict"])
        logging.info(f'Model loaded from {args.load}')


    net.to(device=device)
    try:
        val_net(net,
              device,          
              batch_size=args.batch_size,
              val_percent=args.val / 100,
              img_scale=args.scale,
              nae= args.nae)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
