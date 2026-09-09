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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

#from utils.data_loading import BasicDataset, CarvanaDataset
from utils.data_loading_fixed_ids import CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from torch.utils.data import ConcatDataset



nnname="SHDR_random_attack_results_robust_subspace_correcttrainset_pixcons"

basep = "/path/to/workspace/program_output/DeepSteg"
dir_img = Path("/path/to/workspace/program_output/DeepUWF/random_attack_results_robust_subspace_correcttrainset_pixcons")

nae_dir =  Path("/path/to/workspace/program_output/DeepUWF/random_attack_results_robust_subspace_correcttrainset_pixcons")


dir_mask = Path('/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/train') 
transName = False

dir_checkpoint = Path(f'{basep}/{nnname}/unet_hierar')
print("dir_img:{}".format(dir_img))
print("dir_mask:{}".format(dir_mask))
print("nae_dir:{}".format(nae_dir))
print("dir_checkpoint:{}".format(dir_checkpoint))
print("transName:{}".format(transName))
def build_merged_dataset(
    images_root,
    masks_root,
    class_names,
    scale=1.0,
    split="train",
    transName=False,
):
    datasets = []

    images_root = Path(images_root)
    masks_root = Path(masks_root)

    for class_name in class_names:
        images_dir = images_root / class_name
        masks_dir = masks_root / class_name

        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: {images_dir}"
            )
        if not masks_dir.is_dir():
            raise FileNotFoundError(
                f"Mask directory does not exist: {masks_dir}"
            )
        if not nae_dir.is_dir():
            raise FileNotFoundError(
                f"NAE directory does not exist: {nae_dir}"
            )

        class_dataset = CarvanaDataset(
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            nae_dir="",
            scale=scale,
            split=split,
            transName=transName,
        )

             


        datasets.append(class_dataset)

        print(
            f"Loaded class '{class_name}': "
            f"{len(class_dataset)} samples"
        )

    merged_dataset = ConcatDataset(datasets)

    print(
        f"Merged {len(datasets)} datasets: "
        f"{len(merged_dataset)} total samples"
    )

    return merged_dataset

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              nae: bool=False):
    
    

    class_names = ["class_0", "class_1"]

    train_datasets = []
    val_datasets = []

    for class_name in class_names:
        train_datasets.append(
            CarvanaDataset(
                images_dir=Path(dir_img) / class_name,
                masks_dir=Path(dir_mask) / class_name,
                nae_dir=Path(nae_dir) / class_name,
                scale=1,
                split="train",
                transName=transName,
                train_ratio=0.75,
                val_ratio=0.125,
                test_ratio=0.125,
                split_seed=42,

            )
        )


        val_datasets.append(
            CarvanaDataset(
                images_dir=Path(dir_img) / class_name,
                masks_dir=Path(dir_mask) / class_name,
                nae_dir=Path(nae_dir) / class_name,
                scale=1,
                split="val",
                transName=transName,
                train_ratio=0.75,
                val_ratio=0.125,
                test_ratio=0.125,
                split_seed=42,
            )
        )

    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets)

    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))

    n_train = len(train_set) 
    n_val = len(val_set) 
   

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.L
    global_step = 0

    best_val_score = float('inf')
    best_epoch = 0
    best_global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                nae_img = batch['nae']

                # assert images.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                nae_img = nae_img.to(device=device, dtype=torch.float32)
                if nae==False:
                    input_img = images
                else:
                    input_img = torch.cat((images,nae_img),dim=1)
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(input_img)
                    loss = torch.norm(masks_pred-true_masks)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Validation once per epoch
        val_score = evaluate(net, val_loader, device, nae)

        if torch.is_tensor(val_score):
            val_score_float = val_score.item()
        else:
            val_score_float = float(val_score)

        scheduler.step(val_score_float)

        logging.info(
            f'Epoch {epoch}: Validation reconstruction error: {val_score_float:.6f}'
        )

        # save best checkpoint: lower is better
        if val_score_float < best_val_score:
            best_val_score = val_score_float
            best_epoch = epoch
            best_global_step = global_step

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_score': best_val_score,
                    'amp': amp,
                }, str(dir_checkpoint / 'best_checkpoint.pth'))

                logging.info(
                    f'Best checkpoint saved! '
                    f'epoch={best_epoch}, step={best_global_step}, '
                    f'best_val_score={best_val_score:.6f}'
                )
                
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
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
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  nae = args.nae)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
