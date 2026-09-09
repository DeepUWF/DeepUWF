import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torchvision.utils import save_image

def evaluate(net, dataloader, device,use_nae=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    nae_img = None
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        if len(batch) == 3:
            image, mask_true,nae_img = batch['image'], batch['mask'],batch['nae']
        else:
            image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        if nae_img is not None:
            nae_img = nae_img.to(device=device, dtype=torch.float32)
        #mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        if use_nae==True:
            input_img = torch.cat((image,nae_img),dim=1)
        else:
            input_img = image
        with torch.no_grad():
            # predict the mask
            mask_pred = net(input_img)
            #save_image(mask_pred,'/path/to/workspace/DeepSteg/data/val_img/1.png',normalize=True)
            dice_score += torch.norm(mask_pred-mask_true)
            # convert to one-hot format
            # if net.n_classes == 1:
            #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            #     # compute the Dice score
            #     dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            # else:
            #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #     # compute the Dice score, ignoring background
            #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score/num_val_batches
