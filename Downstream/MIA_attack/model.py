import torch
import sys
sys.path.append('..')

from Model import models_vit



def load_model_from_checkpoint(checkpoint_path, num_classes, img_size=224, device='cuda', model_name='vit_large_patch16'):
    model = models_vit.__dict__[model_name](
        img_size=img_size,
        num_classes=num_classes,
        drop_path_rate=0.1,
        global_pool=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint_model = checkpoint['model']
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_model = checkpoint['model_state_dict']
    else:
        print("wrong checkpoint in prepare.py")


    model.load_state_dict(checkpoint_model)
    model = model.to(device)
    return model

