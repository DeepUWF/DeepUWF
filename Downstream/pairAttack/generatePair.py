import torch,os,random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch
from collections import OrderedDict

import os
import sys



from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


import numpy as np
from torchvision.utils import make_grid,save_image
import cw_attack
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from torch.utils.data import Subset


import os
import pickle
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, label, path

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
mean_imgnet_list = [0.485, 0.456, 0.406]
std_imgnet_list = [0.229, 0.224, 0.225]
mean_imgnet = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
std_imgnet = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

mean_list = [0.5, 0.5, 0.5]
std_list = [0.5, 0.5, 0.5]
_mean_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)
_std_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)

clist = [i for i in range(1, 21)]
klist = [i for i in range(1, 21)]
stlist = [i for i in range(20, 100)]
#stlist = [i for i in range(80, 350)]  # 不要命名为 st，避免和函数内部 steps 变量冲突

beta = 10




def generate_shuffled_manifest(
    args,
    output_txt,
    seed=42,
    max_samples=None,
    overwrite=False,
):
    """
    Generate a reproducible shuffled image manifest.

    Expected dataset:
        torchvision.datasets.ImageFolder or another dataset with
        a `samples` attribute containing:
            [(image_path, label), ...]

    Output format:
        shuffled_index<TAB>dataset_index<TAB>stem<TAB>label<TAB>path

    Args:
        dataset:
            Dataset with `dataset.samples`.
        output_txt:
            Path of the output txt file.
        seed:
            Random seed used to shuffle the samples.
        max_samples:
            Maximum number of shuffled samples to save.
            If None, save all samples.
        overwrite:
            Whether to overwrite an existing manifest.
    """
    data_dir = f"{args.datasetBase}/train"

    dataset = ImageFolderWithPath(
        root=data_dir,
        transform=get_transform(size=448, useimgnet=True)
    )
    print(f"threatModelDeepUWF:Loading data from: {data_dir}")


    if not hasattr(dataset, "samples"):
        raise AttributeError(
            "The dataset must have a `samples` attribute, "
            "such as torchvision.datasets.ImageFolder."
        )



    output_txt = Path(output_txt)

    if output_txt.exists() and not overwrite:
        print(f"Manifest already exists, skipped: {output_txt}")
        return

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    dataset_indices = list(range(len(dataset.samples)))

    rng = random.Random(seed)
    rng.shuffle(dataset_indices)

    if max_samples is not None:
        dataset_indices = dataset_indices[:max_samples]

    with output_txt.open("w", encoding="utf-8") as f:
        f.write(
            "shuffled_index\tdataset_index\tstem\tlabel\tpath\n"
        )

        for shuffled_index, dataset_index in enumerate(dataset_indices):
            image_path, label = dataset.samples[dataset_index]
            image_path = Path(image_path)
            stem = image_path.stem

            f.write(
                f"{shuffled_index}\t"
                f"{dataset_index}\t"
                f"{stem}\t"
                f"{int(label)}\t"
                f"{image_path}\n"
            )

    print(f"Shuffled manifest saved to: {output_txt}")
    print(f"Number of saved samples: {len(dataset_indices)}")
    print(f"Shuffle seed: {seed}")




def build_loader_from_manifest(
    dataset,
    manifest_txt,
    batch_size=16,
    num_samples=None,
    num_workers=4,
    pin_memory=True,
):
    """
    Build a DataLoader using the fixed shuffled order stored in a manifest.
    """
    dataset_indices = []

    with open(manifest_txt, "r", encoding="utf-8") as f:
        header = next(f)

        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 5:
                raise ValueError(
                    f"Invalid manifest line: {line.rstrip()}"
                )

            dataset_index = int(parts[1])
            dataset_indices.append(dataset_index)

            if (
                num_samples is not None
                and len(dataset_indices) >= num_samples
            ):
                break

    subset = Subset(dataset, dataset_indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,  # 顺序已经在 manifest 中固定
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Loaded {len(dataset_indices)} samples from: {manifest_txt}")
    return loader
    

def compute_lpips(img1, img2, lpips_model):
    """
    img1, img2: torch.Tensor, shape [C, H, W], value range [0, 1]
    return: float
    """
    with torch.no_grad():
        img1 = img1.detach().to(next(lpips_model.parameters()).device).float()
        img2 = img2.detach().to(next(lpips_model.parameters()).device).float()

        img1 = torch.clamp(img1, 0, 1).unsqueeze(0)
        img2 = torch.clamp(img2, 0, 1).unsqueeze(0)

        # lpips 库默认输入范围是 [-1, 1]
        img1 = img1 * 2.0 - 1.0
        img2 = img2 * 2.0 - 1.0

        dist = lpips_model(img1, img2)

    return float(dist.item())

def get_transform(size,useimgnet=True):
    t = []
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(size))
    t.append(transforms.ToTensor())
    if useimgnet:
        t.append(transforms.Normalize(mean_imgnet_list,std_imgnet_list))
    else:
        t.append(transforms.Normalize(mean_list,std_list))
    transform = transforms.Compose(t)
    return transform




def load_checkpoint(filepath, net, optimizer=None, device=None, strict=True,scheduler=None):
    if not os.path.isfile(filepath):
        print("=> no checkpoint found at '{}'".format(filepath))
        return 0

    print("=> loading checkpoint '{}'".format(filepath))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)

    # 兼容不同 checkpoint 格式
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        start_epoch = checkpoint.get("epoch", 0)
    else:
        state_dict = checkpoint
        start_epoch = 0

    # 当前模型是不是 DataParallel
    model_keys = list(net.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())

    model_has_module = len(model_keys) > 0 and model_keys[0].startswith("module.")
    ckpt_has_module = len(checkpoint_keys) > 0 and checkpoint_keys[0].startswith("module.")

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_k = k

        # checkpoint 有 module.，当前模型没有 module.：去掉 module.
        if ckpt_has_module and not model_has_module:
            if k.startswith("module."):
                new_k = k[len("module."):]

        # checkpoint 没有 module.，当前模型有 module.：加上 module.
        elif not ckpt_has_module and model_has_module:
            new_k = "module." + k

        new_state_dict[new_k] = v

    # 加载模型参数
    msg = net.load_state_dict(new_state_dict, strict=strict)
    print("=> model loading message:", msg)

    # 加载 optimizer
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> optimizer loaded")
        except Exception as e:
            print("=> optimizer loading failed:", e)

    if scheduler is not None and isinstance(checkpoint, dict) and "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("=> scheduler loaded")
        except Exception as e:
            print("=> scheduler loading failed:", e)


            

    print("=> loaded checkpoint (epoch {})".format(start_epoch))

    return start_epoch



def _make_python_rng(seed=None):
    """
    用于在 changepara=True 时抽取 c/kappa/steps。

    如果 seed 不为 None，则参数随机化是可复现的：
    同一个 seed 会得到同一组 c/kappa/steps。
    """
    if seed is None:
        return random

    if isinstance(seed, torch.Tensor):
        seed = int(seed.detach().cpu().reshape(-1)[0].item())
    else:
        seed = int(seed)

    return random.Random(seed + 202401)

def getAdvZChaPara(
    img_size,
    labels,
    batch_size,
    train=False,
    seed=None,
    type='gaussian',
    changepara=False,
    changegauSigma=False,
    gau_sigmalist = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    target_model = None
):
    if not changepara:
        c = 10
        k = 10

        if train:
            steps = 80
        else:
            steps = 80

        if type != 'gaussian':
            #steps = 300
            steps = 20

    else:
        rng = _make_python_rng(seed)

        c = rng.choice(clist)
        k = rng.choice(klist)
        steps = rng.choice(stlist)


    # init random noise
    if type == 'gaussian':

        if changegauSigma:


            rng = _make_python_rng(seed)

            gau_sigma = rng.choice(gau_sigmalist)

            z0 = torch.randn(

                (batch_size, 3, img_size, img_size),

                device='cuda'

            ) * gau_sigma

        else:

            gau_sigma = 1.0

            z0 = torch.randn(

                (batch_size, 3, img_size, img_size),

                device='cuda'

            )

        z0 = (z0 - torch.min(z0)) / (torch.max(z0) - torch.min(z0) + 1e-12)

        cwatt = cw_attack.CW(
            target_model,
            c=c,
            kappa=k,
            steps=steps,
            targeted=True,
            target_labels=labels,
            seed=seed,
            noise_type='gaussian'
        )

    else:
        z0 = torch.rand((batch_size, 3, img_size, img_size)).cuda()

        cwatt = cw_attack.CW(
            target_model,
            c=c,
            kappa=k,
            steps=steps,
            targeted=True,
            target_labels=labels,
            seed=seed,
            noise_type='uniform'
        )

    adv = cwatt(z0, labels)
    outputs = target_model(adv)
    _, pre = torch.max(outputs, 1)

    succ = len(torch.where(pre == labels)[0])

    return adv, succ


def convert1(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return img



class PickleLabeledImageDataset(Dataset):
    """
    Dataset for pickle label file.

    Expected pickle format:
        [
            ["STDR052-20160831@161220-R4-S", 1],
            ["STDR052-20160831@161220-L1-S", 1],
            ...
        ]

    The first element is image filename stem.
    The second element is label.
    """

    def __init__(
        self,
        pkl_path,
        image_root,
        transform=None,
        extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        recursive=True,
    ):
        self.pkl_path = Path(pkl_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.extensions = extensions

        with open(self.pkl_path, "rb") as f:
            raw_items = pickle.load(f)

        self.items = []
        for item in raw_items:
            img_id, label = item[0], item[1]
            self.items.append((str(img_id), int(label)))

        # Build filename-stem -> full path index
        self.path_map = self._build_path_map(recursive=recursive)

        # Convert img_id to actual image path
        self.samples = []
        missing = []

        for img_id, label in self.items:
            if img_id in self.path_map:
                self.samples.append((self.path_map[img_id], label, img_id))
            else:
                missing.append(img_id)

        if len(missing) > 0:
            print(f"[Warning] Missing {len(missing)} images.")
            print("First 10 missing image ids:")
            for x in missing[:10]:
                print("  ", x)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid image found. Please check image_root={self.image_root}"
            )

        print(f"Loaded {len(self.samples)} samples from {self.pkl_path}")
        print(f"Image root: {self.image_root}")
        print(f"Label counts: {self._label_counts()}")

    def _build_path_map(self, recursive=True):
        path_map = {}

        if recursive:
            all_files = []
            for ext in self.extensions:
                all_files.extend(self.image_root.rglob(f"*{ext}"))
                all_files.extend(self.image_root.rglob(f"*{ext.upper()}"))
        else:
            all_files = []
            for ext in self.extensions:
                all_files.extend(self.image_root.glob(f"*{ext}"))
                all_files.extend(self.image_root.glob(f"*{ext.upper()}"))

        for path in all_files:
            stem = path.stem

            # If duplicate stems exist, warn but keep the first one.
            if stem in path_map:
                print(f"[Warning] Duplicate image stem found: {stem}")
                print(f"  Existing: {path_map[stem]}")
                print(f"  New:      {path}")
                continue

            path_map[stem] = str(path)

        print(f"Indexed {len(path_map)} image files under {self.image_root}")
        return path_map

    def _label_counts(self):
        counts = {}
        for _, label, _ in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, img_id = self.samples[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def create_pickle_labeled_loader(
    pkl_path,
    image_root,
    heresize,
    batch_size=10,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    useimgnet=False,
):
    dataset = PickleLabeledImageDataset(
        pkl_path=pkl_path,
        image_root=image_root,
        transform=get_transform(size=heresize, useimgnet=useimgnet),
        recursive=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataset, loader





def parse_args():
    parser = argparse.ArgumentParser(description="DeepUWF RIC / DP experiment")

    parser.add_argument(
        "--datasetBase",
        type=str,
        #default="/path/to/workspace/dataset/DeepUWF/DR",
        default="/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr",
        help="Base path of the dataset."
    )

    parser.add_argument(
        "--data_mode",
        type=str,
        default="imagefolder",
        help="dr_pickle (for DR ) or imagefolder (for other datasets)"
    )


    parser.add_argument(
        "--modelBase",
        type=str,
        #default="/path/to/hpc/workspace/dataset/DeepUWF/RIC_weights",
        #default="/path/to/workspace/weights/DeepSteg/DR_ric/rdr_ric_kpa_in_loop",
        #default="/path/to/workspace/weights/DeepSteg/DR_ric/dr_ric_miloss", 
        #default="/path/to/workspace/weights/DeepSteg/SHDR_ric/shdr_ric_miloss", 
        default = "/path/to/workspace/weights/DeepSteg/SHDR_ric_midown64/shdr_ric_miloss_centerPlainmodel",
        #default ="/path/to/workspace/weights/DeepSteg/SHDR_ric_kpa/shdr_ric_kpa_in_loop_centerplainmodel",
        #default = "/path/to/workspace/weights/DeepSteg/SHDR_ric_feattrans/shdr_ric_miloss_centerPlainmodel",
        help="Base path of the RIC/model weights."
    )

    parser.add_argument(
        "--savepath",
        type=str,
        #default="/path/to/hpc/workspace/program_output/DeepSteg/pairAttack/SHDR_baseline_bysvhnric",
        #default="/path/to/workspace/program_output/DeepSteg/rdr_ric_kpa_in_loop_resultimgs_ep31",
        #default="/path/to/workspace/program_output/DeepSteg/dr_ric_miloss_resultimgs_ep2",
       # default="/path/to/workspace/program_output/DeepSteg/drgrad_resultimgs",#这里的grad是针对在ffm的基础上finetune好的密文分类器的

        #default="/path/to/workspace/program_output/DeepSteg/shdr_ric_miloss_resultimgs_ep1",
        #default="/path/to/workspace/program_output/DeepSteg/shdr_ric_miloss_resultimgs_ep1_testset",
        #default="/path/to/workspace/program_output/DeepSteg/shdrgrad_of_ffm_resultimg",#这里的grad是针对在ffm的，模拟真实的不使用RIC的场景，直接用明文fintune中心的FFM
        #default="/path/to/workspace/program_output/DeepSteg/shdr_ric_miloss_resultimgs_ep7",
        #default="/path/to/workspace/program_output/DeepSteg/shdrgrad_nodp_ffmfinetune_bestep",
        #default="/path/to/workspace/program_output/DeepSteg/shdrgrad_dp1_ffmfinetune_bestep",
        #default="/path/to/workspace/program_output/DeepSteg/shdrgrad_dp32_ffmfinetune_bestep",
        #default="/path/to/workspace/program_output/DeepSteg/shdrgrad_dp8_ffmfinetune_bestep",
        #default="/path/to/workspace/program_output/DeepSteg/shdr_ric_milossdown64_resultimgs_ep7",
        #default="/path/to/workspace/program_output/DeepSteg/shdr_ric_kpalossdown64_resultimgs_ep7",
        default="/path/to/workspace/program_output/DeepSteg/shdr_ric_milossdown64_resultimgs_ep0",
        #default="/path/to/workspace/program_output/DeepSteg/SHDR_ric_feattrans_resultimgs_ep7",
        help="Path to save outputs."
    )

    parser.add_argument(
        "--changepara",
        type=int,
        default=1,
        help=""
    )
    parser.add_argument(
        "--changegauSigma",
        type=int,
        default=1,
        help=""
    )

    parser.add_argument(
        "--useimgnet",
        type=int,
        default=0,
        help=""
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=""
    )

    parser.add_argument(
        "--ntype",
        type=str,
        default="uniform",
        help="gaussian or uniform"
    )




    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    return args

def setSeed(seed):
    np.random.seed(seed) #[0,2^32-1]
    torch.manual_seed(seed)  #[-9223372036854775808,18446744073709551615] seed (int) – The desired seed. Value must be within the inclusive range [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]. 
    #Otherwise, a RuntimeError is raised. Negative inputs are remapped to positive values with the formula 0xffff_ffff_ffff_ffff + seed.
    if cuda:
        torch.cuda.manual_seed(seed) # 同manual seed
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False       
        torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False    #现实的test阶段需要实现相同的attack的时候设置为false,否则model输出会稍有不同


def getTargetModel(outsideconfig=None):
    from related_code.prepare import prepare_dataset, prepare_model, fix_seed
    from related_code.arguments import get_args_parser 
    args2 = get_args_parser()
    if outsideconfig is not None:
        # for key, value in outsideconfig.items():
        #     setattr(args2, key, value)
        setattr(args2, "input_size", outsideconfig.get("model").get("input_size", 224))
        setattr(args2, "chkpt_path", outsideconfig.get("model").get("chkpt_path", None))
        setattr(args2, "blr", outsideconfig.get("hyperparams").get("blr", None))
        setattr(args2, "min_lr", outsideconfig.get("hyperparams").get("min_lr", None))
        setattr(args2, "warmup_epochs", outsideconfig.get("hyperparams").get("warmup_epochs", None))
        setattr(args2, "batch_size", outsideconfig.get("hyperparams").get("batch_size", None))
        setattr(args2, "label_smoothing", outsideconfig.get("hyperparams").get("label_smoothing", None))
        setattr(args2, "weight_decay", outsideconfig.get("hyperparams").get("weight_decay", None))
        setattr(args2, "pretrain", outsideconfig.get("checkpoint").get("pretrain", None))
        setattr(args2, "finetune", outsideconfig.get("checkpoint").get("finetune", None))
        setattr(args2, "eval", outsideconfig.get("checkpoint").get("eval", None))
        print("outsideconfig loaded")

    print("Target model configuration:")
    for k, v in vars(args2).items():
        print(f"  {k}: {v}")
    model, model_without_ddp, n_parameters, criterion, optimizer, loss_scaler = prepare_model(args2)
    return model,args2

def split_dr_pickle_stratified(
    input_pkl_path,
    output_dir=None,
    train_ratio=0.9,
    seed=42,
    overwrite=False,
):
  

    """
    Stratified split for DR pickle label file.

    Input pickle format:
        [
            ["STDR052-20160831@161220-R4-S", 1],
            ["STDR052-20160831@161220-L1-S", 1],
            ...
        ]

    Output:
        train_split.pkl
        val_split.pkl

    Each class is split independently according to train_ratio,
    so train/val will have approximately the same class ratio.
    """
  
    import os
    import pickle
    import random
    from pathlib import Path
    from collections import defaultdict, Counter

    input_pkl_path = Path(input_pkl_path)

    if output_dir is None:
        output_dir = input_pkl_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_pkl_path = output_dir / f"{input_pkl_path.stem}_train_split.pkl"
    val_pkl_path = output_dir / f"{input_pkl_path.stem}_val_split.pkl"

    if train_pkl_path.exists() and val_pkl_path.exists() and not overwrite:
        print(f"[Info] Split files already exist. Reuse them:")
        print(f"  train: {train_pkl_path}")
        print(f"  val:   {val_pkl_path}")
        return str(train_pkl_path), str(val_pkl_path)

    with open(input_pkl_path, "rb") as f:
        raw_items = pickle.load(f)

    # Normalize format
    items = []
    for item in raw_items:
        img_id = str(item[0])
        label = int(item[1])
        items.append([img_id, label])

    samples_by_class = defaultdict(list)
    for img_id, label in items:
        samples_by_class[label].append([img_id, label])

    rng = random.Random(seed)

    train_items = []
    val_items = []

    print(f"[Info] Original label counts: {dict(Counter([x[1] for x in items]))}")

    for label in sorted(samples_by_class.keys()):
        cls_items = samples_by_class[label]
        rng.shuffle(cls_items)

        num_cls = len(cls_items)

        if num_cls < 2:
            raise RuntimeError(
                f"Class {label} has only {num_cls} sample(s). "
                f"Cannot guarantee both train and val contain this class."
            )

        num_train = int(round(num_cls * train_ratio))

        # Ensure both train and val contain this class
        num_train = max(1, min(num_train, num_cls - 1))

        train_cls_items = cls_items[:num_train]
        val_cls_items = cls_items[num_train:]

        train_items.extend(train_cls_items)
        val_items.extend(val_cls_items)

        print(
            f"[Class {label}] total={num_cls}, "
            f"train={len(train_cls_items)}, val={len(val_cls_items)}"
        )

    # Shuffle final train/val lists, but deterministically
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    with open(train_pkl_path, "wb") as f:
        pickle.dump(train_items, f)

    with open(val_pkl_path, "wb") as f:
        pickle.dump(val_items, f)

    print("[Info] Saved stratified split:")
    print(f"  train: {train_pkl_path}")
    print(f"  val:   {val_pkl_path}")
    print(f"[Info] Train label counts: {dict(Counter([x[1] for x in train_items]))}")
    print(f"[Info] Val label counts:   {dict(Counter([x[1] for x in val_items]))}")

    return str(train_pkl_path), str(val_pkl_path)

import re
import torch.nn.functional as F
from torchvision.utils import save_image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def safe_stem_from_path_or_id(path_or_id):
    """
    Compatible with:
        /path/to/xxx.png
        STDR052-20160831@161220-R4-S
    """
    stem = Path(str(path_or_id)).stem
    stem = re.sub(r"[^a-zA-Z0-9_.@+-]", "_", stem)
    return stem


def tensor_to_01_for_save(x, useimgnet=False):
    """
    Convert image tensor to [0, 1] for saving.
    x: [3, H, W]
    """
    x = x.detach().cpu()

    if useimgnet:
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)
        x = x * std + mean

    return torch.clamp(x, 0.0, 1.0)


def grad_to_vis(g, mode="abs"):
    """
    Convert raw gradient to [0, 1] visualization.
    Raw gradient should still be saved as .pt.
    """
    g = g.detach().cpu()

    if mode == "abs":
        g = g.abs()
        g = g / (g.max() + 1e-12)
    else:
        g_min = g.min()
        g_max = g.max()
        g = (g - g_min) / (g_max - g_min + 1e-12)

    return torch.clamp(g, 0.0, 1.0)

@torch.no_grad()
def genGradPair(train=True,datasetBase="",MODELS_PATH="",savepath="",changepara=False,changegauSigma=False,seed=0,useimgnet=False,ntype='uniform',args=None):
    from trainRIC import DRPickleDatasetWithPath,ImageFolderWithPath

    # lpips_model = lpips.LPIPS(net='alex',pretrained=True).to(device)
    # lpips_model.eval()

    # 数据路径：指向包含 class_0 / class_1 的目录
    target_model, modelargs = getTargetModel()
    target_model = target_model.eval().cuda()

    for p in target_model.parameters():
        p.requires_grad = False

    heresize = modelargs.input_size
    if args.data_mode == "dr_pickle":
        dataset = DRPickleDatasetWithPath(
            pkl_path=os.path.join(args.datasetBase,"rDR_test.pkl"),
            image_root=os.path.join(args.datasetBase,"DR"),
            transform=get_transform(size=heresize, useimgnet=True),
            recursive=False,
        )

    elif args.data_mode == "imagefolder":
        data_dir = f"{args.datasetBase}/train"

        dataset = ImageFolderWithPath(
            root=data_dir,
            transform=get_transform(size=heresize, useimgnet=True)
        )
        print(f"threatModelDeepUWF:Loading data from: {data_dir}")

    else:
        raise ValueError(f"Unsupported data_mode: {args.data_mode}")
    
    # loader = DataLoader(
    #     dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True
    # )


    loader = build_loader_from_manifest(
        dataset=dataset,
        manifest_txt="/path/to/workspace/dataset/DeepUWF/SH_DR/shuffled_train_images.txt",
        batch_size=32,
        num_samples=4000,
        num_workers=4,
        pin_memory=True,
    )

    #print("Class to index:", dataset.class_to_idx)
    print("Number of images:", len(dataset))

    # 确认class_0的label是0
    # for path, label in dataset.samples[:10]:
    #     print(path, label)
    # 读取一个 batch
 
    acctotal,total,cover_succ = 0,0,0
    l2loss_secrets,l2loss_covers,psnr_secrets,ssim_secrets,mean_pixel_errors=0.,0.,0.,0.,0.
    lpips_errors,lpips_scs,mse_scs,psnr_scs,ssim_scs = 0.,0.,0.,0.,0.
    ii= 0

    if not os.path.exists(savepath):
        os.makedirs(savepath,exist_ok=True)
    if not os.path.exists(savepath+"/test_oriimg"):
        os.makedirs(savepath+"/test_oriimg",exist_ok=True)
    if not os.path.exists(savepath+"/test_miximg"):
        os.makedirs(savepath+"/test_miximg",exist_ok=True)


    print(f"save to: {savepath}")
    for idx, test_batch in enumerate(loader):
        # if total>=1:
        #     break

        # if idx<2:
        #     continue
        test_secrets, labels,paths  = test_batch
        #print(f"paths:{paths}")
        test_secrets = test_secrets.to(device)
        assert test_secrets.shape[2] == test_secrets.shape[3], "Expected 3 channels in the input images"
        heresize = test_secrets.shape[2]  # Assuming square images, get the height (or width)
        labels= labels.to(device)
        # if int(labels) != target_labels:
        #     continue
        total += len(test_secrets)

        with torch.enable_grad():
            # --------------------------------------------------
            # Compute input gradients:
            # grad = d CE(target_model(test_secrets), labels) / d test_secrets
            # --------------------------------------------------
            x = test_secrets.detach().clone().requires_grad_(True)
            labels_for_grad = labels.long()

            target_model.zero_grad(set_to_none=True)

            logits = target_model(x)
            loss = F.cross_entropy(logits, labels_for_grad, reduction="sum")

            loss.backward()

            input_grads = x.grad.detach().cpu()
            x_cpu = x.detach().cpu()

            preds = torch.argmax(logits.detach(), dim=1)
            batch_correct = preds.eq(labels_for_grad).sum().item()
            acctotal += batch_correct

            # --------------------------------------------------
            # Save original images and gradients
            # --------------------------------------------------
            labels_cpu = labels_for_grad.detach().cpu()
            preds_cpu = preds.detach().cpu()

            for b in range(x_cpu.size(0)):
                path_or_id = paths[b]
                stem = safe_stem_from_path_or_id(path_or_id)

                label_int = int(labels_cpu[b].item())
                pred_int = int(preds_cpu[b].item())

                out_base = f"{ii:06d}_{stem}_y{label_int}_p{pred_int}"

                ori_img_01 = tensor_to_01_for_save(
                    x_cpu[b],
                    useimgnet=True,
                )

                grad_raw = input_grads[b]
                #grad_vis = grad_to_vis(grad_raw, mode="abs")
                grad_vis = grad_to_vis(grad_raw, mode="minmax")


                # Original image
                # ori_png_path = os.path.join(
                #     savepath,
                #     "test_oriimg",
                #     f"{out_base}.png",
                # )
                # save_image(ori_img_01, ori_png_path, normalize=False)

                # Raw gradient tensor
                grad_pt_path = os.path.join(
                    savepath,
                    "test_miximg",
                    f"{out_base}.pt",
                )

                torch.save(
                    {
                        "grad": grad_raw.float(),
                        "label": label_int,
                        "pred": pred_int,
                        "path": str(path_or_id),
                        "loss_type": "cross_entropy_sum",
                    },
                    grad_pt_path,
                )

                # Optional gradient visualization
                grad_png_path = os.path.join(
                    savepath,
                    "test_miximg",
                    f"{out_base}_grad_abs.png",
                )
                save_image(grad_vis, grad_png_path, normalize=False)

                ii += 1

            print(
                f"[Batch {idx}] "
                f"acc={acctotal / max(total, 1):.4f}, "
                f"loss={loss.item():.6f}, "
                f"saved={ii}"
            )

            del x, logits, loss, input_grads
            
    print("finished.") 

if __name__ == "__main__":

    import argparse

    args = parse_args()

    changepara = False if args.changepara == 0 else True 
    changegauSigma = False if args.changegauSigma == 0 else True 
    useimgnet = False if args.useimgnet == 0 else True 


    setSeed(args.seed)
    # threatModelDeepUWF(train=True,datasetBase=args.datasetBase,MODELS_PATH=args.modelBase,\
    #       savepath=args.savepath,changepara=changepara,changegauSigma=changegauSigma,seed=args.seed,useimgnet=useimgnet,ntype=args.ntype,args=args)

    threatModelDeepUWFFortrainClassifier(train=True,datasetBase=args.datasetBase,MODELS_PATH=args.modelBase,\
        savepath=args.savepath,changepara=changepara,changegauSigma=changegauSigma,seed=args.seed,useimgnet=useimgnet,ntype=args.ntype,args=args)



    #genGradPair(train=True,datasetBase=args.datasetBase,MODELS_PATH=args.modelBase,\
     #   savepath=args.savepath,changepara=changepara,changegauSigma=changegauSigma,seed=args.seed,useimgnet=useimgnet,ntype=args.ntype,args=args)

    # testDeepUWFRICAUROC(train=True,datasetBase=args.datasetBase,MODELS_PATH=args.modelBase,\
    #     savepath=args.savepath,changepara=changepara,changegauSigma=changegauSigma,seed=args.seed,useimgnet=useimgnet,ntype=args.ntype,args=args)

