import os,sys
import argparse
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import csv,torch
from pathlib import Path
from PIL import Image
import pickle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
print(f"CURRENT_DIR:{CURRENT_DIR}")
print(f"PROJECT_ROOT:{PROJECT_ROOT}")
print(sys.path)
from generatePair import getAdvZChaPara,setSeed,get_transform,_std_torch,_mean_torch,getTargetModel,load_checkpoint,device,beta
import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
from collections import defaultdict

from collections import OrderedDict
import numpy as np
from torchvision.utils import make_grid,save_image
import cw_attack
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, label, path



import pickle
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class DRPickleDatasetWithPath(Dataset):
    """
    Dataset for DR pickle labels.

    Expected pickle format:
        [
            ["STDR052-20160831@161220-R4-S", 1],
            ["STDR052-20160831@161220-L1-S", 1],
            ...
        ]

    It returns:
        image_tensor, label, image_id

    Here image_id is used as the key to load fixed NAE covers.
    """

   
    def __init__(
        self,
        pkl_path,
        image_root,
        transform=None,
        extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        recursive=True,
        split_name=None,
    ):
        self.pkl_path = Path(pkl_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.extensions = extensions
        self.split_name = split_name
        with open(self.pkl_path, "rb") as f:
            raw_items = pickle.load(f)
            

        self.items = [(str(item[0]), int(item[1])) for item in raw_items]

        self.path_map = self._build_path_map(recursive=recursive)

        self.samples = []
        missing = []

        for img_id, label in self.items:
            if img_id in self.path_map:
                self.samples.append((self.path_map[img_id], label, img_id))
            else:
                missing.append(img_id)

        if len(missing) > 0:
            print(f"[Warning] Missing {len(missing)} DR images.")
            print("First 10 missing image ids:")
            for x in missing[:10]:
                print("  ", x)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid DR image found. "
                f"Please check pkl_path={self.pkl_path}, image_root={self.image_root}"
            )

        
        print(f"Loaded DR pickle dataset: {self.pkl_path}")
        if self.split_name is not None:
            print(f"Split name: {self.split_name}")
        print(f"Image root: {self.image_root}")
        print(f"Total valid samples: {len(self.samples)}")
        print(f"Label counts: {self._label_counts()}")

    def _build_path_map(self, recursive=True):
        all_files = []

        for ext in self.extensions:
            if recursive:
                all_files.extend(self.image_root.rglob(f"*{ext}"))
                all_files.extend(self.image_root.rglob(f"*{ext.upper()}"))
            else:
                all_files.extend(self.image_root.glob(f"*{ext}"))
                all_files.extend(self.image_root.glob(f"*{ext.upper()}"))

        path_map = {}

        for path in all_files:
            stem = path.stem

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

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), img_id

def collect_pickle_samples_by_class(pkl_path):
    """
    Load DR pickle label file.

    Expected format:
        [
            ["STDR052-20160831@161220-R4-S", 1],
            ["STDR052-20160831@161220-L1-S", 1],
            ...
        ]

    Returns:
        samples_by_class:
            {
                0: [(img_id, 0), ...],
                1: [(img_id, 1), ...],
            }
    """
    with open(pkl_path, "rb") as f:
        raw_items = pickle.load(f)

    samples_by_class = defaultdict(list)

    for item in raw_items:
        img_id = str(item[0])
        label = int(item[1])
        samples_by_class[label].append((img_id, label))

    samples_by_class = dict(samples_by_class)

    print(f"Loaded pickle: {pkl_path}")
    print(f"Total samples: {len(raw_items)}")
    print("Class counts:")
    for cls_idx in sorted(samples_by_class.keys()):
        print(f"  class {cls_idx}: {len(samples_by_class[cls_idx])}")

    return samples_by_class

def get_cover_path(cover_root, ntype, label, image_path):
    """
    Build the saved cover path.

    Expected structure:
        cover_root/
            gaussian/
                0/
                    xxx.pt
                1/
                    yyy.pt
            uniform/
                0/
                    xxx.pt
                1/
                    yyy.pt
    """
    label_int = int(label)
    filename = Path(image_path).stem + ".pt"
    return Path(cover_root) / ntype / str(label_int) / filename


    torch.save(cover_fp16, str(save_path))
def get_fixed_cover_path(cover_root, ntype, label, path_or_id):
    label_int = int(label)
    # path_or_id can be:

    #   /xxx/yyy/image.png

    #   STDR052-20160831@161220-R4-S
    out_name = safe_output_name(str(path_or_id), suffix=".pt")
    return Path(cover_root) / ntype / str(label_int) / out_name




def load_cover_fp16(load_path):
    load_path = Path(load_path)

    try:
        cover = torch.load(str(load_path), map_location="cpu", weights_only=True)
    except TypeError:
        cover = torch.load(str(load_path), map_location="cpu")

    if not torch.is_tensor(cover):
        raise RuntimeError(f"Invalid cover file, not a tensor: {load_path}")

    return cover.float()


def save_cover_fp16(cover_tensor, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cover_fp16 = torch.clamp(cover_tensor.detach().cpu(), 0.0, 1.0).half()
    torch.save(cover_fp16, str(save_path))


# def get_fixed_cover_path(cover_root, ntype, label, path_or_id):
#     label_int = int(label)

#     # path_or_id can be:
#     #   /xxx/yyy/image.png
#     #   STDR052-20160831@161220-R4-S
#     out_name = safe_output_name(str(path_or_id), suffix=".pt")

#     return Path(cover_root) / ntype / str(label_int) / out_name


def load_or_generate_fixed_covers(
    paths,
    labels,
    cover_root,
    ntype,
    heresize,
    target_model,
    args,
    device,
):
    """
    Load pre-generated fixed NAE covers.
    If some covers are missing, generate only the missing ones online.

    Compatible with:
        1. ImageFolder paths
        2. DR pickle image_ids
    """
    cover_root = Path(cover_root)

    labels_cpu = labels.detach().cpu().tolist()

    covers = [None for _ in range(len(paths))]
    missing_indices = []
    missing_labels = []
    missing_paths = []
    missing_cover_paths = []

    for i, path_or_id in enumerate(paths):
        label_i = int(labels_cpu[i])

        cover_path = get_fixed_cover_path(
            cover_root=cover_root,
            ntype=ntype,
            label=label_i,
            path_or_id=path_or_id,
        )

        if cover_path.exists():
            cover = load_cover_fp16(cover_path)
            covers[i] = cover
        else:
            missing_indices.append(i)
            missing_labels.append(label_i)
            missing_paths.append(path_or_id)
            missing_cover_paths.append(cover_path)

    succ = len(paths) - len(missing_indices)

    if len(missing_indices) > 0:
        print(
            f"[Warning] Missing {len(missing_indices)} fixed covers for ntype={ntype}. "
            f"Generating online..."
        )

        missing_labels_tensor = torch.tensor(
            missing_labels,
            dtype=torch.long,
            device=device,
        )

        with torch.enable_grad():
            gen_covers, gen_succ = getAdvZChaPara(
                heresize,
                missing_labels_tensor,
                len(missing_indices),
                train=True,
                seed=args.seed,
                type=ntype,
                changepara=args.changepara,
                changegauSigma=args.changegauSigma,
                target_model=target_model,
            )

        gen_covers = gen_covers.detach().cpu()

        for j, original_index in enumerate(missing_indices):
            cover = torch.clamp(gen_covers[j], 0.0, 1.0)
            covers[original_index] = cover

            save_cover_fp16(
                cover,
                missing_cover_paths[j],
            )

        succ += int(gen_succ)

    covers = torch.stack(covers, dim=0).to(device, non_blocking=True)

    return covers, succ

# def load_or_generate_fixed_covers(
#     paths,
#     labels,
#     cover_root,
#     ntype,
#     heresize,
#     target_model,
#     args,
#     device,
# ):
#     """
#     For each sample in the current batch:
#     1. If fixed cover exists, load it.
#     2. If not exists, generate it with getAdvZChaPara(), save it, and return it.

#     Return:
#         covers: [B, 3, H, W], float32, on device
#         succ_for_log: int, only for rough logging
#     """
#     batch_size = len(paths)
#     labels_cpu = labels.detach().cpu().long()

#     covers_cpu = [None for _ in range(batch_size)]


#     missing_indices = []
#     missing_labels = []
#     missing_paths = []

#     # 1. Try loading existing covers
#     for i, path in enumerate(paths):
#         label_int = int(labels_cpu[i].item())
#         cover_path = get_cover_path(
#             cover_root=cover_root,
#             ntype=ntype,
#             label=label_int,
#             image_path=path,
#         )

#         if cover_path.exists():
#             covers_cpu[i] = load_cover_fp16(cover_path)
#         else:
#             missing_indices.append(i)
#             missing_labels.append(label_int)
#             missing_paths.append(path)

#     succ_for_log = batch_size - len(missing_indices)

#     # 2. Generate missing covers only
#     if len(missing_indices) > 0:
#         missing_labels_tensor = torch.tensor(
#             missing_labels,
#             dtype=torch.long,
#             device=device,
#         )

#         # Use a deterministic but changing seed for this missing subset.
#         # This avoids generating identical covers if getAdvZChaPara resets seed internally.
#         batch_seed = (
#             int(args.seed)
#             + int(labels_cpu.sum().item()) * 1000
#             + len(missing_indices) * 17
#             + (0 if ntype == "gaussian" else 1000000)
#         )

#         generated_covers, succ = getAdvZChaPara(
#             heresize,
#             missing_labels_tensor,
#             len(missing_indices),
#             train=True,
#             seed=batch_seed,
#             type=ntype,
#             changepara=args.changepara,
#             changegauSigma=args.changegauSigma,
#             target_model=target_model,
#         )

#         generated_covers_cpu = generated_covers.detach().cpu()

#         # Save generated covers and put them back to the correct batch positions
#         for j, batch_idx in enumerate(missing_indices):
#             label_int = missing_labels[j]
#             image_path = missing_paths[j]

#             cover_path = get_cover_path(
#                 cover_root=cover_root,
#                 ntype=ntype,
#                 label=label_int,
#                 image_path=image_path,
#             )

#             save_cover_fp16(generated_covers_cpu[j], cover_path)
#             covers_cpu[batch_idx] = generated_covers_cpu[j].float()

#         try:
#             succ_for_log += int(succ)
#         except Exception:
#             succ_for_log += len(missing_indices)

#     # 3. Safety check
#     for i, c in enumerate(covers_cpu):
#         if c is None:
#             raise RuntimeError(f"Cover at batch index {i} is still None.")

#     covers = torch.stack(covers_cpu, dim=0).to(device, non_blocking=True)
#     return covers, succ_for_log

def collect_imagefolder_samples_by_class(data_dir):
    """
    Collect ImageFolder samples and group them by class index.

    Return:
        dataset: ImageFolder object
        samples_by_class: dict[class_idx] -> list[(path, label)]
    """
    dataset = datasets.ImageFolder(root=data_dir)

    samples_by_class = {}
    for path, label in dataset.samples:
        samples_by_class.setdefault(label, []).append((path, label))

    # Make order deterministic
    for label in samples_by_class:
        samples_by_class[label] = sorted(samples_by_class[label], key=lambda x: x[0])

    return dataset, samples_by_class

def load_fixed_covers(paths, labels, cover_root, ntype, device):
    covers = []

    for path, label in zip(paths, labels):
        label_int = int(label.item()) if torch.is_tensor(label) else int(label)
        filename = Path(path).stem + ".pt"

        cover_path = Path(cover_root) / ntype / str(label_int) / filename

        if not cover_path.exists():
            raise FileNotFoundError(f"Pre-generated cover not found: {cover_path}")

        cover = torch.load(str(cover_path), map_location="cpu")

        # float16 -> float32, keep value range [0, 1]
        cover = cover.float()

        covers.append(cover)

    covers = torch.stack(covers, dim=0).to(device, non_blocking=True)
    return covers
    
def safe_output_name(input_path, suffix=".pt"):
    p = Path(input_path)
    return p.stem + suffix


import torch
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MyClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        # x: [B, C, H, W], value range [0, 1]
        if x.dim() != 4:
            raise ValueError(f"Expected x to be [B, C, H, W], but got shape {x.shape}")

        if x.size(1) != 3:
            raise ValueError(f"Expected 3-channel input, but got shape {x.shape}")

        # Optional runtime check
        assert torch.min(x) >= 0, f"x min is {torch.min(x).item()}"
        assert torch.max(x) <= 1, f"x max is {torch.max(x).item()}"

        x = (x - self.mean) / self.std
        return self.model(x)


@torch.no_grad()
def pre_generate_nae_covers_for_dr_pickle(
    args,
    save_root,
    pkl_path,
    class_counts=None,
    batch_size=32,
    overwrite=False,
):
    """
    Pre-generate fixed Gaussian and Uniform NAE/cover images for DR pickle training.

    Pickle format:
        [
            ["STDR052-20160831@161220-R4-S", 1],
            ["STDR052-20160831@161220-L1-S", 1],
            ...
        ]

    It generates:
        gaussian/0, gaussian/1
        uniform/0, uniform/1

    Each generated cover is saved with the same image_id stem from the pickle file.
    """

    setSeed(args.seed)

    target_model, modelargs = getTargetModel()
    target_model = MyClassifier(target_model)
    target_model = target_model.to(device)
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    heresize = modelargs.input_size

    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    samples_by_class = collect_pickle_samples_by_class(pkl_path)

    if class_counts is None:
        class_counts = {
            cls_idx: len(samples)
            for cls_idx, samples in samples_by_class.items()
        }

    for cls_idx, required_num in class_counts.items():
        if cls_idx not in samples_by_class:
            raise RuntimeError(
                f"Class index {cls_idx} not found in pickle. "
                f"Available class indices: {list(samples_by_class.keys())}"
            )

        available_num = len(samples_by_class[cls_idx])
        if available_num < required_num:
            raise RuntimeError(
                f"Class {cls_idx} has only {available_num} samples, "
                f"but required {required_num}."
            )

    manifest_path = save_root / "manifest.csv"

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "ntype",
            "class_idx",
            "image_id",
            "saved_path",
            "filename",
            "storage_format",
        ])

        for ntype in ["gaussian", "uniform"]:
            for cls_idx, required_num in class_counts.items():
                selected_samples = samples_by_class[cls_idx][:required_num]

                out_dir = save_root / ntype / str(cls_idx)
                out_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"Generating {ntype} covers for class {cls_idx}: "
                    f"{required_num} samples -> {out_dir}"
                )

                for start in range(0, required_num, batch_size):
                    end = min(start + batch_size, required_num)
                    batch_samples = selected_samples[start:end]

                    cur_bs = len(batch_samples)

                    labels = torch.full(
                        (cur_bs,),
                        fill_value=cls_idx,
                        dtype=torch.long,
                        device=device,
                    )

                    batch_seed = (
                        args.seed
                        + start
                        + cls_idx * 100000
                        + (0 if ntype == "gaussian" else 1000000)
                    )

                    with torch.enable_grad():
                        covers, succ = getAdvZChaPara(
                            heresize,
                            labels,
                            cur_bs,
                            train=True,
                            seed=batch_seed,
                            type=ntype,
                            changepara=args.changepara,
                            changegauSigma=args.changegauSigma,
                            target_model=target_model,
                        )

                    covers = covers.detach().cpu()

                    for i, (img_id, label) in enumerate(batch_samples):
                        out_name = safe_output_name(img_id, suffix=".pt")
                        out_path = out_dir / out_name

                        if out_path.exists() and not overwrite:
                            writer.writerow([
                                ntype,
                                cls_idx,
                                img_id,
                                str(out_path),
                                out_name,
                                "float16_pt",
                            ])
                            continue

                        cover_fp16 = torch.clamp(covers[i], 0.0, 1.0).half().cpu()
                        torch.save(cover_fp16, str(out_path))

                        writer.writerow([
                            ntype,
                            cls_idx,
                            img_id,
                            str(out_path),
                            out_name,
                            "float16_pt",
                        ])

                    print(
                        f"[{ntype}][class {cls_idx}] "
                        f"{end}/{required_num}, succ={succ}, "
                        f"asr={succ / max(len(covers), 1):.4f}"
                    )

    print(f"Done. Manifest saved to: {manifest_path}")
    
# @torch.no_grad()
# def pre_generate_nae_covers_for_shdr(
#     args,
#     save_root,
#     class_counts=None,
#     batch_size=32,
#     overwrite=False,
# ):
#     """
#     Pre-generate fixed Gaussian and Uniform NAE/cover images for SHDR training.

#     It generates:
#         gaussian/class0, gaussian/class1
#         uniform/class0, uniform/class1

#     Each generated cover is saved with the same filename stem as the original SHDR image.
#     A manifest.csv is also saved for later checking.

#     Args:
#         args: argparse args, must include datasetBase, useimgnet, changepara,
#               changegauSigma, seed.
#         save_root: output directory for pre-generated covers.
#         class_counts: dict, e.g., {0: 18000, 1: 3100}
#         batch_size: generation batch size
#         overwrite: whether to regenerate existing files
#     """
#     if class_counts is None:
#         class_counts = {
#             0: 18000,
#             1: 3100,
#         }

#     setSeed(args.seed)

#     target_model, modelargs = getTargetModel()
#     target_model = MyClassifier(target_model)
#     target_model = target_model.to(device)
#     target_model.eval()
#     for p in target_model.parameters():
#         p.requires_grad = False

#     heresize = modelargs.input_size
#     print(f"heresize:{heresize}")
#     data_dir = f"{args.datasetBase}/train"
#     save_root = Path(save_root)
#     save_root.mkdir(parents=True, exist_ok=True)

#     dataset, samples_by_class = collect_imagefolder_samples_by_class(data_dir)

#     print("ImageFolder classes:", dataset.classes)
#     print("ImageFolder class_to_idx:", dataset.class_to_idx)

#     for cls_idx, required_num in class_counts.items():
#         if cls_idx not in samples_by_class:
#             raise RuntimeError(
#                 f"Class index {cls_idx} not found in dataset. "
#                 f"Available class indices: {list(samples_by_class.keys())}"
#             )

#         available_num = len(samples_by_class[cls_idx])
#         if available_num < required_num:
#             raise RuntimeError(
#                 f"Class {cls_idx} has only {available_num} images, "
#                 f"but required {required_num}."
#             )

#     manifest_path = save_root / "manifest.csv"

#     with open(manifest_path, "w", newline="") as f:
#         writer = csv.writer(f)

#         writer.writerow([
#             "ntype",
#             "class_idx",
#             "original_path",
#             "saved_path",
#             "filename",
#             "storage_format",
#         ])

#         for ntype in ["gaussian", "uniform"]:
#             for cls_idx, required_num in class_counts.items():
#                 selected_samples = samples_by_class[cls_idx][:required_num]

#                 out_dir = save_root / ntype / str(cls_idx)
#                 out_dir.mkdir(parents=True, exist_ok=True)

#                 print(
#                     f"Generating {ntype} covers for class {cls_idx}: "
#                     f"{required_num} images -> {out_dir}"
#                 )

#                 for start in range(0, required_num, batch_size):
#                     end = min(start + batch_size, required_num)
#                     batch_samples = selected_samples[start:end]

#                     cur_bs = len(batch_samples)
#                     labels = torch.full(
#                         (cur_bs,),
#                         fill_value=cls_idx,
#                         dtype=torch.long,
#                         device=device
#                     )

#                     # Important:
#                     # Use a changing seed for each batch to avoid repeated covers
#                     # if getAdvZChaPara internally resets random seed.
#                     batch_seed = args.seed + start + cls_idx * 100000 + (0 if ntype == "gaussian" else 1000000)

#                     with torch.enable_grad():
#                         covers, succ = getAdvZChaPara(
#                             heresize,
#                             labels,
#                             cur_bs,
#                             train=True,
#                             seed=batch_seed,
#                             type=ntype,
#                             changepara=args.changepara,
#                             changegauSigma=args.changegauSigma,
#                             target_model=target_model,
#                         )

#                     covers = covers.detach().cpu()

#                     for i, (ori_path, label) in enumerate(batch_samples):

#                         out_name = safe_output_name(ori_path, suffix=".pt")
#                         out_path = out_dir / out_name

#                         if out_path.exists() and not overwrite:
#                             writer.writerow([
#                                 ntype,
#                                 cls_idx,
#                                 ori_path,
#                                 str(out_path),
#                                 out_name,
#                                 "float16_pt",
#                             ])
#                             continue

#                         cover_fp16 = torch.clamp(covers[i], 0.0, 1.0).half().cpu()

#                         torch.save(cover_fp16, str(out_path))

#                         writer.writerow([
#                             ntype,
#                             cls_idx,
#                             ori_path,
#                             str(out_path),
#                             out_name,
#                             "float16_pt",
#                         ])


#                     print(
#                         f"[{ntype}][class {cls_idx}] "
#                         f"{end}/{required_num}, succ={succ}, asr={succ/len(covers)}"
#                     )

#     print(f"Done. Manifest saved to: {manifest_path}")


@torch.no_grad()
def pre_generate_nae_covers_for_shdr(
    args,
    save_root,
    class_counts=None,
    batch_size=32,
    overwrite=False,
):
    """
    Pre-generate fixed Gaussian and Uniform NAE/cover images for SHDR training.

    Existing .pt files are preserved when overwrite=False.
    Interrupted generation can therefore be resumed safely.
    """
    if class_counts is None:
        class_counts = {
            0: 18000,
            1: 3100,
        }

    setSeed(args.seed)

    target_model, modelargs = getTargetModel()
    target_model = MyClassifier(target_model)
    target_model = target_model.to(device)
    target_model.eval()

    for parameter in target_model.parameters():
        parameter.requires_grad_(False)

    heresize = modelargs.input_size
    print(f"heresize: {heresize}")

    data_dir = f"{args.datasetBase}/train"

    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    dataset, samples_by_class = collect_imagefolder_samples_by_class(
        data_dir
    )

    print("ImageFolder classes:", dataset.classes)
    print("ImageFolder class_to_idx:", dataset.class_to_idx)

    for cls_idx, required_num in class_counts.items():
        if cls_idx not in samples_by_class:
            raise RuntimeError(
                f"Class index {cls_idx} not found in dataset. "
                f"Available class indices: "
                f"{list(samples_by_class.keys())}"
            )

        available_num = len(samples_by_class[cls_idx])

        if available_num < required_num:
            raise RuntimeError(
                f"Class {cls_idx} has only {available_num} images, "
                f"but required {required_num}."
            )

    manifest_path = save_root / "manifest.csv"

    # Append instead of overwriting the old manifest.
    manifest_exists = (
        manifest_path.exists()
        and manifest_path.stat().st_size > 0
    )

    file_mode = "a" if manifest_exists else "w"

    with open(manifest_path, file_mode, newline="") as file_handle:
        writer = csv.writer(file_handle)

        if not manifest_exists:
            writer.writerow([
                "ntype",
                "class_idx",
                "original_path",
                "saved_path",
                "filename",
                "storage_format",
            ])

        for ntype in ["gaussian", "uniform"]:
            for cls_idx, required_num in class_counts.items():
                selected_samples = (
                    samples_by_class[cls_idx][:required_num]
                )

                out_dir = save_root / ntype / str(cls_idx)
                out_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                print(
                    f"Generating {ntype} covers for class {cls_idx}: "
                    f"{required_num} images -> {out_dir}"
                )

                for start in range(0, required_num, batch_size):
                    end = min(start + batch_size, required_num)
                    batch_samples = selected_samples[start:end]

                    cur_bs = len(batch_samples)
                    labels = torch.full(
                        (cur_bs,),
                        fill_value=cls_idx,
                        dtype=torch.long,
                        device=device
                    )

                    # Important:
                    # Use a changing seed for each batch to avoid repeated covers
                    # if getAdvZChaPara internally resets random seed.
                    batch_seed = args.seed + start + cls_idx * 100000 + (0 if ntype == "gaussian" else 1000000)

                    with torch.enable_grad():
                        covers, succ = getAdvZChaPara(
                            heresize,
                            labels,
                            cur_bs,
                            train=True,
                            seed=batch_seed,
                            type=ntype,
                            changepara=args.changepara,
                            changegauSigma=args.changegauSigma,
                            target_model=target_model,
                        )

                    covers = covers.detach().cpu()

                    for i, (ori_path, label) in enumerate(batch_samples):


                        out_name = safe_output_name(ori_path, suffix=".pt")
                        out_path = out_dir / out_name

                        if out_path.exists() and not overwrite:
                            writer.writerow([
                                ntype,
                                cls_idx,
                                ori_path,
                                str(out_path),
                                out_name,
                                "float16_pt",
                            ])
                            continue

                        cover_fp16 = torch.clamp(covers[i], 0.0, 1.0).half().cpu()

                        torch.save(cover_fp16, str(out_path))

                        writer.writerow([
                            ntype,
                            cls_idx,
                            ori_path,
                            str(out_path),
                            out_name,
                            "float16_pt",
                        ])


                    print(
                        f"[{ntype}][class {cls_idx}] "
                        f"{end}/{required_num}, succ={succ}, asr={succ/len(covers)}"
                    )

    print(f"Done. Manifest saved to: {manifest_path}")



# def getCEloss(labels,outputs):
#     one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

#     i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
#     j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

#     return torch.clamp((i-j), min=-kappa)
    
def getCEloss(labels, outputs, kappa=5):
    labels = labels.long().to(outputs.device)

    num_classes = outputs.shape[1]

    one_hot_labels = torch.eye(
        num_classes,
        device=outputs.device,
        dtype=torch.bool
    )[labels]

    other_logits = outputs.masked_fill(one_hot_labels, -1e9)

    i, _ = torch.max(other_logits, dim=1)
    j = outputs.gather(1, labels.view(-1, 1)).squeeze(1)

    return torch.clamp(i - j, min=-kappa)

def save_checkpoint(state, filename,is_best=False):
    checkpointname = filename+'_checkpoint.pth.tar'
    torch.save(state, checkpointname)
    if is_best:
        shutil.copyfile(checkpointname, filename+'_model_best.pth.tar')

def to_01_secret(S, mean, std):
    """
    S: normalized secret image, usually in [-1, 1] if mean=std=0.5
    return: [0, 1]
    """
    return torch.clamp(S * std + mean, 0.0, 1.0)


def to_01_cipher(mix_img):
    """
    mix_img: ciphertext / encrypted image.
    If your net already outputs [0, 1], this is fine.
    If it outputs [-1, 1], change this function accordingly.
    """
    return torch.clamp(mix_img, 0.0, 1.0)


def flatten_lowfreq(x, down_size=32):
    """
    Extract low-frequency representation by downsampling.

    x: [B, C, H, W], value range [0, 1]
    return: [B, C * down_size * down_size]
    """
    x_low = F.interpolate(
        x,
        size=(down_size, down_size),
        mode="bilinear",
        align_corners=False
    )
    return x_low.flatten(start_dim=1)


def standardize_features(x, eps=1e-6):
    """
    Feature-wise normalization for stable RBF kernel.
    """
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True) + eps)
    return x


def rbf_kernel(x, sigma=None, eps=1e-6):
    """
    x: [B, D]
    return: [B, B]
    """
    x = standardize_features(x)

    dist2 = torch.cdist(x, x, p=2).pow(2)

    if sigma is None:
        # Median heuristic. Detach to avoid unstable gradients through sigma.
        with torch.no_grad():
            valid = dist2[dist2 > eps]
            if valid.numel() == 0:
                sigma2 = torch.tensor(1.0, device=x.device)
            else:
                sigma2 = torch.median(valid)
                sigma2 = torch.clamp(sigma2, min=eps)
    else:
        sigma2 = torch.tensor(float(sigma) ** 2, device=x.device)

    K = torch.exp(-dist2 / (2.0 * sigma2 + eps))
    return K


def hsic_dependence_loss(x, y, down_size=32):
    """
    HSIC dependence loss between x and y.
    Smaller value means weaker statistical dependence.

    x, y: [B, C, H, W], value range [0, 1]
    """
    b = x.size(0)

    if b <= 1:
        return torch.tensor(0.0, device=x.device)

    x_feat = flatten_lowfreq(x, down_size=down_size)
    y_feat = flatten_lowfreq(y, down_size=down_size)

    K = rbf_kernel(x_feat)
    L = rbf_kernel(y_feat)

    H = torch.eye(b, device=x.device) - torch.ones((b, b), device=x.device) / b

    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic = torch.sum(Kc * Lc) / ((b - 1) ** 2)

    return hsic

def attack1_loss(
    S_prime,
    mix_img,
    S,
    C,
    B,
    labels,
    target_model,
    lambda_mi=0.0,
    mi_down_size=32,
):
    """
    S: original secret image, normalized
    C: NAE / cover image, usually in [0, 1]
    mix_img: encrypted image / ciphertext
    S_prime: recovered secret image
    """

    # authorized recovery loss
    S_01 = to_01_secret(S, _mean_torch, _std_torch)
    mix_01 = to_01_cipher(mix_img)

    loss_secret = torch.nn.functional.mse_loss(S_prime, S_01)

    # keep ciphertext close to NAE cover
    loss_cover = torch.nn.functional.mse_loss(mix_img, C)

    # classification preservation loss
    
    outputs = target_model(mix_img)
    classloss = torch.mean(getCEloss(labels, outputs))

    # plaintext-ciphertext dependence loss
    # This discourages mix_img from preserving low-frequency structures of S.
    if lambda_mi > 0:
        loss_mi = hsic_dependence_loss(
            S_01,
            mix_01,
            down_size=mi_down_size
        )
    else:
        loss_mi = torch.tensor(0.0, device=S.device)

    loss_all = (
        B * loss_secret
        + loss_cover
        + classloss
        + lambda_mi * loss_mi
    )

    return loss_all, loss_secret, classloss, loss_cover, loss_mi

# def attack1_loss(S_prime, mix_img, S,C,  B,labels):
#     ''' Calculates loss specified on the paper.'''
    

#     loss_secret = torch.nn.functional.mse_loss(S_prime,  S*_std_torch+_mean_torch)
#     loss_cover= torch.nn.functional.mse_loss(mix_img,  C)
#     outputs = target_model(mix_img)
#     classloss = torch.mean(getCEloss(labels,outputs))
    
#     loss_all =   B*loss_secret  + loss_cover + classloss
#     #loss_all = -loss_cover + B * loss_secret + classloss + loss_cover2

#     return loss_all, loss_secret,classloss,loss_cover

def plotloss(
    train_losses,
    train_loss_secret_history,
    attloss_history,
    loss_cover_history,
    mi_loss_history,
    outputname,
    epoch
):
    plt.clf()
    plt.plot(train_losses)
    plt.title('Total loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/lossCurve_{}.png'.format(outputname, epoch))

    plt.clf()
    plt.plot(train_loss_secret_history)
    plt.title('Secret recovery loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/secretlossCurve_{}.png'.format(outputname, epoch))

    plt.clf()
    plt.plot(attloss_history)
    plt.title('Classification loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/classlossCurve_{}.png'.format(outputname, epoch))

    plt.clf()
    plt.plot(loss_cover_history)
    plt.title('Cover loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/coverlossCurve_{}.png'.format(outputname, epoch))

    plt.clf()
    plt.plot(mi_loss_history)
    plt.title('Plaintext-ciphertext dependence loss')
    plt.ylabel('HSIC loss')
    plt.xlabel('Batch')
    plt.savefig('{}/milossCurve_{}.png'.format(outputname, epoch))

def print_cuda_mem(tag):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(
            f"[CUDA MEM] {tag}: "
            f"allocated={allocated:.2f}GB, "
            f"reserved={reserved:.2f}GB, "
            f"max_allocated={max_allocated:.2f}GB"
        )

def parse_args():
    parser = argparse.ArgumentParser(description="DeepUWF RIC / DP experiment")

    parser.add_argument(
        "--datasetBase",
        type=str,
        #default="/path/to/hpc/workspace/dataset/DeepUWF/matchedfile/SH_DR/shrdr",
        #default="/path/to/workspace/dataset/DeepUWF/DR",
        default="/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr",
        help="Base path of the dataset."
    )

    parser.add_argument(
        "--modelBase",
        type=str,
        default="/path/to/hpc/workspace/dataset/DeepUWF/RIC_weights/SHDR_ric_midown64",
        help="Base path of the RIC/model weights."
    )

    parser.add_argument(
        "--savepath",
        type=str,
        #default="/path/to/hpc/workspace/program_output/DeepSteg/",
        #default="/path/to/workspace/weights/DeepSteg/DR_ric/",
        #default="/path/to/workspace/weights/DeepSteg/SHDR_ric/", 
        default="/path/to/workspace/weights/DeepSteg/SHDR_ric_midown64/",    
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
        "--ric_learning_rate",
        type=float,
        default=1e-5,
        help=""
    )



    parser.add_argument(
        "--useimgnet",
        type=int,
        default=0,
        help=""
    )

    parser.add_argument(
        "--ric_batch_size",
        type=int,
        default=16,
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
        default="gaussian",
        help="gaussian or uniform"
    )

    parser.add_argument(
        "--lambda_mi",
        type=float,
        default=1,
        help="Weight for plaintext-ciphertext dependence penalty."
    )

    parser.add_argument(
        "--mi_down_size",
        type=int,
        #default=32,#for DR
        default=64,
        help="Downsample size for low-frequency HSIC dependence loss."
    )

    parser.add_argument(
        "--mi_warmup_epochs",
        type=int,
        default=1,
        help="Number of warm-up epochs before applying MI/HSIC loss."
    )

    parser.add_argument(
        "--pretrain",
        type=int,
        default=1,
        help="0:no;1:faceprertain;2:svhnpretrain"
    )


    parser.add_argument(
        "--nepochs",
        type=int,
        default=100,
        help="Number of warm-up epochs before applying MI/HSIC loss."
    )

    parser.add_argument(
        "--data_mode",
        type=str,
        #default="dr_pickle",
        default="imagefolder",
        help="dr_pickle (for DR ) or imagefolder (for other datasets)"
    )


    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--nrow", type=int, default=4)
    #utils paras
    parser.add_argument(
        "--pregenerate_covers",
        action="store_true",
        default=False,
        help="Pre-generate fixed Gaussian and Uniform covers before training."
    )

    parser.add_argument(
        "--cover_save_root",
        type=str,
        #default="/path/to/hpc/workspace/program_output/DeepSteg/pre_generated_covers/SHDR",
        #default="/path/to/workspace/dataset/DeepUWF/DR/NAE224model",
        default="/path/to/workspace/dataset/DeepUWF/SH_DR/NAE",        
        help="Path to save pre-generated covers."
    )

    parser.add_argument(
        "--cover_gen_batch_size",
        type=int,
        default=64,
        help="Batch size for pre-generating covers."
    )

    parser.add_argument(
        "--cover_overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing pre-generated cover images."
    )
    #parser.add_argument("--amp", action="store_true", default=True)
    ##utils paras end

    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    return args
