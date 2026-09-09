import logging,re
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def build_fixed_ids(images_dir, start, end):
    """
    Build fixed ids by image index.

    For each i in [start, end), find either:
        i_0.*
        i_1.*

    Return ids without file extension, e.g. "12_0", "13_1".
    """
    ids = []

    for i in range(start, end):
        candidates = []

        for suffix in [0, 1]:
            matched = list(Path(images_dir).glob(f"{i}_{suffix}.*"))
            if len(matched) > 0:
                candidates.extend(matched)

        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one file for index {i}, "
                f"but found {len(candidates)}: {candidates}"
            )

        ids.append(candidates[0].stem)

    return ids

class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        nae_dir: str,
        scale: float = 1.0,
        mask_suffix: str = '',
        split: str = 'train',
        transName: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.nae_dir = Path(nae_dir)
        self.transName = transName

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        assert split in ['train', 'test', 'val','all'], "split must be 'train', 'test', or 'all' , 'val'"

        self.scale = scale
        self.mask_suffix = mask_suffix
        self.split = split

        # hard-coded fixed split for 500 images:
        # train: 0_0.png ~ 449_0.png
        # test:  450_0.png ~ 499_0.png

        # train_ids = build_fixed_ids(self.images_dir, 0, 400)
        #val_ids = build_fixed_ids(self.images_dir, 400, 450)
        #test_ids = build_fixed_ids(self.images_dir, 450, 500)

        train_ids = build_fixed_ids(self.images_dir, 0, 3000)
        val_ids = build_fixed_ids(self.images_dir, 3000, 3500)
        test_ids = build_fixed_ids(self.images_dir, 3500, 4000)


        if split == 'train':
            self.ids = train_ids
        elif split == 'test':
            self.ids = test_ids
        elif split == 'val':
            self.ids = val_ids
    
        else:
            self.ids = train_ids + test_ids + val_ids

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating {split} dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        # 保持你原来的处理逻辑
        img_ndarray = np.asarray(pil_img)

        # 如果读进来是灰度图，转成 1 channel
        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255.0
        img_ndarray = img_ndarray * 2.0 - 1.0

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename).convert('RGB')

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_file = list(
            self.images_dir.glob(name + '.*')
        )


        if self.transName:
            # 例如：
            # name = "0_0"
            # name.split('_')[0] = "0"
            # int(...) = 0
            # f"{...:06d}" = "000000"
            #print(f"name: {name}")
            image_index = int(name.split('_')[0])
            #print(f"image_index: {image_index}")
            image_prefix = f"{image_index:06d}"
            #print(f"image_prefix: {image_prefix}")
            mask_file = list(
                self.masks_dir.glob(image_prefix + '_*')
            )
        else:
            mask_file = list(
            self.masks_dir.glob(name + self.mask_suffix + '.*')
            )


        if len(mask_file) != 1:
            raise RuntimeError(
                f"Expected exactly one mask for name={name}, "
                f"but found {len(mask_file)}: {mask_file}"
            )

        if len(img_file) != 1:
            raise RuntimeError(
                f"Expected exactly one image for name={name}, "
                f"prefix={image_prefix if transName else name}, "
                f"but found {len(img_file)}: {img_file}"
            )
        nae_file = list(self.nae_dir.glob(name + '.*'))

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(nae_file) == 1, \
            f'Either no nae or multiple nae found for the ID {name}: {nae_file}'

        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        nae = self.load(nae_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        assert img.size == nae.size, \
            f'Image and nae {name} should be the same size, but are {img.size} and {nae.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        nae = self.preprocess(nae, self.scale, is_mask=False)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous(),
            'nae': torch.as_tensor(nae.copy()).float().contiguous(),
            'id': name,
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, nae_dir, scale=1, split='train',transName=False):
        super().__init__(
            images_dir=images_dir,
            masks_dir=masks_dir,
            nae_dir=nae_dir,
            scale=scale,
            mask_suffix='',
            split=split,
            transName=transName
        )

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GradientPairDataset(Dataset):
    """
    Dataset for gradient-to-image reconstruction.

    Expected structure:
        images_dir:
            xxx.pt
            xxx_grad_abs.png   # ignored
        masks_dir:
            xxx.png

    Each xxx.pt is expected to be:
        {
            "grad": Tensor [3, H, W],
            "label": int,
            "pred": int,
            "path": str,
            ...
        }

    Returns:
        image: normalized gradient tensor [3, H, W]
        mask: original image tensor in [-1, 1], [3, H, W]
        nae: dummy zero tensor, for compatibility with old training loop
        id: sample id
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        scale=1.0,
        split="train",
        grad_norm="maxabs",
        totalnum=4000
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.split = split
        self.grad_norm = grad_norm

        assert split in ["train", "val", "test", "all"]
        assert 0 < scale <= 1

        all_pt_files = sorted(list(self.images_dir.glob("*.pt")))

        if len(all_pt_files) == 0:
            raise RuntimeError(f"No .pt gradient files found in {self.images_dir}")

        # Use only .pt files. Ignore *_grad_abs.png.
        all_ids = [p.stem for p in all_pt_files]

        # Keep only ids whose original image exists.
        valid_ids = []
        missing_masks = []

        for sample_id in all_ids:
            mask_file = self._find_mask_file(sample_id)
            if mask_file is None:
                missing_masks.append(sample_id)
            else:
                valid_ids.append(sample_id)

        if len(missing_masks) > 0:
            logging.warning(f"Missing {len(missing_masks)} original images.")
            logging.warning(f"First 10 missing ids: {missing_masks[:10]}")

        if len(valid_ids) == 0:
            raise RuntimeError(
                f"No valid gradient-image pairs found. "
                f"images_dir={self.images_dir}, masks_dir={self.masks_dir}"
            )

        # Fixed index split.
        # This follows the old setting:
        # train: first 400 samples
        # val:   next 50 samples
        # test:  next 50 samples
        if totalnum== 4000:
            train_ids = valid_ids[0:3000]
            val_ids = valid_ids[3000:3500]
            test_ids = valid_ids[3500:4000]
        else:
            print(f"default 2/8 splot")
            train_ids = valid_ids[0:int(len(valid_ids)*0.8)]
            val_ids = valid_ids[int(len(valid_ids)*0.8):int(len(valid_ids)*0.9)]
            test_ids = valid_ids[int(len(valid_ids)*0.9):int(len(valid_ids))]
    


        if split == "train":
            self.ids = train_ids
        elif split == "val":
            self.ids = val_ids
        elif split == "test":
            self.ids = test_ids
        else:
            self.ids = train_ids + val_ids + test_ids

        if len(self.ids) == 0:
            raise RuntimeError(
                f"No samples found for split={split}. "
                f"Total valid samples={len(valid_ids)}. "
                f"Expected fixed split train[0:400], val[400:450], test[450:500]."
            )

        logging.info(
            f"Creating GradientPairDataset split={split}, "
            f"num={len(self.ids)}, total_valid={len(valid_ids)}"
        )

        logging.info(
            f"Fixed split sizes: "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )
        
    def __len__(self):
        return len(self.ids)

    def _find_mask_file(self, sample_id):
        base_sample_id = re.sub(r"_p\d+$", "", sample_id)
        candidates = []
        
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidates.extend(list(self.masks_dir.glob(f"{base_sample_id}_p*{ext}")))
            candidates.extend(list(self.masks_dir.glob(f"{base_sample_id}_p*{ext.upper()}")))

        if len(candidates) == 0:
            return None

        if len(candidates) > 1:
            logging.warning(f"Multiple mask files found for {sample_id}: {candidates}")

        return candidates[0]

    @staticmethod
    def load_image_as_tensor_01(path):
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    @staticmethod
    def normalize_grad(grad, mode="maxabs"):
        grad = grad.float()

        if mode == "none":
            return grad

        if mode == "maxabs": #之前默认的
            denom = grad.abs().amax()
            grad = grad / (denom + 1e-12)
            grad = torch.clamp(grad, -1.0, 1.0)
            return grad

        if mode == "standard":
            mean = grad.mean()
            std = grad.std()
            grad = (grad - mean) / (std + 1e-12)
            grad = torch.clamp(grad, -5.0, 5.0) / 5.0
            return grad

        if mode == "abs_max":
            grad = grad.abs()
            grad = grad / (grad.amax() + 1e-12)
            grad = grad * 2.0 - 1.0
            return grad
        # if mode == "minmax":

        #     g = g.detach().cpu()

        #     grad = g.min()
        #     g_max = g.max()
        #     g = (g - g_min) / (g_max - g_min + 1e-12)


        raise ValueError(f"Unsupported grad_norm mode: {mode}")

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        grad_path = self.images_dir / f"{sample_id}.pt"
        mask_path = self._find_mask_file(sample_id)

        ckpt = torch.load(str(grad_path), map_location="cpu")

        if isinstance(ckpt, dict):
            grad = ckpt["grad"]
            label = ckpt.get("label", -1)
            pred = ckpt.get("pred", -1)
            original_path = ckpt.get("path", "")
        else:
            grad = ckpt
            label = -1
            pred = -1
            original_path = ""

        if grad.ndim != 3:
            raise RuntimeError(f"Expected grad [C,H,W], got {grad.shape} from {grad_path}")

        grad = self.normalize_grad(grad, mode=self.grad_norm)

        # target original image: [0,1] -> [-1,1], consistent with old CarvanaDataset
        mask = self.load_image_as_tensor_01(mask_path)
        mask = mask * 2.0 - 1.0

        # dummy nae, keep old training loop compatible
        nae = torch.zeros_like(grad)

        return {
            "image": grad.float().contiguous(),
            "mask": mask.float().contiguous(),
            "nae": nae.float().contiguous(),
            "id": sample_id,
            "label": int(label),
            "pred": int(pred),
            "path": str(original_path),
        }




class GradientPairCarvanaDataset(GradientPairDataset):
    def __init__(self, images_dir, masks_dir, nae_dir=None, scale=1, split="train"):
        super().__init__(
            images_dir=images_dir,
            masks_dir=masks_dir,
            scale=scale,
            split=split,
            grad_norm="maxabs",
        )
