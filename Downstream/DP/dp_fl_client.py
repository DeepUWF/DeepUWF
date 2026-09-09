"""
Federated Learning Client for DP FL Training
Adapted from Flower framework for DeepUWF
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import flwr as fl
from collections import OrderedDict
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from dp_utils import PrivacyBudget, DPTrainer
from pathlib import Path
from torchvision.utils import save_image
import torch.nn.functional as F
import os,re
from torchvision.utils import save_image



import hashlib
import hmac
import os


logger = logging.getLogger(__name__)

@staticmethod
def _normalize_sample_nonce(path_or_id):
    """
    Use only the filename, not the absolute path.

    Example:
        /root/class_0/024910-...-L5-S.png
    becomes:
        024910-...-L5-S.png
    """
    return Path(str(path_or_id)).name.encode(
        "utf-8"
    )


@staticmethod
def _derive_keyed_seed(
    master_key,
    nonce,
    purpose,
):
    """
    Derive an independent deterministic seed using HMAC-SHA256.

    Different purposes produce independent random streams.
    """
    if not isinstance(master_key, bytes):
        raise TypeError("master_key must be bytes.")

    if not isinstance(nonce, bytes):
        nonce = str(nonce).encode("utf-8")

    if not isinstance(purpose, bytes):
        purpose = str(purpose).encode("utf-8")

    digest = hmac.new(
        master_key,
        purpose + b"\x00" + nonce,
        hashlib.sha256,
    ).digest()

    # torch.Generator.manual_seed accepts a signed 64-bit-range seed
    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % (2**63 - 1)

def safe_stem_from_path_or_id(path_or_id):
    """
    Compatible with:
        /path/to/xxx.png
        STDR052-20160831@161220-R4-S
    """
    stem = Path(str(path_or_id)).stem
    stem = re.sub(r"[^a-zA-Z0-9_.@+-]", "_", stem)
    return stem


import torch
import torch.nn.functional as F


class DPFLClient(fl.client.NumPyClient):
    """
    Differential Privacy Federated Learning Client
    """


    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer_class: type,
        privacy_budget: PrivacyBudget,
        device: torch.device,
        client_id: int = 0,
        local_epochs: int = 1,
        dp_enabled: bool = True,
        optimizer_kwargs=None,
        checkpoint_path: str = None,
        evalmode: bool= False,
        load_optimizer_from_checkpoint: bool = False,
        save_exposed_gradients: bool = False,
        gradient_save_dir: str = None,
        gradient_save_interval: int = 1,
        gradient_save_max_batches_per_epoch: int = 1,
        gradient_save_max_total: Optional[int] = None,
        reconstruction_enabled: bool = False,
        attackType: str = "sample",

    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            criterion: 损失函数
            optimizer_class: 优化器类
            privacy_budget: 隐私预算配置
            device: torch device
            client_id: 客户端ID
            local_epochs: 本地训练轮数
            dp_enabled: 是否启用DP
        """
        self.attackType = attackType
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.privacy_budget = privacy_budget
        self.device = device
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.dp_enabled = dp_enabled
        self.save_exposed_gradients = bool(save_exposed_gradients)
        self.gradient_save_dir = gradient_save_dir
        self.gradient_save_interval = max(1, int(gradient_save_interval or 1))
        self.gradient_save_max_batches_per_epoch = int(
            gradient_save_max_batches_per_epoch
            if gradient_save_max_batches_per_epoch is not None
            else -1
        )
        self.gradient_save_max_total = (
            None
            if gradient_save_max_total is None
            else int(gradient_save_max_total)
        )
        self._gradient_saves_total = 0
        self._global_train_step = 0

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
                
        # 创建优化器


        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.setdefault("lr", 1e-3)

        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_kwargs,
        )

        print(f"[Client {self.client_id}] Optimizer: {optimizer_class.__name__}")
        print(f"[Client {self.client_id}] Optimizer kwargs: {optimizer_kwargs}")
        print(f"[Client {self.client_id}] evalmode: {evalmode}, checkpoint_path: {checkpoint_path}")
        if evalmode and checkpoint_path is not None and checkpoint_path != "":
            self.checkpoint_metadata = self.load_checkpoint(
                checkpoint_path=checkpoint_path,
                load_optimizer=load_optimizer_from_checkpoint,
                strict=True,
            )
            print(f"{checkpoint_path} loaded.")

        else:
            self.checkpoint_metadata = {}

        # DP 训练器
        if dp_enabled or (not dp_enabled and reconstruction_enabled):
            if dp_enabled :
                print(f"DP enabled. Privacy budget: {privacy_budget}")
            self.dp_trainer = DPTrainer(
                model=self.model,
                privacy_budget=privacy_budget,
                batch_size=train_loader.batch_size,
                dataset_size=len(train_loader.dataset),
                device=device,
            )
            if self.save_exposed_gradients:
                if not self.gradient_save_dir:
                    self.gradient_save_dir = os.path.join(
                        os.getcwd(),
                        "dp_exposed_gradients",
                    )
                os.makedirs(self.gradient_save_dir, exist_ok=True)
                print(
                    f"[Client {self.client_id}] Will save exposed DP gradients to: "
                    f"{self.gradient_save_dir}"
                )
        
        self.local_round = 0




    def _load_or_create_client_key(
        self,
        key_path,
        key_bytes=32,
    ):
        """
        Load a persistent client-specific master key.

        If the key does not exist, create it once and keep using it.
        """
        key_path = Path(key_path)
        key_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if key_path.exists():
            key = key_path.read_bytes()

            if len(key) < key_bytes:
                raise RuntimeError(
                    f"Invalid client key length: {len(key)} bytes. "
                    f"Expected at least {key_bytes} bytes."
                )

            print(
                f"[Client {self.client_id}] "
                f"Loaded client key from {key_path}"
            )

            return key[:key_bytes]

        key = os.urandom(key_bytes)

        temporary_path = key_path.with_suffix(
            key_path.suffix + ".tmp"
        )
        temporary_path.write_bytes(key)
        temporary_path.replace(key_path)

        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass

        print(
            f"[Client {self.client_id}] "
            f"Created client key at {key_path}"
        )

        return key



    def _keyed_transform_single(
        self,
        image,
        master_key,
        nonce,
        block_size=16,
    ):
        """
        Apply an exactly invertible, differentiable keyed transform.

        Args:
            image:
                Tensor [C, H, W].

            block_size:
                Patch size used for patch permutation.

        Returns:
            Transformed Tensor [C, H, W].
        """
        if image.ndim != 3:
            raise ValueError(
                f"Expected [C,H,W], got {tuple(image.shape)}"
            )

        channels, height, width = image.shape

        if height % block_size != 0:
            raise ValueError(
                f"Image height {height} is not divisible "
                f"by block_size={block_size}."
            )

        if width % block_size != 0:
            raise ValueError(
                f"Image width {width} is not divisible "
                f"by block_size={block_size}."
            )

        # ---------------------------------------------------------
        # 1. Channel permutation
        # ---------------------------------------------------------
        channel_seed = self._derive_keyed_seed(
            master_key=master_key,
            nonce=nonce,
            purpose=b"channel-permutation-v1",
        )

        channel_generator = torch.Generator(
            device="cpu"
        )
        channel_generator.manual_seed(
            channel_seed
        )

        channel_permutation = torch.randperm(
            channels,
            generator=channel_generator,
            device="cpu",
        ).to(image.device)

        transformed = image[channel_permutation]

        # ---------------------------------------------------------
        # 2. Patch permutation
        # ---------------------------------------------------------
        num_patch_rows = height // block_size
        num_patch_cols = width // block_size
        num_patches = num_patch_rows * num_patch_cols

        patch_seed = self._derive_keyed_seed(
            master_key=master_key,
            nonce=nonce,
            purpose=b"patch-permutation-v1",
        )

        patch_generator = torch.Generator(
            device="cpu"
        )
        patch_generator.manual_seed(
            patch_seed
        )

        patch_permutation = torch.randperm(
            num_patches,
            generator=patch_generator,
            device="cpu",
        ).to(image.device)

        # [C,H,W] -> [N,C,P,P]
        patches = (
            transformed
            .unfold(1, block_size, block_size)
            .unfold(2, block_size, block_size)
        )

        patches = (
            patches
            .permute(1, 2, 0, 3, 4)
            .contiguous()
            .view(
                num_patches,
                channels,
                block_size,
                block_size,
            )
        )

        patches = patches[
            patch_permutation
        ]

        # [N,C,P,P] -> [C,H,W]
        transformed = (
            patches
            .view(
                num_patch_rows,
                num_patch_cols,
                channels,
                block_size,
                block_size,
            )
            .permute(2, 0, 3, 1, 4)
            .contiguous()
            .view(
                channels,
                height,
                width,
            )
        )

        # ---------------------------------------------------------
        # 3. Keyed flips and cyclic spatial shift
        # ---------------------------------------------------------
        spatial_seed = self._derive_keyed_seed(
            master_key=master_key,
            nonce=nonce,
            purpose=b"spatial-transform-v1",
        )

        spatial_generator = torch.Generator(
            device="cpu"
        )
        spatial_generator.manual_seed(
            spatial_seed
        )

        transform_values = torch.randint(
            low=0,
            high=max(height, width, 2),
            size=(4,),
            generator=spatial_generator,
            device="cpu",
        ).tolist()

        use_vertical_flip = (
            transform_values[0] % 2 == 1
        )
        use_horizontal_flip = (
            transform_values[1] % 2 == 1
        )

        shift_y = (
            transform_values[2] % height
        )
        shift_x = (
            transform_values[3] % width
        )

        if use_vertical_flip:
            transformed = torch.flip(
                transformed,
                dims=(-2,),
            )

        if use_horizontal_flip:
            transformed = torch.flip(
                transformed,
                dims=(-1,),
            )

        transformed = torch.roll(
            transformed,
            shifts=(shift_y, shift_x),
            dims=(-2, -1),
        )

        return transformed


    def _keyed_transform_batch(
        self,
        images,
        master_key,
        nonces,
        block_size=16,
    ):
        """
        Batch version of the keyed pixel transform.
        """
        if images.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {tuple(images.shape)}"
            )

        if len(nonces) != images.shape[0]:
            raise ValueError(
                f"Expected {images.shape[0]} nonces, "
                f"but got {len(nonces)}."
            )

        transformed = [
            self._keyed_transform_single(
                image=images[sample_index],
                master_key=master_key,
                nonce=nonces[sample_index],
                block_size=block_size,
            )
            for sample_index in range(
                images.shape[0]
            )
        ]

        return torch.stack(
            transformed,
            dim=0,
        )

    def _build_keyed_carrier(
        self,
        reference_images,
        master_key,
        nonces,
    ):
        """
        Generate one deterministic uniform-noise carrier per sample.

        Args:
            reference_images:
                Tensor [B, C, H, W], used for shape/device/dtype.

            master_key:
                Client-specific bytes key.

            nonces:
                List[bytes], one nonce per sample.

        Returns:
            Tensor [B, C, H, W] in [0, 1].
        """
        batch_size = reference_images.shape[0]

        if len(nonces) != batch_size:
            raise ValueError(
                f"Expected {batch_size} nonces, "
                f"but received {len(nonces)}."
            )

        carriers = []

        for sample_index in range(batch_size):
            seed = self._derive_keyed_seed(
                master_key=master_key,
                nonce=nonces[sample_index],
                purpose=b"nae-carrier-v1",
            )

            generator = torch.Generator(
                device="cpu"
            )
            generator.manual_seed(seed)

            carrier = torch.rand(
                reference_images.shape[1:],
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )

            carriers.append(carrier)

        carrier_batch = torch.stack(
            carriers,
            dim=0,
        )

        return carrier_batch.to(
            device=reference_images.device,
            dtype=reference_images.dtype,
            non_blocking=True,
        )

    def _hierarchical_alignment_loss(
        self,
        nae_features,
        hierarchical_targets,
        layer_weights=None,
    ):
        """
        Align each NAE feature with its sample-specific hierarchical target.
        """
        if layer_weights is None:
            layer_weights = {
                "block_12": 0.25,
                "block_18": 0.30,
                "block_24": 0.15,
                "final": 0.30,
            }

        first_feature = next(
            iter(nae_features.values())
        )

        total_loss = torch.zeros(
            (),
            device=first_feature.device,
            dtype=first_feature.dtype,
        )

        statistics = {}

        for layer_name, nae_feature in nae_features.items():
            if layer_name not in hierarchical_targets:
                continue

            weight = float(
                layer_weights.get(layer_name, 0.0)
            )

            if weight == 0.0:
                continue

            nae_normalized = F.normalize(
                nae_feature,
                p=2,
                dim=1,
                eps=1e-8,
            )

            target = hierarchical_targets[
                layer_name
            ].to(
                device=nae_feature.device,
                dtype=nae_feature.dtype,
            )

            target = F.normalize(
                target,
                p=2,
                dim=1,
                eps=1e-8,
            )

            target_cosine = F.cosine_similarity(
                nae_normalized,
                target,
                dim=1,
                eps=1e-8,
            )

            alignment_loss = (
                1.0 - target_cosine
            ).mean()

            total_loss = (
                total_loss
                + weight * alignment_loss
            )

            statistics[layer_name] = {
                "hierarchical_loss": alignment_loss.detach(),
                "target_cosine": target_cosine.mean().detach(),
                "target_cosine_min": target_cosine.min().detach(),
                "target_cosine_max": target_cosine.max().detach(),
            }

        if not statistics:
            raise RuntimeError(
                "No matching layers were found between "
                "nae_features and hierarchical_targets."
            )

        return total_loss, statistics


    def _build_hierarchical_targets(
        self,
        plain_features,
        labels,
        subject_ids,
        class_prototypes,
        subject_prototypes,
        class_weight=0.2,
        subject_weight=0.6,
        sample_weight=0.2,
    ):
        """
        Construct one hierarchical target for each plaintext sample:

            target =
                class_weight   * class_prototype
                + subject_weight * subject_prototype
                + sample_weight  * sample_feature

        Expected structures:

            plain_features:
                dict[layer_name] -> Tensor[B, D]

            class_prototypes:
                dict[layer_name] -> Tensor[num_classes, D]

            subject_prototypes:
                dict[layer_name][subject_id] -> Tensor[D]
        """
        weights = torch.tensor(
            [
                float(class_weight),
                float(subject_weight),
                float(sample_weight),
            ],
            dtype=torch.float32,
        )

        if torch.any(weights < 0):
            raise ValueError(
                "Hierarchical weights must be non-negative, but got "
                f"class_weight={class_weight}, "
                f"subject_weight={subject_weight}, "
                f"sample_weight={sample_weight}."
            )

        weight_sum = float(weights.sum().item())

        if weight_sum <= 0.0:
            raise ValueError(
                "At least one hierarchical weight must be positive."
            )

        # Normalize the coefficients so they sum to 1.
        class_weight = float(class_weight) / weight_sum
        subject_weight = float(subject_weight) / weight_sum
        sample_weight = float(sample_weight) / weight_sum

        labels = labels.long()

        subject_ids = [
            str(subject_id)
            for subject_id in subject_ids
        ]

        hierarchical_targets = {}

        for layer_name, plain_feature in plain_features.items():
            if layer_name not in class_prototypes:
                raise KeyError(
                    f"Layer '{layer_name}' is missing from class_prototypes."
                )

            if layer_name not in subject_prototypes:
                raise KeyError(
                    f"Layer '{layer_name}' is missing from subject_prototypes."
                )

            sample_component = F.normalize(
                plain_feature,
                p=2,
                dim=1,
                eps=1e-8,
            )

            class_component = class_prototypes[
                layer_name
            ].to(
                device=plain_feature.device,
                dtype=plain_feature.dtype,
            )[labels]

            class_component = F.normalize(
                class_component,
                p=2,
                dim=1,
                eps=1e-8,
            )

            subject_component_list = []

            for subject_id in subject_ids:
                if (
                    subject_id
                    not in subject_prototypes[layer_name]
                ):
                    raise KeyError(
                        f"Subject '{subject_id}' is missing from "
                        f"subject_prototypes['{layer_name}']."
                    )

                subject_component_list.append(
                    subject_prototypes[
                        layer_name
                    ][subject_id].to(
                        device=plain_feature.device,
                        dtype=plain_feature.dtype,
                    )
                )

            subject_component = torch.stack(
                subject_component_list,
                dim=0,
            )

            subject_component = F.normalize(
                subject_component,
                p=2,
                dim=1,
                eps=1e-8,
            )

            target = (
                class_weight * class_component
                + subject_weight * subject_component
                + sample_weight * sample_component
            )

            hierarchical_targets[layer_name] = F.normalize(
                target,
                p=2,
                dim=1,
                eps=1e-8,
            ).detach()

        return hierarchical_targets


    def _class_subspace_loss(
        self,
        nae_features,
        labels,
        class_subspaces,
        layer_weights=None,
    ):
        """
        Penalize the distance between each NAE feature and the affine
        PCA subspace of its target class.
        """
        if layer_weights is None:
            layer_weights = {
                "block_12": 0.25,
                "block_18": 0.30,
                "block_24": 0.15,
                "final": 0.30,
            }

        total_loss = torch.zeros(
            (),
            device=labels.device,
            dtype=next(iter(nae_features.values())).dtype,
        )

        statistics = {}

        for name, feature in nae_features.items():
            if name not in class_subspaces:
                continue

            normalized_feature = F.normalize(
                feature,
                dim=1,
            )

            sample_losses = []

            for sample_index in range(
                normalized_feature.shape[0]
            ):
                class_index = int(
                    labels[sample_index].item()
                )

                subspace = class_subspaces[
                    name
                ][class_index]

                mean = subspace["mean"].to(
                    device=feature.device,
                    dtype=feature.dtype,
                )

                basis = subspace["basis"].to(
                    device=feature.device,
                    dtype=feature.dtype,
                )

                centered = (
                    normalized_feature[sample_index]
                    - mean
                )

                if basis.shape[1] > 0:
                    projection = (
                        basis
                        @ (
                            basis.transpose(0, 1)
                            @ centered
                        )
                    )
                else:
                    projection = torch.zeros_like(
                        centered
                    )

                residual = centered - projection

                sample_loss = residual.pow(2).mean()
                sample_losses.append(sample_loss)

            layer_loss = torch.stack(
                sample_losses
            ).mean()

            weight = float(
                layer_weights.get(name, 1.0)
            )

            total_loss = (
                total_loss
                + weight * layer_loss
            )

            statistics[name] = {
                "subspace_loss": layer_loss.detach(),
            }

        return total_loss, statistics

    @torch.no_grad()
    def build_class_prototypes(
        self,
        data_loader = None,
        selected_blocks=(11, 17, 23),
        num_classes=2,
    ):
        """
        Build class prototypes from plaintext images.

        The images returned by data_loader are assumed to have already
        undergone ImageNet normalization.

        Returns:
            prototypes = {
                "block_12": Tensor[num_classes, feature_dim],
                "block_18": Tensor[num_classes, feature_dim],
                "block_24": Tensor[num_classes, feature_dim],
                "final": Tensor[num_classes, feature_dim],
            }
        """
        self.model.eval()

        feature_sums = {}
        class_counts = torch.zeros(
            num_classes,
            device=self.device,
            dtype=torch.long,
        )



        for batch in data_loader:
            if len(batch) == 4:
                images, labels, _, _ = batch
            elif len(batch) == 3:
                images, labels, _ = batch

            elif len(batch) == 2:
                images, labels = batch
            else:
                raise ValueError(
                    "build_class_prototypes expects each batch to contain "
                    "either (images, labels) or "
                    "(images, labels, subject_ids, paths), "
                    f"but received {len(batch)} elements."
                )
            
            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            ).long()

            features = self._extract_vit_features(
                images,
                selected_blocks=selected_blocks,
            )

            # Normalize each sample before averaging.
            features = {
                name: F.normalize(value, dim=1)
                for name, value in features.items()
            }

            if not feature_sums:
                for name, value in features.items():
                    feature_sums[name] = torch.zeros(
                        num_classes,
                        value.shape[1],
                        device=self.device,
                        dtype=value.dtype,
                    )

            for class_index in range(num_classes):
                mask = labels == class_index

                if not mask.any():
                    continue

                class_counts[class_index] += mask.sum()

                for name, value in features.items():
                    feature_sums[name][class_index] += (
                        value[mask].sum(dim=0)
                    )

        if torch.any(class_counts == 0):
            missing = torch.where(class_counts == 0)[0].tolist()
            raise ValueError(
                f"No samples were found for classes: {missing}"
            )

        prototypes = {}

        for name, feature_sum in feature_sums.items():
            prototype = feature_sum / class_counts.unsqueeze(1)

            # Normalize again after averaging.
            prototypes[name] = F.normalize(
                prototype,
                dim=1,
            ).detach()

        return prototypes



    def _prototype_alignment_loss(
        self,
        nae_features,
        labels,
        class_prototypes,
        layer_weights=None,
        margin_weight=0.0,
        margin=0.2,
    ):
        """
        Align each NAE feature with the prototype of its class.

        Optionally adds an inter-class margin loss to avoid ambiguous
        representations.

        Args:
            nae_features:
                Dict[str, Tensor[B, D]].

            labels:
                Tensor[B].

            class_prototypes:
                Dict[str, Tensor[num_classes, D]].

            layer_weights:
                Dict[str, float].

            margin_weight:
                Weight for the inter-class margin term.

            margin:
                Desired cosine-similarity margin between the correct and
                incorrect class prototypes.

        Returns:
            total_loss
            statistics
        """
        if layer_weights is None:
            layer_weights = {
                "block_12": 0.25,
                "block_18": 0.30,
                "block_24": 0.15,
                "final": 0.30,
            }

        labels = labels.long()

        total_loss = torch.zeros(
            (),
            device=labels.device,
            dtype=next(iter(nae_features.values())).dtype,
        )

        statistics = {}

        for name, nae_feature in nae_features.items():
            if name not in class_prototypes:
                continue

            weight = float(layer_weights.get(name, 1.0))

            nae_normalized = F.normalize(
                nae_feature,
                dim=1,
            )

            prototypes = class_prototypes[name].to(
                device=nae_feature.device,
                dtype=nae_feature.dtype,
            )

            target_prototypes = prototypes[labels]

# 建议加入随机化，避免同类 NAE 完全坍缩

# 纯 prototype alignment 会让同类 NAE 都趋向同一个特征点。这有利于隐私，但可能让生成数据过于单一。

# 可以对 prototype 加一个小的随机扰动：

            # target_prototypes = prototypes[labels]

            # prototype_noise_std = 0.03

            # if prototype_noise_std > 0:
            #     target_prototypes = target_prototypes + (
            #         prototype_noise_std
            #         * torch.randn_like(target_prototypes)
            #     )

            #     target_prototypes = F.normalize(
            #         target_prototypes,
            #         dim=1,
            #     )
                
            target_cosine = F.cosine_similarity(
                nae_normalized,
                target_prototypes,
                dim=1,
            )

            prototype_loss = (
                1.0 - target_cosine
            ).mean()

            layer_loss = prototype_loss

            margin_loss = torch.zeros_like(
                prototype_loss
            )

            # General implementation that also supports more than two classes.
            if margin_weight > 0.0:
                all_cosine = torch.matmul(
                    nae_normalized,
                    prototypes.t(),
                )

                target_scores = all_cosine.gather(
                    1,
                    labels.view(-1, 1),
                ).squeeze(1)

                incorrect_mask = F.one_hot(
                    labels,
                    num_classes=prototypes.shape[0],
                ).bool()

                incorrect_scores = all_cosine.masked_fill(
                    incorrect_mask,
                    float("-inf"),
                )

                max_incorrect_score = incorrect_scores.max(
                    dim=1
                ).values

                margin_loss = F.relu(
                    margin
                    + max_incorrect_score
                    - target_scores
                ).mean()

                layer_loss = (
                    prototype_loss
                    + margin_weight * margin_loss
                )

            total_loss = (
                total_loss
                + weight * layer_loss
            )

            statistics[name] = {
                "prototype_loss": prototype_loss.detach(),
                "target_cosine": target_cosine.mean().detach(),
                "margin_loss": margin_loss.detach(),
            }

        return total_loss, statistics

    def _unwrap_model(self):
        """Return the underlying model when DDP/DataParallel is used."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model


    def _extract_vit_features(
        self,
        images,
        selected_blocks=(11, 17, 23),
    ):
        """
        Extract mean-pooled patch-token features from selected ViT blocks.

        Args:
            images:
                Input tensor [B, 3, H, W].
            selected_blocks:
                Zero-based block indices.
                (11, 17, 23) means blocks 12, 18, and 24.

        Returns:
            Dict[str, Tensor], where each feature has shape [B, 1024].
        """
        model = self._unwrap_model()

        batch_size = images.shape[0]

        tokens = model.patch_embed(images)

        cls_tokens = model.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        tokens = torch.cat(
            (cls_tokens, tokens),
            dim=1,
        )

        tokens = tokens + model.pos_embed
        tokens = model.pos_drop(tokens)

        features = {}

        for block_index, block in enumerate(model.blocks):
            tokens = block(tokens)

            if block_index in selected_blocks:
                # Exclude the CLS token and average all patch tokens.
                pooled_feature = tokens[:, 1:, :].mean(dim=1)

                features[f"block_{block_index + 1}"] = (
                    pooled_feature
                )

        if model.global_pool:
            pooled = tokens[:, 1:, :].mean(dim=1)
            final_feature = model.fc_norm(pooled)
        else:
            normalized_tokens = model.norm(tokens)
            final_feature = normalized_tokens[:, 0]

        features["final"] = final_feature

        return features


    def _multilayer_feature_loss(
        self,
        nae_features,
        plain_features,
        layer_weights,
        norm_weight=0.0,
    ):
        """
        Match NAE and plaintext features.

        The primary loss is cosine distance. An optional relative feature-norm
        loss can be added to prevent large magnitude discrepancies.
        """
        total_loss = torch.zeros(
            (),
            device=self.device,
        )

        layer_losses = {}

        for layer_name, weight in layer_weights.items():
            nae_feature = nae_features[layer_name]
            plain_feature = plain_features[layer_name].detach()

            cosine_similarity = F.cosine_similarity(
                nae_feature,
                plain_feature,
                dim=1,
                eps=1e-8,
            )

            cosine_loss = (
                1.0 - cosine_similarity
            ).mean()

            nae_norm = torch.linalg.vector_norm(
                nae_feature,
                ord=2,
                dim=1,
            )

            plain_norm = torch.linalg.vector_norm(
                plain_feature,
                ord=2,
                dim=1,
            )

            relative_norm_loss = torch.abs(
                nae_norm / (plain_norm + 1e-8) - 1.0
            ).mean()

            current_loss = (
                cosine_loss
                + norm_weight * relative_norm_loss
            )

            total_loss = (
                total_loss
                + weight * current_loss
            )

            layer_losses[layer_name] = {
                "cosine_loss": cosine_loss.detach(),
                "norm_loss": relative_norm_loss.detach(),
            }

        return total_loss, layer_losses


    def _compute_feature_metrics(
        self,
        nae_features,
        plain_features,
    ):
        metrics = {}

        for layer_name in plain_features:
            nae_feature = nae_features[layer_name].detach()
            plain_feature = plain_features[layer_name].detach()

            cosine = F.cosine_similarity(
                nae_feature,
                plain_feature,
                dim=1,
                eps=1e-8,
            )

            nae_normalized = F.normalize(
                nae_feature,
                p=2,
                dim=1,
                eps=1e-8,
            )

            plain_normalized = F.normalize(
                plain_feature,
                p=2,
                dim=1,
                eps=1e-8,
            )

            normalized_l2 = torch.linalg.vector_norm(
                nae_normalized - plain_normalized,
                ord=2,
                dim=1,
            )

            raw_l2 = torch.linalg.vector_norm(
                nae_feature - plain_feature,
                ord=2,
                dim=1,
            )

            raw_l1 = torch.mean(
                torch.abs(
                    nae_feature - plain_feature
                ),
                dim=1,
            )

            metrics[layer_name] = {
                "cosine_sum": cosine.sum().item(),
                "normalized_l2_sum": normalized_l2.sum().item(),
                "raw_l2_sum": raw_l2.sum().item(),
                "raw_l1_sum": raw_l1.sum().item(),
                "count": nae_feature.shape[0],
            }

        return metrics

    def _should_save_exposed_gradients(self, batch_idx: int, saved_this_epoch: int) -> bool:
        if not self.save_exposed_gradients:
            return False
        if self.gradient_save_max_total is not None:
            if self._gradient_saves_total >= self.gradient_save_max_total:
                return False
        if self.gradient_save_max_batches_per_epoch >= 0:
            if saved_this_epoch >= self.gradient_save_max_batches_per_epoch:
                return False
        return batch_idx % self.gradient_save_interval == 0

    def _imagenet_normalize(self, images):
        """
        Normalize images from pixel range [0, 1] using ImageNet statistics.

        Args:
            images: Tensor of shape [B, 3, H, W] in [0, 1].

        Returns:
            ImageNet-normalized tensor.
        """
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)

        return (images - mean) / std

    def _save_exposed_gradients(
        self,
        dp_grads: List[torch.Tensor],
        dp_metadata: Dict[str, Any],
        labels: torch.Tensor,
        paths,
        config: Dict[str, Any],
        local_epoch: int,
        batch_idx: int,
        learning_rate: float,
    ) -> str:
        round_idx = int(config.get("round", self.local_round))
        filename = (
            f"client{self.client_id}_round{round_idx:04d}_"
            f"localep{local_epoch:02d}_batch{batch_idx:05d}_"
            f"step{self._global_train_step:08d}.pt"
        )
        save_path = os.path.join(self.gradient_save_dir, filename)
        grad_state = OrderedDict(
            (name, grad.detach().cpu())
            for (name, _), grad in zip(self.model.named_parameters(), dp_grads)
        )
        payload = {
            "gradient_kind": "dp_clipped_noisy_averaged_exposed_to_optimizer",
            "client_id": self.client_id,
            "round": round_idx,
            "local_epoch": local_epoch,
            "batch_idx": batch_idx,
            "global_train_step": self._global_train_step,
            "learning_rate": float(learning_rate),
            "optimizer": self.optimizer_class.__name__,
            "dp_metadata": dp_metadata,
            "labels": labels.detach().cpu(),
            "paths": list(paths),
            "gradients": grad_state,
        }
        torch.save(payload, save_path)
        self._gradient_saves_total += 1
        print(
            f"[Client {self.client_id}] Saved exposed DP gradients to {save_path}"
        )
        return save_path
        
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Return model parameters to server
        支持部分权重共享（Partial Weight Sharing）
        """
        print(f"[Client {self.client_id}] Sending parameters (round {config.get('round', 'N/A')})")
        
        # 获取所有参数
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        # 部分权重共享：只返回一部分权重
        partial_fraction = config.get("partial_weight_fraction", 1.0)
        if partial_fraction < 1.0:
            num_params = len(params)
            num_share = max(1, int(num_params * partial_fraction))
            
            # 根据round重现随机选择
            np.random.seed(config.get("round", 0) + self.client_id)
            shared_indices = np.random.choice(num_params, size=num_share, replace=False)
            
            params = [params[i] for i in shared_indices]
            self.shared_param_indices = shared_indices
            print(f"[Client {self.client_id}] Sharing {num_share}/{num_params} parameters")
        
        return params
    
    def set_parameters(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> None:
        """
        Update model with parameters from server
        """
        print(f"[Client {self.client_id}] Receiving parameters")
        
        # 获取当前模型参数
        params_dict = self.model.state_dict()
        param_names = list(params_dict.keys())
        
        # 部分权重共享：需要合并参数
        partial_fraction = config.get("partial_weight_fraction", 1.0)
        if partial_fraction < 1.0 and hasattr(self, 'shared_param_indices'):
            # 只更新共享的参数
            for i, param_idx in enumerate(self.shared_param_indices):
                param_name = param_names[param_idx]
                params_dict[param_name] = torch.Tensor(parameters[i])
        else:
            # 更新所有参数
            for param_name, param in zip(param_names, parameters):
                params_dict[param_name] = torch.Tensor(param)
        
        self.model.load_state_dict(params_dict)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = False,
        strict: bool = True,
        map_location: str = None,
    ) -> Dict[str, Any]:
        """
        Load model weights from a saved checkpoint.

        Supported formats:
            1. {
                   "epoch": ...,
                   "model_state_dict": model.state_dict(),
                   ...
               }
            2. model.state_dict() directly

        Args:
            checkpoint_path: path to checkpoint file
            load_optimizer: whether to load optimizer state if available
            strict: whether to strictly enforce that checkpoint keys match model keys
            map_location: device mapping for torch.load

        Returns:
            checkpoint metadata dict
        """
        if map_location is None:
            map_location = self.device

        print(
            f"[Client {self.client_id}] Loading checkpoint from: {checkpoint_path}"
        )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=map_location,
            )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            metadata = {
                k: v for k, v in checkpoint.items()
                if k not in ["model_state_dict", "optimizer_state_dict"]
            }
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            # Treat it as a raw state_dict
            state_dict = checkpoint["model"]
            metadata = {
                k: v for k, v in checkpoint.items()
                if k not in ["model",  "optimizer", "epoch", "scaler", "args"]
            }
            metadata = {}

        elif isinstance(checkpoint, dict):
            # Treat it as a raw state_dict
            state_dict = checkpoint
            metadata = {}

        else:
            raise RuntimeError(
                f"Unsupported checkpoint format: {type(checkpoint)}"
            )

        self.model.load_state_dict(state_dict, strict=strict)
        self.model.to(self.device)
        if load_optimizer:
            if (
                isinstance(checkpoint, dict)
                and "optimizer_state_dict" in checkpoint
            ):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print(
                    f"[Client {self.client_id}] Optimizer state loaded."
                )
            elif (
                isinstance(checkpoint, dict)
                and "optimizer" in checkpoint
            ):
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print(
                    f"[Client {self.client_id}] Optimizer state loaded from key 'optimizer'."
                )
            else:
                logger.warning(
                    f"[Client {self.client_id}] load_optimizer=True, "
                    f"but no optimizer state found in checkpoint."
                )

        if "epoch" in metadata:
            print(
                f"[Client {self.client_id}] Loaded checkpoint epoch: {metadata['epoch']}"
            )

        if "val_auroc" in metadata:
            print(
                f"[Client {self.client_id}] Loaded checkpoint val_auroc: "
                f"{metadata['val_auroc']}"
            )

        if "best_val_auroc" in metadata:
            print(
                f"[Client {self.client_id}] Loaded checkpoint best_val_auroc: "
                f"{metadata['best_val_auroc']}"
            )

        return metadata

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train model locally.

        Additionally records the image paths that actually participate
        in training, for later MIA softmax generation.

        Args:
            parameters: 服务器发送的模型参数
            config: 配置字典

        Returns:
            (updated_parameters, num_examples, metrics_dict)
        """
        print(
            f"[Client {self.client_id}] "
            f"Starting training round"
        )

        # =========================================================
        # Optional: record paths of samples participating in training
        # =========================================================
        record_train_paths = bool(
            config.get(
                "record_train_paths",
                True,
            )
        )

        train_paths_output = Path(
            config.get(
                "train_paths_output",
                (
                    f"./train_paths/"
                    f"client_{self.client_id}_"
                    f"round_{self.local_round:04d}.txt"
                ),
            )
        )
        print(f"--------------------save training image paths to {train_paths_output}")
        # Use a set to avoid duplicates when local_epochs > 1.
        train_paths_seen = set()

        # ---------------------------------------------------------
        # 更新参数
        # ---------------------------------------------------------
        self.set_parameters(
            parameters,
            config,
        )

        self.model.train()

        # ---------------------------------------------------------
        # 更新学习率
        # ---------------------------------------------------------
        lr = float(
            config.get(
                "learning_rate",
                0.01,
            )
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        print(
            f"[Client {self.client_id}] "
            f"Learning rate: {lr:.8f}"
        )

        # ---------------------------------------------------------
        # 本地训练
        # ---------------------------------------------------------
        num_examples = len(
            self.train_loader.dataset
        )

        losses = []

        for epoch in range(
            self.local_epochs
        ):
            epoch_loss = 0.0
            num_batches = 0
            saved_gradients_this_epoch = 0

            for batch_idx, (
                images,
                labels,
                paths,
            ) in enumerate(
                self.train_loader
            ):
                # =====================================================
                # Record training paths.
                # This is purely bookkeeping and does not affect training.
                # =====================================================
                if record_train_paths:
                    for path in paths:
                        path = Path(str(path))
                        relative_path = f"{path.parent.name}/{path.name}"
                        train_paths_seen.add(relative_path)
                        #以下是append全路径
                        # train_paths_seen.add(
                        #     str(path)
                        # )

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                self.optimizer.zero_grad()

                if self.dp_enabled:
                    # -------------------------------------------------
                    # DP-SGD 训练
                    # -------------------------------------------------

                    # 计算逐样本梯度
                    per_sample_grads = (
                        self.dp_trainer.compute_per_sample_grads(
                            (images, labels),
                            self.criterion,
                        )
                    )

                    # 应用DP（剪切+加噪）
                    dp_grads = (
                        self.dp_trainer.apply_dp_gradient_update(
                            per_sample_grads,
                            add_noise=True,
                            return_metadata=True,
                        )
                    )

                    dp_grads, dp_metadata = (
                        dp_grads
                    )

                    # DP 后梯度先暴露给 optimizer，
                    # Adam/AdamW/SGD 再基于这些梯度更新。
                    with torch.no_grad():
                        for param, grad in zip(
                            self.model.parameters(),
                            dp_grads,
                        ):
                            param.grad = (
                                grad.detach().clone()
                            )

                    if self._should_save_exposed_gradients(
                        batch_idx=batch_idx,
                        saved_this_epoch=(
                            saved_gradients_this_epoch
                        ),
                    ):
                        self._save_exposed_gradients(
                            dp_grads=dp_grads,
                            dp_metadata=dp_metadata,
                            labels=labels,
                            paths=paths,
                            config=config,
                            local_epoch=epoch,
                            batch_idx=batch_idx,
                            learning_rate=lr,
                        )

                        saved_gradients_this_epoch += 1

                    # 前向传播获得loss（用于logging）
                    with torch.no_grad():
                        outputs = self.model(
                            images
                        )

                        loss = self.criterion(
                            outputs,
                            labels,
                        )

                else:
                    # -------------------------------------------------
                    # 标准SGD训练
                    # -------------------------------------------------
                    outputs = self.model(
                        images
                    )

                    loss = self.criterion(
                        outputs,
                        labels,
                    )

                    loss.backward()

                self.optimizer.step()

                self._global_train_step += 1

                epoch_loss += (
                    loss.item()
                )

                num_batches += 1

                if batch_idx % 10 == 0:
                    print(
                        f"[Client {self.client_id}] "
                        f"Epoch {epoch}, "
                        f"Batch {batch_idx}, "
                        f"Loss: {loss.item():.6f}"
                    )

            avg_loss = (
                epoch_loss / num_batches
                if num_batches > 0
                else 0.0
            )

            losses.append(
                avg_loss
            )

            print(
                f"[Client {self.client_id}] "
                f"Epoch {epoch} finished, "
                f"avg loss: {avg_loss:.6f}"
            )

        # =========================================================
        # Save unique training paths after training
        # =========================================================
        if record_train_paths:
            train_paths_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            sorted_train_paths = sorted(
                train_paths_seen
            )

            with train_paths_output.open(
                "w",
                encoding="utf-8",
            ) as file_handle:
                for path in sorted_train_paths:
                    file_handle.write(
                        path + "\n"
                    )

            print(
                f"[Client {self.client_id}] "
                f"Recorded {len(sorted_train_paths)} "
                f"unique training image paths to: "
                f"{train_paths_output}"
            )

        # ---------------------------------------------------------
        # 返回更新后的参数
        # ---------------------------------------------------------
        updated_params = (
            self.get_parameters(
                config
            )
        )

        # ---------------------------------------------------------
        # 收集指标
        # ---------------------------------------------------------
        metrics = {
            "train_loss": float(
                np.mean(losses)
            ),
            "num_examples": (
                num_examples
            ),
            "client_id": (
                self.client_id
            ),
            "learning_rate": (
                lr
            ),
        }

        if record_train_paths:
            metrics[
                "num_unique_train_paths"
            ] = len(
                train_paths_seen
            )

        self.local_round += 1

        return (
            updated_params,
            num_examples,
            metrics,
        )
     

    def evaluate(
        self,
        parameters,
        config,
    ):
        print(f"[Client {self.client_id}] Starting evaluation")

        self.set_parameters(parameters, config)

        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0

        all_labels = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for images, labels, paths in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total += batch_size

                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                correct += preds.eq(labels).sum().item()

                all_labels.extend(labels.detach().cpu().numpy().tolist())
                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = np.array(all_preds)

        if len(np.unique(y_true)) == 2:
            val_auroc = float(roc_auc_score(y_true, y_prob))
            val_auprc = float(average_precision_score(y_true, y_prob))
        else:
            val_auroc = float("nan")
            val_auprc = float("nan")

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        metrics = {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "val_sensitivity": sensitivity,
            "val_specificity": specificity,
            "val_f1": f1,
            "client_id": self.client_id,
        }

        print(
            f"[Client {self.client_id}] Evaluation - "
            f"Loss: {avg_loss:.6f}, "
            f"Acc: {accuracy:.4f}, "
            f"AUROC: {val_auroc:.4f}, "
            f"AUPRC: {val_auprc:.4f}, "
            f"Sens: {sensitivity:.4f}, "
            f"Spec: {specificity:.4f}, "
            f"F1: {f1:.4f}"
        )

        return avg_loss, total, metrics
        

    def _imagenet_denormalize(self, images):
        """
        Convert ImageNet-normalized images back to RGB pixel space [0, 1].

        Args:
            images: Tensor [B, 3, H, W] in normalized space.

        Returns:
            Tensor [B, 3, H, W] in RGB pixel space.
        """
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)

        return images * std + mean


    def _subject_alignment_loss(
        self,
        nae_features,
        subject_ids,
        subject_prototypes,
        layer_weights=None,
    ):
        """
        Align each NAE feature with the prototype of its subject.

        Args:
            nae_features:
                dict[layer_name] -> Tensor[B, D]

            subject_ids:
                List[str] or Tuple[str] of length B.

            subject_prototypes:
                dict[layer_name][subject_id] -> Tensor[D]

            layer_weights:
                Weight assigned to each selected layer.

        Returns:
            total_loss
            statistics
        """
        if layer_weights is None:
            layer_weights = {
                "block_12": 0.25,
                "block_18": 0.30,
                "block_24": 0.15,
                "final": 0.30,
            }

        subject_ids = [
            str(subject_id)
            for subject_id in subject_ids
        ]

        first_feature = next(iter(nae_features.values()))

        total_loss = torch.zeros(
            (),
            device=first_feature.device,
            dtype=first_feature.dtype,
        )

        statistics = {}

        for layer_name, nae_feature in nae_features.items():
            if layer_name not in subject_prototypes:
                continue

            weight = float(
                layer_weights.get(layer_name, 0.0)
            )

            if weight == 0.0:
                continue

            nae_normalized = F.normalize(
                nae_feature,
                p=2,
                dim=1,
                eps=1e-8,
            )

            target_list = []

            for subject_id in subject_ids:
                if (
                    subject_id
                    not in subject_prototypes[layer_name]
                ):
                    raise KeyError(
                        f"Subject '{subject_id}' was not found in "
                        f"subject prototypes for layer '{layer_name}'."
                    )

                target_list.append(
                    subject_prototypes[
                        layer_name
                    ][subject_id].to(
                        device=nae_feature.device,
                        dtype=nae_feature.dtype,
                    )
                )

            target_prototypes = torch.stack(
                target_list,
                dim=0,
            )

            target_prototypes = F.normalize(
                target_prototypes,
                p=2,
                dim=1,
                eps=1e-8,
            )

            target_cosine = F.cosine_similarity(
                nae_normalized,
                target_prototypes,
                dim=1,
                eps=1e-8,
            )

            alignment_loss = (
                1.0 - target_cosine
            ).mean()

            total_loss = (
                total_loss
                + weight * alignment_loss
            )

            statistics[layer_name] = {
                "subject_loss": alignment_loss.detach(),
                "target_cosine": target_cosine.mean().detach(),
                "target_cosine_min": target_cosine.min().detach(),
                "target_cosine_max": target_cosine.max().detach(),
            }

        if not statistics:
            raise RuntimeError(
                "No matching layers were found between nae_features and "
                "subject_prototypes."
            )

        return total_loss, statistics


    def _target_feature_alignment_loss_pixcons(
        self,
        nae_features,
        target_features,
        nae_image,
        noise_anchor,
        layer_weights=None,
        noise_pixel_weight=0.00000001,
    ):
        """
        Align NAE features with preconstructed target features,
        while constraining the optimized NAE to remain close to
        its original random-noise initialization.

        Total loss:

            L_total
                =
                L_feature
                +
                noise_pixel_weight * L_noise

        where:

            L_feature
                =
                sum_l w_l *
                [1 - cos(F_l(x_NAE), t_l)]

        and:

            L_noise
                =
                mean(
                    (x_NAE - x_noise)^2
                )

        Args:
            nae_features:
                dict[layer_name] -> Tensor[B, D]
                Multi-layer features extracted from the current
                optimized NAE.

            target_features:
                dict[layer_name] -> Tensor[B, D]
                Preconstructed subject-aware/sample-specific
                target representations.

            nae_image:
                Tensor[B, C, H, W]
                Current optimized NAE image.

            noise_anchor:
                Tensor[B, C, H, W]
                The ORIGINAL uniform random noise used to
                initialize nae_image.

                IMPORTANT:
                This tensor should remain fixed throughout
                optimization and should not require gradients.

            layer_weights:
                Optional layer weighting dictionary.

            noise_pixel_weight:
                Weight lambda_noise controlling the strength
                of the noise-anchor constraint.

                0.0:
                    Original feature-alignment objective only.

                Larger values:
                    Keep the optimized NAE closer to its
                    initial random noise.

        Returns:
            total_loss:
                Scalar total loss.

            statistics:
                Dictionary containing feature-alignment
                and noise-distance statistics.
        """

        if layer_weights is None:
            layer_weights = {
                "block_12": 0.25,
                "block_18": 0.30,
                "block_24": 0.15,
                "final": 0.30,
            }

        noise_pixel_weight = float(
            noise_pixel_weight
        )

        if noise_pixel_weight < 0.0:
            raise ValueError(
                "noise_pixel_weight must be non-negative, "
                f"but got {noise_pixel_weight}."
            )

        # ========================================================
        # 1. Feature alignment loss
        # ========================================================

        first_feature = next(
            iter(nae_features.values())
        )

        feature_alignment_loss = torch.zeros(
            (),
            device=first_feature.device,
            dtype=first_feature.dtype,
        )

        statistics = {}

        for layer_name, nae_feature in (
            nae_features.items()
        ):

            if layer_name not in target_features:
                continue

            weight = float(
                layer_weights.get(
                    layer_name,
                    0.0,
                )
            )

            if weight == 0.0:
                continue

            # ----------------------------------------------------
            # Normalize current NAE features
            # ----------------------------------------------------

            nae_normalized = F.normalize(
                nae_feature,
                p=2,
                dim=1,
                eps=1e-8,
            )

            # ----------------------------------------------------
            # Normalize target features
            # ----------------------------------------------------

            target_normalized = F.normalize(
                target_features[
                    layer_name
                ].to(
                    device=nae_feature.device,
                    dtype=nae_feature.dtype,
                ),
                p=2,
                dim=1,
                eps=1e-8,
            )

            # ----------------------------------------------------
            # Cosine similarity
            # ----------------------------------------------------

            target_cosine = (
                F.cosine_similarity(
                    nae_normalized,
                    target_normalized,
                    dim=1,
                    eps=1e-8,
                )
            )

            layer_loss = (
                1.0
                - target_cosine
            ).mean()

            feature_alignment_loss = (
                feature_alignment_loss
                + weight * layer_loss
            )

            statistics[
                layer_name
            ] = {
                "alignment_loss": (
                    layer_loss.detach()
                ),
                "target_cosine": (
                    target_cosine
                    .mean()
                    .detach()
                ),
                "target_cosine_min": (
                    target_cosine
                    .min()
                    .detach()
                ),
                "target_cosine_max": (
                    target_cosine
                    .max()
                    .detach()
                ),
            }

        if not statistics:
            raise RuntimeError(
                "No matching layers were found between "
                "nae_features and target_features."
            )

        # ========================================================
        # 2. Noise-anchor pixel constraint
        # ========================================================

        if nae_image.shape != noise_anchor.shape:
            raise ValueError(
                "nae_image and noise_anchor must have the "
                "same shape, but got "
                f"{tuple(nae_image.shape)} and "
                f"{tuple(noise_anchor.shape)}."
            )

        noise_anchor = noise_anchor.to(
            device=nae_image.device,
            dtype=nae_image.dtype,
        ).detach()

        # --------------------------------------------------------
        # Per-pixel squared L2 distance:
        #
        #     1/d * ||x_NAE - x_noise||_2^2
        #
        # Equivalent to pixel-wise MSE.
        #
        # Using mean rather than sum prevents the magnitude
        # from depending directly on image resolution.
        # --------------------------------------------------------

        noise_pixel_loss = F.mse_loss(
            nae_image,
            noise_anchor,
            reduction="mean",
        )

        # ========================================================
        # 3. Total loss
        # ========================================================

        total_loss = (
            feature_alignment_loss
            + noise_pixel_weight
            * noise_pixel_loss
        )

        # ========================================================
        # 4. Additional diagnostics
        # ========================================================

        with torch.no_grad():

            # Mean absolute pixel deviation
            noise_pixel_l1 = (
                nae_image
                - noise_anchor
            ).abs().mean()

            # Per-sample RMS pixel deviation
            #
            # sqrt(mean((x - x0)^2))
            #
            noise_pixel_rmse = torch.sqrt(
                (
                    nae_image
                    - noise_anchor
                )
                .pow(2)
                .flatten(1)
                .mean(dim=1)
                + 1e-12
            ).mean()

        statistics[
            "noise_anchor"
        ] = {
            "pixel_mse": (
                noise_pixel_loss
                .detach()
            ),
            "pixel_l1": (
                noise_pixel_l1
                .detach()
            ),
            "pixel_rmse": (
                noise_pixel_rmse
                .detach()
            ),
            "weight": (
                noise_pixel_weight
            ),
        }

        statistics[
            "total"
        ] = {
            "feature_alignment_loss": (
                feature_alignment_loss
                .detach()
            ),
            "noise_pixel_loss": (
                noise_pixel_loss
                .detach()
            ),
            "weighted_noise_pixel_loss": (
                (
                    noise_pixel_weight
                    * noise_pixel_loss
                ).detach()
            ),
            "total_loss": (
                total_loss
                .detach()
            ),
        }

        return (
            total_loss,
            statistics,
        )

    def _target_feature_alignment_loss(
        self,
        nae_features,
        target_features,
        layer_weights=None,
    ):
        """
        Align NAE features with preconstructed target features.
        """
        if layer_weights is None:
            layer_weights = {
                "block_12": 0.25,
                "block_18": 0.30,
                "block_24": 0.15,
                "final": 0.30,
            }

        first_feature = next(
            iter(nae_features.values())
        )

        total_loss = torch.zeros(
            (),
            device=first_feature.device,
            dtype=first_feature.dtype,
        )

        statistics = {}

        for layer_name, nae_feature in (
            nae_features.items()
        ):
            if layer_name not in target_features:
                continue

            weight = float(
                layer_weights.get(
                    layer_name,
                    0.0,
                )
            )

            if weight == 0.0:
                continue

            nae_normalized = F.normalize(
                nae_feature,
                p=2,
                dim=1,
                eps=1e-8,
            )

            target_normalized = F.normalize(
                target_features[layer_name].to(
                    device=nae_feature.device,
                    dtype=nae_feature.dtype,
                ),
                p=2,
                dim=1,
                eps=1e-8,
            )

            target_cosine = (
                F.cosine_similarity(
                    nae_normalized,
                    target_normalized,
                    dim=1,
                    eps=1e-8,
                )
            )

            layer_loss = (
                1.0 - target_cosine
            ).mean()

            total_loss = (
                total_loss
                + weight * layer_loss
            )

            statistics[layer_name] = {
                "alignment_loss": (
                    layer_loss.detach()
                ),
                "target_cosine": (
                    target_cosine.mean().detach()
                ),
                "target_cosine_min": (
                    target_cosine.min().detach()
                ),
                "target_cosine_max": (
                    target_cosine.max().detach()
                ),
            }

        if not statistics:
            raise RuntimeError(
                "No matching layers were found between "
                "nae_features and target_features."
            )

        return total_loss, statistics

    def _build_subject_sample_targets(
        self,
        plain_features,
        subject_ids,
        subject_prototypes,
        subject_weight=0.6,
        sample_weight=0.2,
    ):
        """
        Construct one subject-sample target for each plaintext sample:

            target =
                subject_weight * subject_prototype
                + sample_weight * sample_feature

        The two coefficients are normalized to sum to 1.

        No class prototype is used.

        Args:
            plain_features:
                dict[layer_name] -> Tensor[B, D]

            subject_ids:
                Patient-eye IDs of length B.

            subject_prototypes:
                dict[layer_name][subject_id] -> Tensor[D]

            subject_weight:
                Weight of the patient-eye mean prototype.

            sample_weight:
                Weight of the current sample feature.

        Returns:
            targets:
                dict[layer_name] -> Tensor[B, D]
        """
        subject_weight = float(subject_weight)
        sample_weight = float(sample_weight)

        if subject_weight < 0.0 or sample_weight < 0.0:
            raise ValueError(
                "Subject and sample weights must be non-negative, "
                f"but got subject_weight={subject_weight}, "
                f"sample_weight={sample_weight}."
            )

        weight_sum = subject_weight + sample_weight

        if weight_sum <= 0.0:
            raise ValueError(
                "At least one of subject_weight and sample_weight "
                "must be positive."
            )

        subject_weight /= weight_sum
        sample_weight /= weight_sum

        subject_ids = [
            str(subject_id)
            for subject_id in subject_ids
        ]

        targets = {}

        for layer_name, plain_feature in plain_features.items():
            if layer_name not in subject_prototypes:
                raise KeyError(
                    f"Layer '{layer_name}' is missing from "
                    "subject_prototypes."
                )

            sample_component = F.normalize(
                plain_feature,
                p=2,
                dim=1,
                eps=1e-8,
            )

            subject_component_list = []

            for subject_id in subject_ids:
                if subject_id not in subject_prototypes[layer_name]:
                    raise KeyError(
                        f"Subject '{subject_id}' is missing from "
                        f"subject_prototypes['{layer_name}']."
                    )

                subject_component_list.append(
                    subject_prototypes[
                        layer_name
                    ][subject_id].to(
                        device=plain_feature.device,
                        dtype=plain_feature.dtype,
                    )
                )

            subject_component = torch.stack(
                subject_component_list,
                dim=0,
            )

            subject_component = F.normalize(
                subject_component,
                p=2,
                dim=1,
                eps=1e-8,
            )

            target = (
                subject_weight * subject_component
                + sample_weight * sample_component
            )

            targets[layer_name] = F.normalize(
                target,
                p=2,
                dim=1,
                eps=1e-8,
            ).detach()

        return targets


    def _build_robust_subject_subspace_targets(
        self,
        plain_features,
        subject_ids,
        subject_centers,
        residual_subspaces,
        residual_weight=1.0,
    ):
        """
        Construct:

            target =
                robust_subject_center
                + residual_weight
                * projection_of_sample_residual

        where:

            residual =
                normalized_plain_feature
                - robust_subject_center

            projected_residual =
                U U^T residual

        Args:
            plain_features:
                dict[layer_name] -> Tensor[B, D]

            subject_ids:
                Sequence of patient-eye identifiers.

            subject_centers:
                dict[layer_name][subject_id] -> Tensor[D]

            residual_subspaces:
                dict[layer_name] -> Tensor[D, K]

            residual_weight:
                0.0:
                    Pure robust patient-eye center.

                1.0:
                    Preserve the complete component lying in the shared
                    residual subspace.

                Values between 0 and 1:
                    Suppress part of sample-specific variation.

        Returns:
            dict[layer_name] -> Tensor[B, D]
        """
        residual_weight = float(
            residual_weight
        )

        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError(
                "residual_weight must be in [0,1], "
                f"but got {residual_weight}."
            )

        subject_ids = [
            str(subject_id)
            for subject_id in subject_ids
        ]

        targets = {}

        for layer_name, plain_feature in (
            plain_features.items()
        ):
            if layer_name not in subject_centers:
                raise KeyError(
                    f"Missing layer '{layer_name}' "
                    "from subject_centers."
                )

            if layer_name not in residual_subspaces:
                raise KeyError(
                    f"Missing layer '{layer_name}' "
                    "from residual_subspaces."
                )

            plain_normalized = F.normalize(
                plain_feature,
                p=2,
                dim=1,
                eps=1e-8,
            )

            center_list = []

            for subject_id in subject_ids:
                if (
                    subject_id
                    not in subject_centers[layer_name]
                ):
                    raise KeyError(
                        f"Subject '{subject_id}' is missing "
                        f"from subject_centers['{layer_name}']."
                    )

                center_list.append(
                    subject_centers[
                        layer_name
                    ][subject_id].to(
                        device=plain_feature.device,
                        dtype=plain_feature.dtype,
                    )
                )

            centers = torch.stack(
                center_list,
                dim=0,
            )

            centers = F.normalize(
                centers,
                p=2,
                dim=1,
                eps=1e-8,
            )

            basis = residual_subspaces[
                layer_name
            ].to(
                device=plain_feature.device,
                dtype=plain_feature.dtype,
            )

            residual = (
                plain_normalized
                - centers
            )

            if basis.ndim != 2:
                raise ValueError(
                    f"Expected PCA basis [D,K] for "
                    f"{layer_name}, got {tuple(basis.shape)}."
                )

            if basis.shape[0] != residual.shape[1]:
                raise ValueError(
                    f"Feature dimension mismatch for {layer_name}: "
                    f"basis={basis.shape[0]}, "
                    f"feature={residual.shape[1]}."
                )

            if basis.shape[1] > 0:
                # residual [B,D]
                # basis    [D,K]
                coefficients = residual @ basis

                projected_residual = (
                    coefficients
                    @ basis.transpose(0, 1)
                )
            else:
                projected_residual = (
                    torch.zeros_like(
                        residual
                    )
                )

            target = (
                centers
                + residual_weight
                * projected_residual
            )

            targets[layer_name] = F.normalize(
                target,
                p=2,
                dim=1,
                eps=1e-8,
            ).detach()

        return targets


    def randomAttackRobustSubjectSubspace_pixcons(
        self,
        config,
    ):
        print(
            f"[Client {self.client_id}] "
            "randomAttackRobustSubjectSubspace_pixcons: Starting robust patient-eye residual-subspace attack with noise constraint..."
        )

        self.model.eval()

        # =========================================================
        # Configuration
        # =========================================================
        selected_blocks = tuple(
            config.get(
                "selected_blocks",
                (11, 17, 23),
            )
        )

        layer_weights = config.get(
            "layer_weights",
            {
                "block_12": 0.25,
                "block_18": 0.30,
                "block_24": 0.15,
                "final": 0.30,
            },
        )

        attack_steps = int(
            config.get("attack_steps", 500)
        )

        attack_lr = float(
            config.get("attack_lr", 0.05)
        )

        residual_weight = float(
            config.get("residual_weight", 1.0)
        )

        pca_rank = int(
            config.get("pca_rank", 64)
        )

        save_adv = bool(
            config.get("save_adv", True)
        )

        save_all_adv = bool(
            config.get("save_all_adv", True)
        )

        num_save_batches = int(
            config.get("num_save_batches", 2)
        )

        num_save_per_batch = int(
            config.get("num_save_per_batch", 4)
        )

        log_interval = int(
            config.get("log_interval", 50)
        )

        # =========================================================
        # Save optimization trajectory for ONE image only
        # =========================================================
        save_optimization_process = bool(
            config.get(
                "save_optimization_process",
                False,
            )
        )
        #
        # 1 means save every optimization step.
        optimization_save_interval = int(
            config.get(
                "optimization_save_interval",
                1,
            )
        )

        if optimization_save_interval <= 0:
            raise ValueError(
                "optimization_save_interval must be >= 1."
            )

        optimization_process_dir = Path(
            config.get(
                "optimization_process_dir",
                (
                    "./robust_optimization_process/"
                    f"client_{self.client_id}"
                ),
            )
        )
        print(f"optimization_process_dir:{optimization_process_dir}")
        if save_optimization_process:
            optimization_process_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        scheduler_milestones = [
            max(
                1,
                int(attack_steps * 0.60),
            ),
            max(
                1,
                int(attack_steps * 0.85),
            ),
        ]

        scheduler_gamma = float(
            config.get("scheduler_gamma", 0.3)
        )

        adv_save_dir = Path(
            config.get(
                "adv_save_dir",
                (
                    "./random_robust_subject_subspace/"
                    f"client_{self.client_id}"
                ),
            )
        )

        if save_adv:
            adv_save_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        # =========================================================
        # Build/cache robust statistics
        # =========================================================
        cache_statistics = bool(
            config.get(
                "cache_robust_statistics",
                True,
            )
        )

        cache_is_valid = (
            cache_statistics
            and hasattr(
                self,
                "_cached_robust_subject_centers",
            )
            and hasattr(
                self,
                "_cached_residual_subspaces",
            )
            and hasattr(
                self,
                "_cached_robust_subject_counts",
            )
            and getattr(
                self,
                "_cached_residual_pca_rank",
                None,
            ) == pca_rank
            and getattr(
                self,
                "_cached_selected_blocks",
                None,
            ) == selected_blocks
        )

        if cache_is_valid:
            print(
                f"[Client {self.client_id}] "
                "Using cached robust patient-eye statistics."
            )

            subject_centers = (
                self._cached_robust_subject_centers
            )

            residual_subspaces = (
                self._cached_residual_subspaces
            )

            subject_counts = (
                self._cached_robust_subject_counts
            )

        else:
            # (
            #     subject_centers,
            #     residual_subspaces,
            #     subject_counts,
            # ) = self.build_robust_subject_statistics(
            #     data_loader=self.train_loader,
            #     selected_blocks=selected_blocks,
            #     pca_rank=pca_rank,
            #     geometric_median_iterations=int(
            #         config.get(
            #             "geometric_median_iterations",
            #             100,
            #         )
            #     ),
            #     geometric_median_tolerance=float(
            #         config.get(
            #             "geometric_median_tolerance",
            #             1e-5,
            #         )
            #     ),
            # )

            statistics_cache_path = Path(
                config.get(
                    "statistics_cache_path",
                    (
                        "./robust_statistics_cache/"
                        f"client_{self.client_id}_"
                        f"rank_{pca_rank}.pt"
                    ),
                )
            )

            (
                subject_centers,
                residual_subspaces,
                subject_counts,
            ) = self.build_robust_subject_statistics(
                data_loader=self.train_loader,
                selected_blocks=selected_blocks,
                pca_rank=pca_rank,
                pca_oversample=int(
                    config.get(
                        "pca_oversample",
                        8,
                    )
                ),
                geometric_median_iterations=int(
                    config.get(
                        "geometric_median_iterations",
                        100,
                    )
                ),
                geometric_median_tolerance=float(
                    config.get(
                        "geometric_median_tolerance",
                        1e-5,
                    )
                ),
                cache_path=statistics_cache_path,
                force_rebuild=bool(
                    config.get(
                        "force_rebuild_statistics",
                        False,
                    )
                ),
            )

            if cache_statistics:
                self._cached_robust_subject_centers = (
                    subject_centers
                )

                self._cached_residual_subspaces = (
                    residual_subspaces
                )

                self._cached_robust_subject_counts = (
                    subject_counts
                )

                self._cached_residual_pca_rank = (
                    pca_rank
                )

                self._cached_selected_blocks = (
                    selected_blocks
                )

        # =========================================================
        # Freeze the FFM
        # =========================================================
        original_requires_grad = {
            name: parameter.requires_grad
            for name, parameter
            in self.model.named_parameters()
        }

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        feature_totals = {
            layer_name: {
                "target_cosine_sum": 0.0,
                "plain_cosine_sum": 0.0,
                "subject_center_cosine_sum": 0.0,
                "normalized_l2_sum": 0.0,
                "raw_l2_sum": 0.0,
                "raw_l1_sum": 0.0,
                "count": 0,
            }
            for layer_name in layer_weights
        }

        total_generated = 0

        try:
            for batch_index, batch in enumerate(
                self.train_loader
            ):
                print(f"------------------batch_index:{batch_index}")
                optimization_process_saved = False
                if len(batch) != 4:
                    raise ValueError(
                        "Expected "
                        "(images, labels, subject_ids, paths), "
                        f"but received {len(batch)} elements."
                    )

                (
                    images,
                    labels,
                    subject_ids,
                    paths,
                ) = batch

                images = images.to(
                    self.device,
                    non_blocking=True,
                )

                labels = labels.to(
                    self.device,
                    non_blocking=True,
                ).long()

                subject_ids = [
                    str(subject_id)
                    for subject_id in subject_ids
                ]

                paths = [
                    str(path)
                    for path in paths
                ]

                batch_size = images.shape[0]

                # -----------------------------------------------------
                # Skip samples whose encrypted images already exist.
                # Only attack missing samples.
                # -----------------------------------------------------
                if save_adv:
                    missing_indices = []

                    for sample_index in range(batch_size):
                        label = int(
                            labels[sample_index].item()
                        )

                        class_dir = (
                            adv_save_dir
                            / f"class_{label}"
                        )

                        stem = safe_stem_from_path_or_id(
                            paths[sample_index]
                        )

                        output_path = (
                            class_dir
                            / f"{stem}.png"
                        )

                        if output_path.exists():
                            print(
                                f"[Client {self.client_id}] "
                                f"File already exists, skip attack: "
                                f"{output_path}"
                            )
                        else:
                            missing_indices.append(
                                sample_index
                            )

                    # All images in this batch already exist.
                    if len(missing_indices) == 0:
                        print(
                            f"[Client {self.client_id}] "
                            f"Batch {batch_index}: "
                            f"all {batch_size} images already exist, "
                            "skip entire batch."
                        )
                        continue

                    # Keep only samples that still need to be generated.
                    if len(missing_indices) < batch_size:
                        index_tensor = torch.tensor(
                            missing_indices,
                            device=images.device,
                            dtype=torch.long,
                        )

                        images = images.index_select(
                            0,
                            index_tensor,
                        )

                        labels = labels.index_select(
                            0,
                            index_tensor,
                        )

                        subject_ids = [
                            subject_ids[index]
                            for index in missing_indices
                        ]

                        paths = [
                            paths[index]
                            for index in missing_indices
                        ]

                        print(
                            f"[Client {self.client_id}] "
                            f"Batch {batch_index}: "
                            f"{batch_size - len(missing_indices)} already exist, "
                            f"attack only {len(missing_indices)} missing images."
                        )

                        batch_size = len(
                            missing_indices
                        )
                        
                # -----------------------------------------------------
                # Build plaintext-derived targets.
                # -----------------------------------------------------
                with torch.no_grad():
                    plain_features = (
                        self._extract_vit_features(
                            images,
                            selected_blocks=selected_blocks,
                        )
                    )

                    target_features = (
                        self._build_robust_subject_subspace_targets(
                            plain_features=plain_features,
                            subject_ids=subject_ids,
                            subject_centers=subject_centers,
                            residual_subspaces=residual_subspaces,
                            residual_weight=residual_weight,
                        )
                    )

                # -----------------------------------------------------
                # Initialize from uniform noise.
                # -----------------------------------------------------
                noise_anchor = torch.rand_like(
                    images
                )
                nae_pixel = (
                    noise_anchor
                    .clone()
                    .detach()
                )

                nae_pixel.requires_grad_(True)


                # =================================================
                # Decide whether to record this optimization process
                # =================================================
                record_this_optimization = (
                    save_optimization_process
                    and not optimization_process_saved
                    and batch_size > 0
                )

                trajectory_sample_index = 0

                if record_this_optimization:
                    trajectory_stem = (
                        safe_stem_from_path_or_id(
                            paths[
                                trajectory_sample_index
                            ]
                        )
                    )

                    trajectory_dir = (
                        optimization_process_dir
                        / trajectory_stem
                    )

                    trajectory_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    print(
                        f"[Client {self.client_id}] "
                        "Saving optimization process for ONE image:"
                    )

                    print(
                        f"  source path: "
                        f"{paths[trajectory_sample_index]}"
                    )

                    print(
                        f"  subject ID: "
                        f"{subject_ids[trajectory_sample_index]}"
                    )

                    print(
                        f"  trajectory directory: "
                        f"{trajectory_dir}"
                    )

                    # ---------------------------------------------
                    # Save x^(0): initial uniform noise
                    # ---------------------------------------------
                    save_image(
                        nae_pixel[
                            trajectory_sample_index
                        ]
                        .detach()
                        .cpu(),
                        trajectory_dir
                        / "step_0000.png",
                        normalize=False,
                    )

                optimizer = torch.optim.Adam(
                    [nae_pixel],
                    lr=attack_lr,
                )

                scheduler = (
                    torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=scheduler_milestones,
                        gamma=scheduler_gamma,
                    )
                )

                # -----------------------------------------------------
                # Feature-space attack
                # -----------------------------------------------------
                for attack_step in range(
                    attack_steps
                ):
                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    nae_input = (
                        self._imagenet_normalize(
                            nae_pixel
                        )
                    )

                    nae_features = (
                        self._extract_vit_features(
                            nae_input,
                            selected_blocks=selected_blocks,
                        )
                    )

                    (
                        attack_loss,
                        layer_statistics,
                    ) = self._target_feature_alignment_loss_pixcons(
                        nae_features=nae_features,
                        target_features=target_features,
                        nae_image=nae_pixel,
                        noise_anchor=noise_anchor,
                        layer_weights=layer_weights,
                    )



                    if not torch.isfinite(
                        attack_loss
                    ):
                        raise RuntimeError(
                            "Non-finite robust-subspace loss "
                            f"at batch {batch_index}, "
                            f"step {attack_step + 1}: "
                            f"{attack_loss.item()}"
                        )

                    attack_loss.backward()
                    optimizer.step()
                    scheduler.step()

                    with torch.no_grad():
                        nae_pixel.clamp_(
                            0.0,
                            1.0,
                        )

                    # =============================================
                    # Save optimization trajectory
                    #
                    # x^(1), x^(2), ..., x^(T)
                    # Only for ONE image in the whole experiment.
                    # =============================================
                    if record_this_optimization:
                        current_step = (
                            attack_step + 1
                        )

                        should_save_step = (
                            current_step
                            % optimization_save_interval
                            == 0
                            or current_step
                            == attack_steps
                        )

                        if should_save_step:
                            save_image(
                                nae_pixel[
                                    trajectory_sample_index
                                ]
                                .detach()
                                .cpu(),
                                trajectory_dir
                                / (
                                    f"step_"
                                    f"{current_step:04d}.png"
                                ),
                                normalize=False,
                            )

                    


                    if (
                        attack_step == 0
                        or (
                            log_interval > 0
                            and (attack_step + 1)
                            % log_interval == 0
                        )
                        or attack_step + 1 == attack_steps
                    ):
                        # ---------------------------------------------------------
                        # Feature alignment statistics
                        # ---------------------------------------------------------
                        layer_message = ", ".join(
                            (
                                f"{layer_name}: "
                                f"{statistics['target_cosine'].item():.4f}"
                            )
                            for layer_name, statistics
                            in layer_statistics.items()
                            if "target_cosine" in statistics
                        )

                        # ---------------------------------------------------------
                        # Noise-anchor statistics
                        # ---------------------------------------------------------
                        noise_statistics = (
                            layer_statistics.get(
                                "noise_anchor",
                                {}
                            )
                        )

                        total_statistics = (
                            layer_statistics.get(
                                "total",
                                {}
                            )
                        )

                        noise_mse = (
                            noise_statistics.get(
                                "pixel_mse",
                                None,
                            )
                        )

                        weighted_noise_loss = (
                            total_statistics.get(
                                "weighted_noise_pixel_loss",
                                None,
                            )
                        )

                        feature_loss = (
                            total_statistics.get(
                                "feature_alignment_loss",
                                None,
                            )
                        )

                        message = (
                            f"[Client {self.client_id}] "
                            f"Batch {batch_index}, "
                            f"Step {attack_step + 1}/"
                            f"{attack_steps}, "
                            f"Total loss: "
                            f"{attack_loss.item():.6f}"
                        )

                        if feature_loss is not None:
                            message += (
                                f", Feature loss: "
                                f"{feature_loss.item():.6f}"
                            )

                        if noise_mse is not None:
                            message += (
                                f", Noise MSE: "
                                f"{noise_mse.item():.6f}"
                            )

                        if weighted_noise_loss is not None:
                            message += (
                                f", Weighted noise loss: "
                                f"{weighted_noise_loss.item():.6f}"
                            )

                        message += (
                            f", Target cosine: "
                            f"[{layer_message}]"
                        )

                        print(message)
                        

                nae_pixel = nae_pixel.detach()
                total_generated += len(images)

                if record_this_optimization:
                    optimization_process_saved = True

                    print(
                        f"[Client {self.client_id}] "
                        "Finished saving optimization process:"
                    )

                    print(
                        f"  {trajectory_dir}"
                    )

                    print(
                        f"  saved from step 0 to "
                        f"step {attack_steps}"
                    )

                # -----------------------------------------------------
                # Save output images
                # -----------------------------------------------------
                should_save_batch = (
                    save_all_adv
                    or batch_index < num_save_batches
                )

                if save_adv and should_save_batch:
                    if save_all_adv:
                        num_to_save = len(images)
                    else:
                        num_to_save = min(
                            num_save_per_batch,
                            len(images),
                        )

                    labels_cpu = (
                        labels.detach()
                        .cpu()
                        .tolist()
                    )

                    for sample_index in range(
                        num_to_save
                    ):
                        label = int(
                            labels_cpu[sample_index]
                        )

                        class_dir = (
                            adv_save_dir
                            / f"class_{label}"
                        )

                        class_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )



                        stem = safe_stem_from_path_or_id(
                            paths[sample_index]
                        )

                        output_path = (
                            class_dir
                            / f"{stem}.png"
                        )

                        if output_path.exists():
                            print(
                                f"[Client {self.client_id}] "
                                f"File already exists before saving: "
                                f"{output_path}"
                            )
                            continue                    
                        save_image(
                            nae_pixel[
                                sample_index
                            ].cpu(),
                            output_path,
                            normalize=False,
                        )

                        print(
                            f"[Client {self.client_id}] "
                            f"Saved: {output_path}"
                        )
                            
                # -----------------------------------------------------
                # Evaluation metrics
                # -----------------------------------------------------
                with torch.no_grad():
                    final_nae_input = (
                        self._imagenet_normalize(
                            nae_pixel
                        )
                    )

                    final_nae_features = (
                        self._extract_vit_features(
                            final_nae_input,
                            selected_blocks=selected_blocks,
                        )
                    )

                    plain_metrics = (
                        self._compute_feature_metrics(
                            nae_features=final_nae_features,
                            plain_features=plain_features,
                        )
                    )

                    for layer_name in layer_weights:
                        if (
                            layer_name
                            not in final_nae_features
                        ):
                            continue

                        nae_feature = F.normalize(
                            final_nae_features[
                                layer_name
                            ],
                            p=2,
                            dim=1,
                            eps=1e-8,
                        )

                        target_feature = F.normalize(
                            target_features[
                                layer_name
                            ],
                            p=2,
                            dim=1,
                            eps=1e-8,
                        )

                        center_feature = torch.stack(
                            [
                                subject_centers[
                                    layer_name
                                ][subject_id].to(
                                    device=self.device,
                                    dtype=nae_feature.dtype,
                                )
                                for subject_id in subject_ids
                            ],
                            dim=0,
                        )

                        center_feature = F.normalize(
                            center_feature,
                            p=2,
                            dim=1,
                            eps=1e-8,
                        )

                        target_cosine = (
                            F.cosine_similarity(
                                nae_feature,
                                target_feature,
                                dim=1,
                            )
                        )

                        center_cosine = (
                            F.cosine_similarity(
                                nae_feature,
                                center_feature,
                                dim=1,
                            )
                        )

                        values = plain_metrics[
                            layer_name
                        ]

                        feature_totals[
                            layer_name
                        ]["target_cosine_sum"] += (
                            target_cosine.sum().item()
                        )

                        feature_totals[
                            layer_name
                        ]["subject_center_cosine_sum"] += (
                            center_cosine.sum().item()
                        )

                        feature_totals[
                            layer_name
                        ]["plain_cosine_sum"] += (
                            values["cosine_sum"]
                        )

                        feature_totals[
                            layer_name
                        ]["normalized_l2_sum"] += (
                            values["normalized_l2_sum"]
                        )

                        feature_totals[
                            layer_name
                        ]["raw_l2_sum"] += (
                            values["raw_l2_sum"]
                        )

                        feature_totals[
                            layer_name
                        ]["raw_l1_sum"] += (
                            values["raw_l1_sum"]
                        )

                        feature_totals[
                            layer_name
                        ]["count"] += (
                            values["count"]
                        )

        finally:
            for name, parameter in (
                self.model.named_parameters()
            ):
                parameter.requires_grad_(
                    original_requires_grad[name]
                )

        metrics = {
            "attack_type": (
                "robust_subject_residual_subspace"
            ),
            "attack_steps": attack_steps,
            "attack_lr": attack_lr,
            "residual_weight": residual_weight,
            "pca_rank": pca_rank,
            "num_subjects": len(subject_counts),
            "num_generated": total_generated,
            "client_id": self.client_id,
        }

        for layer_name, values in (
            feature_totals.items()
        ):
            count = max(
                int(values["count"]),
                1,
            )

            for metric_name in (
                "target_cosine_sum",
                "subject_center_cosine_sum",
                "plain_cosine_sum",
                "normalized_l2_sum",
                "raw_l2_sum",
                "raw_l1_sum",
            ):
                output_name = metric_name.replace(
                    "_sum",
                    "",
                )

                metrics[
                    f"{layer_name}_{output_name}"
                ] = (
                    values[metric_name]
                    / count
                )

        print(
            f"[Client {self.client_id}] "
            f"Generated {total_generated} robust-subspace NAEs "
            f"for {len(subject_counts)} patient-eye subjects."
        )

        for layer_name in layer_weights:
            print(
                f"[Client {self.client_id}] "
                f"{layer_name} - "
                f"Target cosine: "
                f"{metrics[f'{layer_name}_target_cosine']:.6f}, "
                f"Center cosine: "
                f"{metrics[f'{layer_name}_subject_center_cosine']:.6f}, "
                f"Plain cosine: "
                f"{metrics[f'{layer_name}_plain_cosine']:.6f}"
            )

        return total_generated, metrics

     
    def randomAttackRobustSubjectSubspace(
        self,
        config,
    ):
        print(
            f"[Client {self.client_id}] "
            "Starting robust patient-eye residual-subspace attack"
        )

        self.model.eval()

        # =========================================================
        # Configuration
        # =========================================================
        selected_blocks = tuple(
            config.get(
                "selected_blocks",
                (11, 17, 23),
            )
        )

        layer_weights = config.get(
            "layer_weights",
            {
                "block_12": 0.25,
                "block_18": 0.30,
                "block_24": 0.15,
                "final": 0.30,
            },
        )

        attack_steps = int(
            config.get("attack_steps", 500)
        )

        attack_lr = float(
            config.get("attack_lr", 0.05)
        )

        residual_weight = float(
            config.get("residual_weight", 1.0)
        )

        pca_rank = int(
            config.get("pca_rank", 64)
        )

        save_adv = bool(
            config.get("save_adv", True)
        )

        save_all_adv = bool(
            config.get("save_all_adv", True)
        )

        num_save_batches = int(
            config.get("num_save_batches", 2)
        )

        num_save_per_batch = int(
            config.get("num_save_per_batch", 4)
        )

        log_interval = int(
            config.get("log_interval", 50)
        )

        # =========================================================
        # Save optimization trajectory for ONE image only
        # =========================================================
        save_optimization_process = bool(
            config.get(
                "save_optimization_process",
                False,
            )
        )
        #
        # 1 means save every optimization step.
        optimization_save_interval = int(
            config.get(
                "optimization_save_interval",
                1,
            )
        )

        if optimization_save_interval <= 0:
            raise ValueError(
                "optimization_save_interval must be >= 1."
            )

        optimization_process_dir = Path(
            config.get(
                "optimization_process_dir",
                (
                    "./robust_optimization_process/"
                    f"client_{self.client_id}"
                ),
            )
        )
        print(f"optimization_process_dir:{optimization_process_dir}")
        if save_optimization_process:
            optimization_process_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        scheduler_milestones = [
            max(
                1,
                int(attack_steps * 0.60),
            ),
            max(
                1,
                int(attack_steps * 0.85),
            ),
        ]

        scheduler_gamma = float(
            config.get("scheduler_gamma", 0.3)
        )

        adv_save_dir = Path(
            config.get(
                "adv_save_dir",
                (
                    "./random_robust_subject_subspace/"
                    f"client_{self.client_id}"
                ),
            )
        )

        if save_adv:
            adv_save_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        # =========================================================
        # Build/cache robust statistics
        # =========================================================
        cache_statistics = bool(
            config.get(
                "cache_robust_statistics",
                True,
            )
        )

        cache_is_valid = (
            cache_statistics
            and hasattr(
                self,
                "_cached_robust_subject_centers",
            )
            and hasattr(
                self,
                "_cached_residual_subspaces",
            )
            and hasattr(
                self,
                "_cached_robust_subject_counts",
            )
            and getattr(
                self,
                "_cached_residual_pca_rank",
                None,
            ) == pca_rank
            and getattr(
                self,
                "_cached_selected_blocks",
                None,
            ) == selected_blocks
        )

        if cache_is_valid:
            print(
                f"[Client {self.client_id}] "
                "Using cached robust patient-eye statistics."
            )

            subject_centers = (
                self._cached_robust_subject_centers
            )

            residual_subspaces = (
                self._cached_residual_subspaces
            )

            subject_counts = (
                self._cached_robust_subject_counts
            )

        else:

            statistics_cache_path = Path(
                config.get(
                    "statistics_cache_path",
                    (
                        "./robust_statistics_cache/"
                        f"client_{self.client_id}_"
                        f"rank_{pca_rank}.pt"
                    ),
                )
            )

            (
                subject_centers,
                residual_subspaces,
                subject_counts,
            ) = self.build_robust_subject_statistics(
                data_loader=self.train_loader,
                selected_blocks=selected_blocks,
                pca_rank=pca_rank,
                pca_oversample=int(
                    config.get(
                        "pca_oversample",
                        8,
                    )
                ),
                geometric_median_iterations=int(
                    config.get(
                        "geometric_median_iterations",
                        100,
                    )
                ),
                geometric_median_tolerance=float(
                    config.get(
                        "geometric_median_tolerance",
                        1e-5,
                    )
                ),
                cache_path=statistics_cache_path,
                force_rebuild=bool(
                    config.get(
                        "force_rebuild_statistics",
                        False,
                    )
                ),
            )

            if cache_statistics:
                self._cached_robust_subject_centers = (
                    subject_centers
                )

                self._cached_residual_subspaces = (
                    residual_subspaces
                )

                self._cached_robust_subject_counts = (
                    subject_counts
                )

                self._cached_residual_pca_rank = (
                    pca_rank
                )

                self._cached_selected_blocks = (
                    selected_blocks
                )

        # =========================================================
        # Freeze the FFM
        # =========================================================
        original_requires_grad = {
            name: parameter.requires_grad
            for name, parameter
            in self.model.named_parameters()
        }

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        feature_totals = {
            layer_name: {
                "target_cosine_sum": 0.0,
                "plain_cosine_sum": 0.0,
                "subject_center_cosine_sum": 0.0,
                "normalized_l2_sum": 0.0,
                "raw_l2_sum": 0.0,
                "raw_l1_sum": 0.0,
                "count": 0,
            }
            for layer_name in layer_weights
        }

        total_generated = 0

        try:
            for batch_index, batch in enumerate(
                self.train_loader
            ):
                print(f"------------------batch_index:{batch_index}")
                optimization_process_saved = False
                if len(batch) != 4:
                    raise ValueError(
                        "Expected "
                        "(images, labels, subject_ids, paths), "
                        f"but received {len(batch)} elements."
                    )

                (
                    images,
                    labels,
                    subject_ids,
                    paths,
                ) = batch

                images = images.to(
                    self.device,
                    non_blocking=True,
                )

                labels = labels.to(
                    self.device,
                    non_blocking=True,
                ).long()

                subject_ids = [
                    str(subject_id)
                    for subject_id in subject_ids
                ]

                paths = [
                    str(path)
                    for path in paths
                ]

                batch_size = images.shape[0]

                # -----------------------------------------------------
                # Skip samples whose encrypted images already exist.
                # Only attack missing samples.
                # -----------------------------------------------------
                if save_adv:
                    missing_indices = []

                    for sample_index in range(batch_size):
                        label = int(
                            labels[sample_index].item()
                        )

                        class_dir = (
                            adv_save_dir
                            / f"class_{label}"
                        )

                        stem = safe_stem_from_path_or_id(
                            paths[sample_index]
                        )

                        output_path = (
                            class_dir
                            / f"{stem}.png"
                        )

                        if output_path.exists():
                            print(
                                f"[Client {self.client_id}] "
                                f"File already exists, skip attack: "
                                f"{output_path}"
                            )
                        else:
                            missing_indices.append(
                                sample_index
                            )

                    # All images in this batch already exist.
                    if len(missing_indices) == 0:
                        print(
                            f"[Client {self.client_id}] "
                            f"Batch {batch_index}: "
                            f"all {batch_size} images already exist, "
                            "skip entire batch."
                        )
                        continue

                    # Keep only samples that still need to be generated.
                    if len(missing_indices) < batch_size:
                        index_tensor = torch.tensor(
                            missing_indices,
                            device=images.device,
                            dtype=torch.long,
                        )

                        images = images.index_select(
                            0,
                            index_tensor,
                        )

                        labels = labels.index_select(
                            0,
                            index_tensor,
                        )

                        subject_ids = [
                            subject_ids[index]
                            for index in missing_indices
                        ]

                        paths = [
                            paths[index]
                            for index in missing_indices
                        ]

                        print(
                            f"[Client {self.client_id}] "
                            f"Batch {batch_index}: "
                            f"{batch_size - len(missing_indices)} already exist, "
                            f"attack only {len(missing_indices)} missing images."
                        )

                        batch_size = len(
                            missing_indices
                        )
                        
                # -----------------------------------------------------
                # Build plaintext-derived targets.
                # -----------------------------------------------------
                with torch.no_grad():
                    plain_features = (
                        self._extract_vit_features(
                            images,
                            selected_blocks=selected_blocks,
                        )
                    )

                    target_features = (
                        self._build_robust_subject_subspace_targets(
                            plain_features=plain_features,
                            subject_ids=subject_ids,
                            subject_centers=subject_centers,
                            residual_subspaces=residual_subspaces,
                            residual_weight=residual_weight,
                        )
                    )

                # -----------------------------------------------------
                # Initialize from uniform noise.
                # -----------------------------------------------------
                nae_pixel = torch.rand_like(
                    images
                )

                nae_pixel.requires_grad_(True)

                # =================================================
                # Decide whether to record this optimization process
                # =================================================
                record_this_optimization = (
                    save_optimization_process
                    and not optimization_process_saved
                    and batch_size > 0
                )

                trajectory_sample_index = 0

                if record_this_optimization:
                    trajectory_stem = (
                        safe_stem_from_path_or_id(
                            paths[
                                trajectory_sample_index
                            ]
                        )
                    )

                    trajectory_dir = (
                        optimization_process_dir
                        / trajectory_stem
                    )

                    trajectory_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    print(
                        f"[Client {self.client_id}] "
                        "Saving optimization process for ONE image:"
                    )

                    print(
                        f"  source path: "
                        f"{paths[trajectory_sample_index]}"
                    )

                    print(
                        f"  subject ID: "
                        f"{subject_ids[trajectory_sample_index]}"
                    )

                    print(
                        f"  trajectory directory: "
                        f"{trajectory_dir}"
                    )

                    # ---------------------------------------------
                    # Save x^(0): initial uniform noise
                    # ---------------------------------------------
                    save_image(
                        nae_pixel[
                            trajectory_sample_index
                        ]
                        .detach()
                        .cpu(),
                        trajectory_dir
                        / "step_0000.png",
                        normalize=False,
                    )

                optimizer = torch.optim.Adam(
                    [nae_pixel],
                    lr=attack_lr,
                )

                scheduler = (
                    torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=scheduler_milestones,
                        gamma=scheduler_gamma,
                    )
                )

                # -----------------------------------------------------
                # Feature-space attack
                # -----------------------------------------------------
                for attack_step in range(
                    attack_steps
                ):
                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    nae_input = (
                        self._imagenet_normalize(
                            nae_pixel
                        )
                    )

                    nae_features = (
                        self._extract_vit_features(
                            nae_input,
                            selected_blocks=selected_blocks,
                        )
                    )

                    (
                        attack_loss,
                        layer_statistics,
                    ) = self._target_feature_alignment_loss(
                        nae_features=nae_features,
                        target_features=target_features,
                        layer_weights=layer_weights,
                    )

                    if not torch.isfinite(
                        attack_loss
                    ):
                        raise RuntimeError(
                            "Non-finite robust-subspace loss "
                            f"at batch {batch_index}, "
                            f"step {attack_step + 1}: "
                            f"{attack_loss.item()}"
                        )

                    attack_loss.backward()
                    optimizer.step()
                    scheduler.step()

                    with torch.no_grad():
                        nae_pixel.clamp_(
                            0.0,
                            1.0,
                        )

                    # =============================================
                    # Save optimization trajectory
                    #
                    # x^(1), x^(2), ..., x^(T)
                    # Only for ONE image in the whole experiment.
                    # =============================================
                    if record_this_optimization:
                        current_step = (
                            attack_step + 1
                        )

                        should_save_step = (
                            current_step
                            % optimization_save_interval
                            == 0
                            or current_step
                            == attack_steps
                        )

                        if should_save_step:
                            save_image(
                                nae_pixel[
                                    trajectory_sample_index
                                ]
                                .detach()
                                .cpu(),
                                trajectory_dir
                                / (
                                    f"step_"
                                    f"{current_step:04d}.png"
                                ),
                                normalize=False,
                            )

                    if (
                        attack_step == 0
                        or (
                            log_interval > 0
                            and (attack_step + 1)
                            % log_interval == 0
                        )
                        or attack_step + 1
                        == attack_steps
                    ):
                        layer_message = ", ".join(
                            (
                                f"{layer_name}: "
                                f"{statistics['target_cosine'].item():.4f}"
                            )
                            for layer_name, statistics
                            in layer_statistics.items()
                        )

                        print(
                            f"[Client {self.client_id}] "
                            f"Batch {batch_index}, "
                            f"Step {attack_step + 1}/"
                            f"{attack_steps}, "
                            f"Loss: {attack_loss.item():.6f}, "
                            f"Target cosine: "
                            f"[{layer_message}]"
                        )

                nae_pixel = nae_pixel.detach()
                total_generated += len(images)

                if record_this_optimization:
                    optimization_process_saved = True

                    print(
                        f"[Client {self.client_id}] "
                        "Finished saving optimization process:"
                    )

                    print(
                        f"  {trajectory_dir}"
                    )

                    print(
                        f"  saved from step 0 to "
                        f"step {attack_steps}"
                    )

                # -----------------------------------------------------
                # Save output images
                # -----------------------------------------------------
                should_save_batch = (
                    save_all_adv
                    or batch_index < num_save_batches
                )

                if save_adv and should_save_batch:
                    if save_all_adv:
                        num_to_save = len(images)
                    else:
                        num_to_save = min(
                            num_save_per_batch,
                            len(images),
                        )

                    labels_cpu = (
                        labels.detach()
                        .cpu()
                        .tolist()
                    )

                    for sample_index in range(
                        num_to_save
                    ):
                        label = int(
                            labels_cpu[sample_index]
                        )

                        class_dir = (
                            adv_save_dir
                            / f"class_{label}"
                        )

                        class_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )



                        stem = safe_stem_from_path_or_id(
                            paths[sample_index]
                        )

                        output_path = (
                            class_dir
                            / f"{stem}.png"
                        )

                        if output_path.exists():
                            print(
                                f"[Client {self.client_id}] "
                                f"File already exists before saving: "
                                f"{output_path}"
                            )
                            continue                    
                        save_image(
                            nae_pixel[
                                sample_index
                            ].cpu(),
                            output_path,
                            normalize=False,
                        )

                        print(
                            f"[Client {self.client_id}] "
                            f"Saved: {output_path}"
                        )
                            
                # -----------------------------------------------------
                # Evaluation metrics
                # -----------------------------------------------------
                with torch.no_grad():
                    final_nae_input = (
                        self._imagenet_normalize(
                            nae_pixel
                        )
                    )

                    final_nae_features = (
                        self._extract_vit_features(
                            final_nae_input,
                            selected_blocks=selected_blocks,
                        )
                    )

                    plain_metrics = (
                        self._compute_feature_metrics(
                            nae_features=final_nae_features,
                            plain_features=plain_features,
                        )
                    )

                    for layer_name in layer_weights:
                        if (
                            layer_name
                            not in final_nae_features
                        ):
                            continue

                        nae_feature = F.normalize(
                            final_nae_features[
                                layer_name
                            ],
                            p=2,
                            dim=1,
                            eps=1e-8,
                        )

                        target_feature = F.normalize(
                            target_features[
                                layer_name
                            ],
                            p=2,
                            dim=1,
                            eps=1e-8,
                        )

                        center_feature = torch.stack(
                            [
                                subject_centers[
                                    layer_name
                                ][subject_id].to(
                                    device=self.device,
                                    dtype=nae_feature.dtype,
                                )
                                for subject_id in subject_ids
                            ],
                            dim=0,
                        )

                        center_feature = F.normalize(
                            center_feature,
                            p=2,
                            dim=1,
                            eps=1e-8,
                        )

                        target_cosine = (
                            F.cosine_similarity(
                                nae_feature,
                                target_feature,
                                dim=1,
                            )
                        )

                        center_cosine = (
                            F.cosine_similarity(
                                nae_feature,
                                center_feature,
                                dim=1,
                            )
                        )

                        values = plain_metrics[
                            layer_name
                        ]

                        feature_totals[
                            layer_name
                        ]["target_cosine_sum"] += (
                            target_cosine.sum().item()
                        )

                        feature_totals[
                            layer_name
                        ]["subject_center_cosine_sum"] += (
                            center_cosine.sum().item()
                        )

                        feature_totals[
                            layer_name
                        ]["plain_cosine_sum"] += (
                            values["cosine_sum"]
                        )

                        feature_totals[
                            layer_name
                        ]["normalized_l2_sum"] += (
                            values["normalized_l2_sum"]
                        )

                        feature_totals[
                            layer_name
                        ]["raw_l2_sum"] += (
                            values["raw_l2_sum"]
                        )

                        feature_totals[
                            layer_name
                        ]["raw_l1_sum"] += (
                            values["raw_l1_sum"]
                        )

                        feature_totals[
                            layer_name
                        ]["count"] += (
                            values["count"]
                        )

        finally:
            for name, parameter in (
                self.model.named_parameters()
            ):
                parameter.requires_grad_(
                    original_requires_grad[name]
                )

        metrics = {
            "attack_type": (
                "robust_subject_residual_subspace"
            ),
            "attack_steps": attack_steps,
            "attack_lr": attack_lr,
            "residual_weight": residual_weight,
            "pca_rank": pca_rank,
            "num_subjects": len(subject_counts),
            "num_generated": total_generated,
            "client_id": self.client_id,
        }

        for layer_name, values in (
            feature_totals.items()
        ):
            count = max(
                int(values["count"]),
                1,
            )

            for metric_name in (
                "target_cosine_sum",
                "subject_center_cosine_sum",
                "plain_cosine_sum",
                "normalized_l2_sum",
                "raw_l2_sum",
                "raw_l1_sum",
            ):
                output_name = metric_name.replace(
                    "_sum",
                    "",
                )

                metrics[
                    f"{layer_name}_{output_name}"
                ] = (
                    values[metric_name]
                    / count
                )

        print(
            f"[Client {self.client_id}] "
            f"Generated {total_generated} robust-subspace NAEs "
            f"for {len(subject_counts)} patient-eye subjects."
        )

        for layer_name in layer_weights:
            print(
                f"[Client {self.client_id}] "
                f"{layer_name} - "
                f"Target cosine: "
                f"{metrics[f'{layer_name}_target_cosine']:.6f}, "
                f"Center cosine: "
                f"{metrics[f'{layer_name}_subject_center_cosine']:.6f}, "
                f"Plain cosine: "
                f"{metrics[f'{layer_name}_plain_cosine']:.6f}"
            )

        return total_generated, metrics
  




    def test(
        self
    ):
        print(f"[Client {self.client_id}] Starting evaluation")


        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0

        all_labels = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for images, labels, paths in self.test_loader:
            #for images, labels, paths in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                batch_size = labels.size(0)
                total += batch_size

                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                correct += preds.eq(labels).sum().item()

                all_labels.extend(labels.detach().cpu().numpy().tolist())
                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                

        accuracy = correct / total if total > 0 else 0.0

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = np.array(all_preds)

        if len(np.unique(y_true)) == 2:
            val_auroc = float(roc_auc_score(y_true, y_prob))
            val_auprc = float(average_precision_score(y_true, y_prob))
        else:
            val_auroc = float("nan")
            val_auprc = float("nan")

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        metrics = {
            "val_accuracy": accuracy,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "val_sensitivity": sensitivity,
            "val_specificity": specificity,
            "val_f1": f1,
            "client_id": self.client_id,
        }

        print(
            f"[Client {self.client_id}] Evaluation - "
            f"Acc: {accuracy:.4f}, "
            f"AUROC: {val_auroc:.4f}, "
            f"AUPRC: {val_auprc:.4f}, "
            f"Sens: {sensitivity:.4f}, "
            f"Spec: {specificity:.4f}, "
            f"F1: {f1:.4f}"
        )

        return total, metrics



