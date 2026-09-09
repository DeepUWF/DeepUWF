import os
import sys

import json
import random
import subprocess
import gc

from torch.utils.data import  Subset

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



from pairAttack.trainRIC import DRPickleDatasetWithPath
#from pairAttack.trainRIC import MyClassifier
from dp_utils import PrivacyBudget, update_privacy_budget
from dp_fl_client import DPFLClient
from dp_fl_server import run_federated_server
import math
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

import flwr as fl
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np

from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全局设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class ImageFolderWithPath(datasets.ImageFolder):
    """
    ImageFolder that returns:
        image, label, path
    """

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        # self.samples[index] = (image_path, class_idx)
        path, _ = self.samples[index]

        return image, label, path
    
def get_or_create_reconstruction_indices(
    dataset,
    indices_file: str,
    samples_per_class: int = 1,
    seed: int = 42,
):
    """
    为每个类别固定选择若干样本。

    DRPickleDatasetWithPath.samples 的结构是：
        (image_path, label, image_id)
    """
    indices_file = Path(indices_file)
    indices_file.parent.mkdir(parents=True, exist_ok=True)

    if indices_file.exists():
        with open(indices_file, "r") as f:
            indices = json.load(f)

        indices = [int(i) for i in indices]
        print(f"Loaded fixed reconstruction indices: {indices}")
        return indices

    by_class = {}

    for index, sample in enumerate(dataset.samples):
        label = int(sample[1])
        by_class.setdefault(label, []).append(index)

    rng = random.Random(seed)
    selected = []

    for label in sorted(by_class.keys()):
        candidates = by_class[label]
        rng.shuffle(candidates)

        if len(candidates) < samples_per_class:
            raise RuntimeError(
                f"Class {label} only has {len(candidates)} samples, "
                f"but samples_per_class={samples_per_class}"
            )

        selected.extend(candidates[:samples_per_class])

    with open(indices_file, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"Saved fixed reconstruction indices: {selected}")
    return selected


def build_fixed_reconstruction_batches(
    dataset,
    recon_config: dict,
    seed: int,
):
    indices = get_or_create_reconstruction_indices(
        dataset=dataset,
        indices_file=recon_config["fixed_indices_file"],
        samples_per_class=int(
            recon_config.get("samples_per_class", 1)
        ),
        seed=seed,
    )

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=int(recon_config.get("attack_batch_size", 1)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 保存在 CPU；每个 epoch 重复使用同一批数据
    return list(loader)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(
    dataset_path: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    train_split: float = 0.8,
    val_split: float = 0.1,
    mode: str = "train",
    datatype: int = 1 #0 for image folder (other dataset), 1 for rDR pickle
) -> tuple:
    """
    Create data loaders for SH_DR/shrdr dataset
    
    Args:
        dataset_path: 数据集根路径
        img_size: 图像大小
        batch_size: 批大小
        num_workers: 数据加载线程数
        train_split: 训练集比例
        val_split: 验证集比例
        
    Returns:
        (train_loader, val_loader, test_loader)
    """

    _, get_transform = import_generatepair_safely()

    print(f"Loading dataset from: {dataset_path}")
    if datatype==0:
        print(f"enter datatype 0")
        # 数据转换
        transform = get_transform(size=img_size, useimgnet=True)
        
        # 加载数据
        #dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

        dataset = ImageFolderWithPath(
            root=dataset_path,
            transform=transform,
        )
        dataset_size = len(dataset)
        
        print(f"Total samples: {dataset_size}")
        print(f"Classes: {dataset.class_to_idx}")
        
        # 分割数据
        train_size = int(dataset_size * train_split)
        val_size = int(dataset_size * val_split)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
    elif datatype==1:
        print(f"enter datatype 1")
        if mode == "train":
        
            train_dataset = DRPickleDatasetWithPath(
                pkl_path=os.path.join(dataset_path,"rDR_train_train_split.pkl"),
                image_root=os.path.join(dataset_path,"DR"),
                transform=get_transform(size=img_size, useimgnet=True),
                recursive=True,
                split_name="train",
            )

            val_dataset = DRPickleDatasetWithPath(
                pkl_path=os.path.join(dataset_path,"rDR_train_val_split.pkl"),
                image_root=os.path.join(dataset_path,"DR"),
                transform=get_transform(size=img_size, useimgnet=True),
                recursive=True,
                split_name="val",
            )

            train_size = len(train_dataset)
            val_size = len(val_dataset)
            test_size = 0
            dataset_size = train_size+val_size
        elif mode == "test":
            train_size = 0
            val_size = 0 
            test_dataset = DRPickleDatasetWithPath(
                pkl_path=os.path.join(dataset_path,"rDR_test.pkl"),
                image_root=os.path.join(dataset_path,"DR"),
                transform=get_transform(size=img_size, useimgnet=True),
                recursive=True,
                split_name=None,
            )
            test_size = len(test_dataset)
            dataset_size = test_size


    elif datatype == 3 or datatype == 4 or datatype == 5:
        # 只测试完全未见病人的图像
        if datatype == 3:
            list_file = "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/shrdr_test_overlap_lists/subject_0_eye_0_sample_0.txt"
        elif datatype == 4: # 只测试“同病人但不同眼睛”的图像：
            list_file = "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/shrdr_test_overlap_lists/subject_1_eye_0_sample_0.txt"
        elif datatype == 5:#只测试“同病人且同一只眼、但不同样本”的图像：
            list_file = "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/shrdr_test_overlap_lists/subject_1_eye_1_sample_0.txt"
        transform = get_transform(size=img_size, useimgnet=True)
        from eyedataset import ImageListDataset
        dataset = ImageListDataset(
            dataset_root=(
                "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/test"
            ),
            list_file=list_file,
            transform=transform,
        )
        
        
        dataset_size = len(dataset)
        
        print(f"Total samples: {dataset_size}")
        
        # 分割数据
        train_size = int(dataset_size * train_split)
        val_size = int(dataset_size * val_split)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )



   
     
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # 创建数据加载器
    if train_size>0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = None
    if val_size>0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        val_loader = None


    if test_size>0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        test_loader = None
    
    return train_loader, val_loader, test_loader, dataset_size

def import_generatepair_safely():
    import sys

    old_argv = sys.argv[:]
    try:
        # 避免 UWFound 内部 argparse 解析 dp_fl_train.py 的参数
        sys.argv = [old_argv[0]]
        from pairAttack.generatePair import getTargetModel, get_transform
    finally:
        sys.argv = old_argv

    return getTargetModel, get_transform


def create_client(
    client_id: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    args
) -> DPFLClient:
    """
    Create a DP-FL client
    
    Args:
        client_id: 客户端ID
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        config: 配置字典
        
    Returns:
        client: DPFLClient 实例
    """
    
    print(f"Creating client {client_id}")
    
    # 获取目标模型
    getTargetModel, get_transform = import_generatepair_safely()
    model, model_args = getTargetModel(outsideconfig=config)
    #model = MyClassifier(model) 这里不用，因为输入数据已经用了imagenet normalize
    
    # 损失函数
    #criterion = nn.CrossEntropyLoss()

    # Loss: label smoothing cross entropy, consistent with previous classifier baseline
    smoothing = config.get("hyperparams", {}).get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    print(f"Using CrossEntropyLoss with label_smoothing={smoothing}")

    # 优化器
    # optimizer_class = torch.optim.SGD
    # if config.get("optimizer", {}).get("name") == "adam":
    #     optimizer_class = torch.optim.Adam

    # Optimizer
    optimizer_name = config.get("optimizer", {}).get("name", "adamw").lower()

    if optimizer_name == "sgd":
        optimizer_class = torch.optim.SGD
    elif optimizer_name == "adam":
        optimizer_class = torch.optim.Adam
    elif optimizer_name == "adamw":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"Using optimizer: {optimizer_name}")

    # 隐私预算配置
    dp_config = config.get("dp", {})
    privacy_budget = PrivacyBudget(
        epsilon=dp_config.get("epsilon", 8.0),
        delta=dp_config.get("delta", 1e-5),
        max_per_sample_grad_norm=dp_config.get("max_per_sample_grad_norm", 1.0),
        mechanism=dp_config.get("mechanism", "rdp"),
        sigma=dp_config.get("sigma"),
    )

    optimizer_config = config.get("optimizer", {})

    optimizer_kwargs = {
        "lr": config.get("hyperparams", {}).get("lr", 1e-3),
    }

    if optimizer_name == "adamw":
        optimizer_kwargs["weight_decay"] = optimizer_config.get("weight_decay", 0.05)
        optimizer_kwargs["betas"] = tuple(optimizer_config.get("betas", [0.9, 0.999]))

    elif optimizer_name == "adam":
        optimizer_kwargs["weight_decay"] = optimizer_config.get("weight_decay", 0.0)
        optimizer_kwargs["betas"] = tuple(optimizer_config.get("betas", [0.9, 0.999]))

    elif optimizer_name == "sgd":
        optimizer_kwargs["momentum"] = optimizer_config.get("momentum", 0.9)
        optimizer_kwargs["weight_decay"] = optimizer_config.get("weight_decay", 0.0)    

    checkpoint_config = config.get("checkpoint", {})
    if args.modedp == "train":
        checkpoint_path = config.get("model", None).get("chkpt_path", "")
        print(f"Checkpoint training path from config: {checkpoint_path}")
    elif args.modedp == "test":
        checkpoint_path = args.modelpath
        print(f"Checkpoint testing path from args: {checkpoint_path}")
    load_optimizer_from_checkpoint = checkpoint_config.get("load_optimizer", False)
    evalmode = checkpoint_config.get("eval", False)

    save_dir = config.get("general", {}).get("save_dir", "./dp_fl_checkpoints")
    gradient_save_config = config.get("gradient_save", {})
    save_exposed_gradients = gradient_save_config.get(
        "enabled",
        dp_config.get("save_exposed_gradients", False),
    )
    gradient_save_dir = gradient_save_config.get(
        "dir",
        dp_config.get(
            "gradient_save_dir",
            os.path.join(save_dir, "exposed_gradients"),
        ),
    )
    gradient_save_interval = gradient_save_config.get(
        "interval",
        dp_config.get("gradient_save_interval", 1),
    )
    gradient_save_max_batches_per_epoch = gradient_save_config.get(
        "max_batches_per_epoch",
        dp_config.get("gradient_save_max_batches_per_epoch", 1),
    )
    gradient_save_max_total = gradient_save_config.get(
        "max_total",
        dp_config.get("gradient_save_max_total", None),
    )


    

    client = DPFLClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        privacy_budget=privacy_budget,
        device=device,
        client_id=client_id,
        local_epochs=config.get("federated_learning", {}).get("local_epochs", 1),
        dp_enabled=dp_config.get("enabled", True),
        evalmode=evalmode,
        checkpoint_path=checkpoint_path,
        load_optimizer_from_checkpoint=load_optimizer_from_checkpoint,
        save_exposed_gradients=save_exposed_gradients,
        gradient_save_dir=gradient_save_dir,
        gradient_save_interval=gradient_save_interval,
        gradient_save_max_batches_per_epoch=gradient_save_max_batches_per_epoch,
        gradient_save_max_total=gradient_save_max_total,
        reconstruction_enabled=config.get("reconstruction", {}).get("enabled", False)
    )   
    
    return client


def run_client(
    client_id: int,
    config: dict,
    server_address: str,
    args
):
    """
    Run a single FL client
    """
    
    print(f"Starting Client {client_id}")
    
    # 加载数据
    dataset_config = config.get("dataset", {})
    dataset_path = dataset_config.get("base_path")
    
    # 为每个客户端分割数据（简单的IID分割）
    datatype = dataset_config.get("datatype",1)
    print(f"datatype:{datatype}")
    train_loader, val_loader, test_loader, dataset_size = create_data_loaders(
        dataset_path=dataset_path,
        img_size=config.get("model", {}).get("input_size", 224),
        batch_size=config.get("hyperparams", {}).get("batch_size", 32),
        num_workers=2,
        train_split=dataset_config.get("train_split", 0.8),
        val_split=dataset_config.get("val_split", 0.2),
        datatype=datatype
    )
    
    # 创建客户端
    client = create_client(
        client_id=client_id,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        args=args
    )
    
    # 启动客户端
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )
def get_warmup_cosine_lr(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
) -> float:
    """
    Linear warmup + cosine decay learning rate.
    """
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))

    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + (base_lr - min_lr) * cosine_decay

def main(args):
    """Main entry point"""
    
    # 加载配置
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Config: {yaml.dump(config)}")
    
    # 设置随机种子
    seed = config.get("general", {}).get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 根据role决定运行模式
    if args.role == "server":
        # 运行服务器
        print("Starting Federated Learning Server")
        
        fl_config = config.get("federated_learning", {})
        server_config = fl_config.get("server", {})
        
        run_federated_server(
            server_address=server_config.get("address", "0.0.0.0:8080"),
            num_rounds=fl_config.get("num_rounds", 50),
            num_clients=fl_config.get("num_clients", 5),
            clients_per_round=fl_config.get("clients_per_round", 3),
            partial_weight_fraction=fl_config.get("partial_weight_fraction", 1.0),
            min_available_clients=max(1, fl_config.get("clients_per_round", 3) - 1),
        )
    
    elif args.role == "client":
        # 运行客户端
        fl_config = config.get("federated_learning", {})
        server_config = fl_config.get("server", {})
        
        client_id = args.client_id if args.client_id is not None else 0
        
        run_client(
            client_id=client_id,
            config=config,
            server_address=server_config.get("address", "127.0.0.1:8080"),
        )
    
    elif args.role == "standalone":
        # 独立训练模式（单机，无FL）
        print("Starting Standalone DP Training (No FL)")
        
        # 加载数据
        dataset_config = config.get("dataset", {})
        dataset_path = dataset_config.get("base_path")
        
        # train_loader, val_loader, test_loader, dataset_size = create_data_loaders(
        #     dataset_path=dataset_path,
        #     img_size=config.get("model", {}).get("input_size", 224),
        #     batch_size=config.get("hyperparams", {}).get("batch_size", 32),
        #     num_workers=4,
        # )

        datatype = dataset_config.get("datatype",1)
        print(f"datatype:{datatype}")
       
        train_loader, val_loader, test_loader, dataset_size = create_data_loaders(
            dataset_path=dataset_path,
            img_size=config.get("model", {}).get("input_size", 224),
            batch_size=config.get("hyperparams", {}).get("batch_size", 32),
            num_workers=2,
            train_split=dataset_config.get("train_split", 0.8),
            val_split=dataset_config.get("val_split", 0.2),
            datatype=datatype,
            mode=args.modedp
        )
        # 创建客户端（作为本地训练器）
        client = create_client(
            client_id=0,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            args = args
        )
        if args.modedp=="test":

            # test
            num_test_samples, test_metrics = client.test()

            val_auroc = float(test_metrics.get("val_auroc", float("nan")))

            print(f"Test metrics: {test_metrics}")


        elif args.modedp=="train":
            base_lr= config.get("hyperparams", {}).get("blr", 0.001)
            min_lr= config.get("hyperparams", {}).get("min_lr", 0.000001)
            warmup_epochs= config.get("hyperparams", {}).get("warmup_epochs", 5)

            # 计算隐私预算
            dp_config = config.get("dp", {})
            hyperparams = config.get("hyperparams", {})
            
            if dp_config.get("enabled"):
                num_batches_per_epoch = len(train_loader)
                grad_acc_steps = hyperparams.get("grad_acc_steps", 1)
                final_epsilon = update_privacy_budget(
                    privacy_budget=client.privacy_budget,
                    num_epochs=hyperparams.get("epochs", 50),
                    num_batches_per_epoch=num_batches_per_epoch,
                    batch_size=hyperparams.get("batch_size", 32),
                    dataset_size=len(train_loader.dataset),
                    alphas=dp_config.get("alphas", []),
                    grad_acc_steps=grad_acc_steps,
                )
            
            # 独立训练
            print("Starting standalone training")
            
            num_epochs = hyperparams.get("epochs", 50)



            best_val_auroc = -float("inf")
            best_epoch = -1

            save_dir = config.get("general", {}).get("save_dir", "./dp_fl_checkpoints")
            os.makedirs(save_dir, exist_ok=True)


            recon_config = config.get("reconstruction", {})
            reconstruction_enabled = bool(recon_config.get("enabled", False))

            if reconstruction_enabled:
                reconstruction_batches = build_fixed_reconstruction_batches(
                    dataset=train_loader.dataset,
                    recon_config=recon_config,
                    seed=seed,
                )
            else:
                reconstruction_batches = []

            target_reconstruction_epochs = sorted({
                0,
                3,
                num_epochs // 2,
                num_epochs - 1,
            })

            print(
                f"Reconstruction target epochs: "
                f"{target_reconstruction_epochs}"
            )

            reconstruction_payload_paths = []


            val_size = dataset_config.get("val_split", 0)

            has_validation = (
                val_size > 0
            )

            print(
                f"Validation samples: {val_size}"
            )

            if not has_validation:
                print(
                    "Validation set is empty. "
                    "Best-model selection by validation AUROC "
                    "will be disabled."
                )
                
            for epoch in range(num_epochs):
                print(f"Epoch {epoch + 1}/{num_epochs}")

                current_lr = get_warmup_cosine_lr(
                    epoch=epoch,
                    total_epochs=num_epochs,
                    base_lr=base_lr,
                    min_lr=min_lr,
                    warmup_epochs=warmup_epochs,
                )

                print(f"Current learning rate: {current_lr:.8f}")

                # Train
                train_paths_dir = Path(config.get("train_paths_output", "./train_paths"))
                train_paths_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                train_paths_file = (
                train_paths_dir
                / f"trainset_{epoch}.txt"
                )
                train_params, num_samples, train_metrics = client.fit(
                    parameters=[],
                    config={
                        "round": epoch,
                        "partial_weight_fraction": 1.0,
                        "learning_rate": current_lr,
                        "record_train_paths":config.get("record_train_paths", True),
                        "train_paths_output":train_paths_file,
                    }
                )

                print(f"Train metrics: {train_metrics}")

                if has_validation:
                    # Validate
                    val_loss, num_val_samples, val_metrics = client.evaluate(
                        parameters=train_params,
                        config={"round": epoch}
                    )

                    val_loss_float = float(val_loss)
                    val_auroc = float(val_metrics.get("val_auroc", float("nan")))

                    print(f"Val metrics: {val_metrics}")

                    # Save best by AUROC
                    if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
                        print(
                            f"Update best val AUROC: {best_val_auroc:.6f} -> {val_auroc:.6f}"
                        )

                        best_val_auroc = val_auroc
                        best_epoch = epoch

                        best_path = os.path.join(save_dir, "model_standalone_best.pt")
                        best_epoch_path = os.path.join(save_dir, f"model_standalone_best_ep{epoch}.pt")

                        checkpoint = {
                            "epoch": epoch,
                            "model_state_dict": client.model.state_dict(),
                            "val_loss": val_loss_float,
                            "val_auroc": val_auroc,
                            "best_val_auroc": best_val_auroc,
                            "best_epoch": best_epoch,
                            "val_metrics": val_metrics,
                            "train_metrics": train_metrics,
                            "learning_rate": current_lr,
                            "num_val_samples": num_val_samples,
                            "num_train_samples": num_samples,
                        }


                        checkpoint = {
                            "epoch": epoch,
                            "model_state_dict": client.model.state_dict(),
                            "val_loss": val_loss_float,
                            "val_auroc": val_auroc,
                            "best_val_auroc": best_val_auroc,
                            "best_epoch": best_epoch,
                            "val_metrics": val_metrics,
                            "train_metrics": train_metrics,
                            "learning_rate": current_lr,
                            "num_val_samples": num_val_samples,
                            "num_train_samples": num_samples,

                            "privacy_metadata": {
                                "enabled": bool(dp_config.get("enabled", False)),
                                "epsilon": float(dp_config.get("epsilon", 8.0)),
                                "delta": float(dp_config.get("delta", 1.0e-5)),
                                "sigma": (
                                    None
                                    if client.privacy_budget.sigma is None
                                    else float(client.privacy_budget.sigma)
                                ),
                                "max_per_sample_grad_norm": float(
                                    dp_config.get("max_per_sample_grad_norm", 1.0)
                                ),
                                "mechanism": str(dp_config.get("mechanism", "rdp")),
                            },

                            "training_metadata": {
                                "label_smoothing": float(
                                    hyperparams.get("label_smoothing", 0.1)
                                ),
                                "batch_size": int(
                                    hyperparams.get("batch_size", 32)
                                ),
                                "input_size": int(
                                    config.get("model", {}).get("input_size", 224)
                                ),
                            },
                        }

                        torch.save(checkpoint, best_path)
                        torch.save(checkpoint, best_epoch_path)

                        print(f"Best model saved to {best_path}")
                        print(f"Best epoch model saved to {best_epoch_path}")

                else:
                    val_loss_float = 0.0
                    val_auroc = float("nan")
                    val_metrics = {}
                    num_val_samples = 0

                    print(
                        "Validation set is empty. "
                        "Skip validation for this epoch."
                    )
                                
                # Save epoch checkpoint
                epoch_path = os.path.join(save_dir, f"model_standalone_ep{epoch}.pt")

                
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": client.model.state_dict(),
                    "val_loss": val_loss_float,
                    "val_auroc": val_auroc,
                    "best_val_auroc": best_val_auroc,
                    "best_epoch": best_epoch,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "learning_rate": current_lr,
                    "num_val_samples": num_val_samples,
                    "num_train_samples": num_samples,

                    "privacy_metadata": {
                        "enabled": bool(dp_config.get("enabled", False)),
                        "epsilon": float(dp_config.get("epsilon", 8.0)),
                        "delta": float(dp_config.get("delta", 1.0e-5)),
                        "sigma": (
                            None
                            if client.privacy_budget.sigma is None
                            else float(client.privacy_budget.sigma)
                        ),
                        "max_per_sample_grad_norm": float(
                            dp_config.get("max_per_sample_grad_norm", 1.0)
                        ),
                        "mechanism": str(dp_config.get("mechanism", "rdp")),
                    },

                    "training_metadata": {
                        "label_smoothing": float(
                            hyperparams.get("label_smoothing", 0.1)
                        ),
                        "batch_size": int(
                            hyperparams.get("batch_size", 32)
                        ),
                        "input_size": int(
                            config.get("model", {}).get("input_size", 224)
                        ),
                    },
                }
            
                torch.save(checkpoint, epoch_path)
                print(f"Model saved to {epoch_path}")
            
                
            print("Standalone training finished")

            best_path = os.path.join(save_dir, "model_standalone_best.pt")
            save_path = os.path.join(save_dir, "final_model_standalone.pt")

            if os.path.exists(best_path) and best_epoch>=0 :
                best_ckpt = torch.load(best_path, map_location=client.device)
                torch.save(best_ckpt, save_path)

                print(
                    f"Final model saved from best checkpoint to {save_path}. "
                    f"Best epoch={best_ckpt.get('epoch')}, "
                    f"Best val AUROC={best_ckpt.get('best_val_auroc'):.6f}"
                )
            else:


                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": client.model.state_dict(),
                    "val_loss": val_loss_float,
                    "val_auroc": val_auroc,
                    "best_val_auroc": best_val_auroc,
                    "best_epoch": best_epoch,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "learning_rate": current_lr,
                    "num_val_samples": num_val_samples,
                    "num_train_samples": num_samples,

                    "privacy_metadata": {
                        "enabled": bool(dp_config.get("enabled", False)),
                        "epsilon": float(dp_config.get("epsilon", 8.0)),
                        "delta": float(dp_config.get("delta", 1.0e-5)),
                        "sigma": (
                            None
                            if client.privacy_budget.sigma is None
                            else float(client.privacy_budget.sigma)
                        ),
                        "max_per_sample_grad_norm": float(
                            dp_config.get("max_per_sample_grad_norm", 1.0)
                        ),
                        "mechanism": str(dp_config.get("mechanism", "rdp")),
                    },

                    "training_metadata": {
                        "label_smoothing": float(
                            hyperparams.get("label_smoothing", 0.1)
                        ),
                        "batch_size": int(
                            hyperparams.get("batch_size", 32)
                        ),
                        "input_size": int(
                            config.get("model", {}).get("input_size", 224)
                        ),
                    },
                }


                torch.save(checkpoint, save_path)
                print(f"Final model saved to {save_path}")


        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP Federated Learning Training for DeepUWF")
    
    parser.add_argument(
        "--config",
        type=str,
        default="/path/to/workspace/code/DeepUWF/DP/dp_fl_config_baseline_encrypt.yaml",
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--role",
        type=str,
        choices=["server", "client", "standalone"],
        default="standalone",
        help="Role to run: server, client, or standalone",
    )
    
    parser.add_argument(
        "--client_id",
        type=int,
        default=None,
        help="Client ID (only used when role=client)",
    )

    parser.add_argument(
        "--modedp",
        type=str,
        default="train",
        help="train or test",
    )



    parser.add_argument(
        "--modelpath",
        type=str,
        default="",
        help="train or test",
    )


    args = parser.parse_args()

    main(args)
