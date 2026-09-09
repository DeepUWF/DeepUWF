import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os

join = os.path.join
from tqdm import tqdm

from Model.models_vit import resnet50 as resnet50
from Model.util.datasets import build_transform, LabeledDataset, split_dataset
from arguments import get_args_parser

from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')


arg_parser = get_args_parser()
args = arg_parser.parse_args()

root = args.data_path
transform = build_transform(True, args)


net = resnet50(args).to(device=args.device)  # 1 normal and 8 diseases


train_dataset, test_dataset = split_dataset(root, args.label, args)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size * 2,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
)

# begin to train
optimizer = torch.optim.Adam(net.parameters(), lr=args.blr)
# loss function for multi-class
criterion = nn.BCELoss()

max_auc = 0

for epoch in range(args.epochs):

    true_label_decode_list = []
    pred_label_decode_list = []
    loss_sum = 0

    net.train(True)
    for i, (inputs, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs = inputs.to(device=args.device)
        targets = targets.reshape(-1, args.num_classes).to(torch.float).to(device=args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = nn.Sigmoid()(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        true_label_decode_list.extend(targets.cpu().numpy())
        pred_label_decode_list.extend(outputs.cpu().detach().numpy())
    # calculate auc
    auc_train = roc_auc_score(true_label_decode_list, pred_label_decode_list,
                              average='samples', multi_class='ovr')
    print(f'Epoch {epoch}, train auc: {auc_train}, loss: {loss_sum / len(train_dataloader)}')

    # test
    true_label_decode_list = []
    pred_label_decode_list = []
    loss_sum = 0

    net.train(False)
    for i, (inputs, targets) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        inputs = inputs.to(device=args.device)
        targets = targets.reshape(-1, args.num_classes).to(torch.float).to(device=args.device)
        outputs = net(inputs)
        outputs = nn.Sigmoid()(outputs)
        loss = criterion(outputs, targets)
        loss_sum += loss.item()

        true_label_decode_list.extend(targets.cpu().numpy())
        pred_label_decode_list.extend(outputs.cpu().detach().numpy())
    # calculate auc
    auc_test = roc_auc_score(true_label_decode_list, pred_label_decode_list,
                             average='samples', multi_class='ovr')

    print(f'Epoch {epoch}, test auc: {auc_test}, loss: {loss_sum / len(test_dataloader)}')

    if auc_test > max_auc:
        max_auc = auc_test
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        torch.save(net.state_dict(), './checkpoints/teacher_model.pth')
        print('Model saved')
