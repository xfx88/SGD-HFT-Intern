import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from sklearn.metrics import r2_score
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import OzeDataset
from src.metrics import MSE
from src.utils import Logger
from tst import Transformer
from tst.loss import OZELoss


def compute_loss(net: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 rank) -> torch.Tensor:
    """Compute the loss of a network on a given dataset.

    Does not compute gradient.

    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.

    Returns
    -------
    Loss as a tensor with no grad.
    """
    running_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            netout = net(x.to(rank)).cpu()
            running_loss += loss_function(y, netout)

    return running_loss / len(dataloader)


def fit(net, optimizer, loss_function, dataloader_train, dataloader_val, epochs=10, pbar=None, rank = None):
    # val_loss_best = np.inf

    # Prepare loss history
    for idx_epoch in range(epochs):
        for idx_batch, (x, y) in enumerate(dataloader_train):
            x = x[0 + rank * 20 : (rank + 1) * 30, :300, :]
            y = y[0 + rank * 20 : (rank + 1) * 30, :300, :]

            optimizer.zero_grad()
            # Propagate input
            netout = net(x.to(rank))

            # Comupte loss
            loss = loss_function(y.to(rank), netout)
            if dist.get_rank() == 0:
                print(loss.item())

            # Backpropage loss
            loss.backward()

            # Update weights
            optimizer.step()

        # val_loss = compute_loss(net, dataloader_val, loss_function, rank).item()

        # if val_loss < val_loss_best:
        #     val_loss_best = val_loss

        if pbar is not None:
            pbar.update()

    return loss.item()


# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# Training parameters
DATASET_PATH = 'data/training.npz'
BATCH_SIZE = 200
NUM_WORKERS = 0
LR = 2e-4
EPOCHS = 1

# Model parameters
d_model = 64  # Latent dim
q = 8  # Query size
v = 8  # Value size
h = 2  # Number of heads
N = 4  # Number of encoder and decoder to stack
attention_size = 10  # Attention window size
dropout = 0.3  # Dropout rate
pe = None  # Positional encoding
chunk_mode = None

d_input = 37  # From dataset
d_output = 8  # From dataset

# Config
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device {device}")
def load_data():
    # Load dataset
    ozeDataset = OzeDataset(DATASET_PATH)

    # Split between train, validation and test
    dataset_train, dataset_val, dataset_test = random_split(
        ozeDataset, (7000, 250, 250))

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=False
                                  )

    dataloader_val = DataLoader(dataset_val,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=NUM_WORKERS
                                )

    dataloader_test = DataLoader(dataset_test,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS
                                 )
    return dataloader_train, dataloader_val, dataloader_test

def train(rank, world_size, dataloader_train, dataloader_val, dataloader_test):
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

    net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                      dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(rank)
    device_id = rank + 1
    ddp_net = DDP(net, device_ids = [device_id])
    optimizer = optim.Adam(ddp_net.parameters(), lr=LR)
    loss_function = OZELoss(alpha=0.3)


    # Load transformer with Adam optimizer and MSE loss function


    metrics = {
        'training_loss': lambda y_true, y_pred: OZELoss(alpha=0.3, reduction='none')(y_true, y_pred).numpy(),
        'mse_tint_total': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none'),
        'mse_cold_total': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[0, 1, 2, 3, 4, 5, 6], reduction='none'),
        'mse_tint_occupation': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none', occupation=occupation),
        'mse_cold_occupation': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[0, 1, 2, 3, 4, 5, 6], reduction='none', occupation=occupation),
        'r2_tint': lambda y_true, y_pred: np.array([r2_score(y_true[:, i, -1], y_pred[:, i, -1]) for i in range(y_true.shape[1])]),
        'r2_cold': lambda y_true, y_pred: np.array([r2_score(y_true[:, i, 0:-1], y_pred[:, i, 0:-1]) for i in range(y_true.shape[1])])
    }

    logger = Logger(f'data/logs/training.csv', model_name=net.name,
                    params=[y for key in metrics.keys() for y in (key, key+'_std')])

    # Fit model
    with tqdm(total=EPOCHS) as pbar:
        loss = fit(net, optimizer, loss_function, dataloader_train,
                   dataloader_val, epochs=EPOCHS, pbar=pbar, rank=rank)

    # Switch to evaluation
    _ = net.eval()

    # Select target values in test split
    # y_true = ozeDataset._y[dataloader_test.dataset.indices]
    #
    # # Compute predictions
    # predictions = torch.empty(len(dataloader_test.dataset), 168, 8)
    # idx_prediction = 0
    # with torch.no_grad():
    #     for x, y in tqdm(dataloader_test, total=len(dataloader_test)):
    #         netout = net(x.cuda()).cpu()
    #         predictions[idx_prediction:idx_prediction+x.shape[0]] = netout
    #         idx_prediction += x.shape[0]
    #
    # # Compute occupation times
    # occupation = ozeDataset._x[dataloader_test.dataset.indices,
    #                            :, ozeDataset.labels['Z'].index('occupancy')]
    #
    # results_metrics = {
    #     key: value for key, func in metrics.items() for key, value in {
    #         key: func(y_true, predictions).mean(),
    #         key+'_std': func(y_true, predictions).std()
    #     }.items()
    # }
    #
    # # Log
    # logger.log(**results_metrics)
    #
    # # Save model
    # torch.save(net.state_dict(),
    #            f'models/{net.name}_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.pth')


def main(dataloader_train, dataloader_val, dataloader_test):
    world_size = 3
    mp.spawn(train,
             args = (world_size,dataloader_train, dataloader_val, dataloader_test, ),
             nprocs=world_size,
             join = True)


if __name__ == "__main__":
    dataloader_train, dataloader_val, dataloader_test = load_data()
    main(dataloader_train, dataloader_val, dataloader_test)