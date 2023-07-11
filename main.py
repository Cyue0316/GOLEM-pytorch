import torch
import os
from trainers.trainers import load_data, train
from data_loader.synthetic_dataset import SyntheticDataset
from utils.train import postprocess
from utils.utils import count_accuracy, set_seed, plot_solution
from models.Stable_golem_model import GolemModel
from models.reweighting import *
from torch.utils.tensorboard import SummaryWriter
import logging
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from tqdm import tqdm
set_seed(1)

log_base = './results'
dataset = 'SyntheticData'
log_path = os.path.join(log_base, dataset, 'log.txt')
log_dir = os.path.dirname(log_path)
tensor_writer = SummaryWriter(log_dir)
# Load dataset
n, d = 1000, 20
graph_type, degree = 'ER', 4
A_scale = 1.0
B_scale = 1.0
noise_type = 'gaussian_ev'
batch_size = 10
Epochs = 1000
dataset = SyntheticDataset(n, d, graph_type, degree,
                           noise_type, B_scale, seed=1)
golem = GolemModel(n=n, d=d, lambda_1=2e-3, lambda_2=0.5, seed=1)
X = torch.from_numpy(dataset.X).to(torch.float32)
# RW = DWR(X, cov_mask=None, order=1, num_steps=5000, lr=0.01, tol=1e-8, loss_lb=0.001, iter_print=500, logger=None,
#             device=None)
# X = X*RW

data_iter = load_data(X, batch_size)
def main():
    model = GolemModel(n=n, d=d, lambda_1=2e-3, lambda_2=0.5, seed=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    _, _, _, B_est = model(X)
    tempB = B_est.clone()
    B_est_np = tempB.detach().numpy()
    B_processed = postprocess(B_est_np, graph_thres=0.3)
    results = count_accuracy(dataset.B != 0, B_processed != 0)
    print(results)
    for epoch in range(1,Epochs+1):
        train(data_iter, model, epoch, target=dataset.B,
              optimizer=optimizer, tensor_writer=tensor_writer)

    _, _, _, B_est = model(X)
    tempB = B_est.clone()
    B_est_np = tempB.detach().numpy()
    B_processed = postprocess(B_est_np, graph_thres=0.3)
    results = count_accuracy(dataset.B != 0, B_processed != 0)
    print(results)

if __name__ == '__main__':
    main()