import time
import logging
import torch
import torch.nn as nn
import torch.optim
import numpy as np
from utils.dir import create_dir
from utils.meters import AverageMeter, ProgressMeter
from utils.sophia import SophiaG
from torch.utils import data
from tqdm import tqdm
import random

from torch.utils.data import DataLoader
from utils.train import postprocess
from utils.utils import count_accuracy, set_seed, plot_solution


# _logger = logging.getLogger(__name__)


def load_data(data_arrays, batch_size, is_train=True):
    return data.DataLoader(data_arrays, batch_size, shuffle=is_train)


def load_seq_data(time_span, batch_size, features):
    len_examples = features.shape[0]
    num_examples = features.shape[1]
    num_subseqs = num_examples // batch_size
    initial_indices = list(range(0, num_subseqs * batch_size, batch_size))
    random.shuffle(initial_indices)

    for i in range(time_span, len_examples - time_span + 1):
        for j in initial_indices:
            yield features[i:i + time_span, j:j + batch_size, :]

def train_epoch(model, train_iter, trainer):

    for X in enumerate(train_iter):
        score, likelihood, h, B_est, A_est = model(X)
        trainer.zero_grad()
        score.backward(retain_graph=True)
        trainer.step()
    return score, likelihood, h, B_est, A_est

def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train(train_loader, model, epoch, target, optimizer, tensor_writer=None, checkpoint_iter=None, output_dir=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch:[{}]".format(epoch)
    )

    end = time.time()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, X in loop:
        # print(X.shape)
        data_time.update(time.time() - end)
        loss, _, _, B_est = model(X)
        losses.update(loss.item(), X.size(0))

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        # loss, likelihood, h, B_est = model(X)
        tempB = B_est.clone()
        B_est_np = tempB.detach().numpy()
        B_processed = postprocess(B_est_np, graph_thres=0.3)
        results = count_accuracy(target != 0, B_processed != 0)
        loop.set_description(f'Epoch [{epoch}/{1000}]')
        loop.set_postfix(loss=loss.item(), fdr=results['fdr'], tpr=results['tpr'],
                         fpr=results['fpr'], shd=results['shd'], pred_size=results['pred_size'])
    # if epoch % 50 == 0:
    #     print(f'epoch {epoch + 1}, loss {loss:f}')
    #     print(results)
    #     plot_solution(target, B_est_np, B_processed, 'Epoch_%d_CM.png' % (epoch))
    tensor_writer.add_scalar('loss/train', losses.avg, epoch)





# def train_checkpoint(i, score, likelihood, h, B_est, output_dir):
#     _logger.info(
#         "[Iter {}] score {:.3E}, likelihood {:.3E}, h{:.3E}".format(
#             i, score, likelihood, h)
#     )
#     if output_dir is not None:
#         create_dir('{}/checkpoints'.format(output_dir))
#         np.save('{}/checkpoints/B_iteration_{}.npy'.format(output_dir, i), B_est)


if __name__ == '__main__':
    from data_loader.synthetic_dataset import SyntheticDataset
    from data_loader.synthetic_seq import SyntheticSeqDataset
    from utils.train import postprocess
    from utils.utils import count_accuracy, set_seed, plot_solution, plot_dysolution
    from models.golem_model import GolemModel
    from models.DYGO import DYGO

    set_seed(1)

    # Load dataset
    n, d = 1, 5
    k, l = 3, 500
    eta = 1.5
    graph_type, degree = 'ER', 4
    A_scale = 1.0
    B_scale = 1.0
    noise_type = 'gaussian_ev'
    batch_size = 1
    Epochs = 1000
    # # golem dataset
    # dataset = SyntheticDataset(n, d, graph_type, degree,
    #                            noise_type, B_scale, seed=1)
    # golem = GolemModel(n=n, d=d, lambda_1=2e-3, lambda_2=0.5, seed=1)
    # X = torch.from_numpy(dataset.X).to(torch.float32)

    # DYGO dataset
    # dataset = SyntheticSeqDataset(n, d, k, l, eta, graph_type, degree,
    #                               noise_type, A_scale, B_scale, seed=1)
    #
    # np.save('features.npy', dataset.X)
    # np.save('intra.npy', dataset.B)
    # np.save('inter.npy', dataset.A)

    # plot_dysolution(dataset.B, dataset.A, save_name='N%dD%dK%dL%d.png' % (n, d, k, l))

    features = np.load('features.npy')
    features = features.reshape(2000,1,5)

    W = np.load('W_true.npy')
    # A = np.load('inter.npy')
    features = torch.from_numpy(features).to(torch.float32)
    dygolem = DYGO(d=d, k=k, lambda_1=2e-3, lambda_2=0.5, lambda_3=2e-3, seed=1)
    # trainer = torch.optim.Adam(dygolem.parameters(), lr=0.003)
    trainer = SophiaG(dygolem.parameters(), lr=0.003)
    for epoch in range(Epochs):
        data_iter = load_seq_data(time_span=k + 1, batch_size=batch_size, features=features)
        for i, X in enumerate(data_iter):
            score, likelihood, h, B_est, A_est = dygolem(X)
            trainer.zero_grad()
            score.backward()
            grad_clipping(dygolem, 1)
            trainer.step()
        tempB = B_est.clone()
        B_est_np = tempB.detach().numpy()
        B_processed = postprocess(B_est_np, graph_thres=0.01)
        results = count_accuracy(W[3*d:, -d:] != 0, B_processed != 0)
        if epoch % 50 == 0:
            print(f'epoch {epoch + 1}, loss {score:f}')
            print(results)
            plot_solution(W[3*d:, -d:], B_est_np, B_processed, 'DYGO/Epoch_%d_CM.png' % (epoch))

    W_full = np.load('W_est_full.npy')

    results = count_accuracy(B != 0, W_full[3*d:, -d:] != 0)
    print(results)




    # data_iter = load_data(X, batch_size)
    # trainer = torch.optim.Adam(golem.parameters(), lr=0.003)
    # for epoch in range(Epochs):
    #     for X in data_iter:
    #         score, likelihood, h, B_est = golem(X)
    #         trainer.zero_grad()
    #         score.backward()
    #         trainer.step()
    #     score, likelihood, h, B_est = golem(X)
    #     tempB = B_est.clone()
    #     B_est_np = tempB.detach().numpy()
    #     B_processed = postprocess(B_est_np, graph_thres=0.3)
    #     results = count_accuracy(dataset.B != 0, B_processed != 0)
    #     if epoch % 50 == 0:
    #         print(f'epoch {epoch+1}, loss {score:f}')
    #         print(results)
    # plot_solution(dataset.B, B_est_np, B_processed, 'Epoch_%d_CM.png'%(epoch))
