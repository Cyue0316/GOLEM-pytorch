import time
import torch
from utils.meters import AverageMeter, ProgressMeter

def validate(val_loader, model, epoch=0, test=True, args=None, tensor_writer=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, data_time, losses],
            prefix="Test:[{}]".format(epoch)
        )
    else:
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, data_time, losses],
            prefix="Val:[{}]".format(epoch)
        )

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, X in enumerate(val_loader):
            loss, _, _, B_est = model(X)
            losses.update(loss.item(), X.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        # print('* Score {}')

        if test:
            tensor_writer.add_scalar('loss/test', loss.item(), epoch)
        else:
            tensor_writer.add_scalar('loss/val', loss.item(), epoch)