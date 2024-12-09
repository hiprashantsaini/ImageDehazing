import time
import warnings
import math
import numpy as np

from metrics import ssim, psnr  # <-- Add this line to import ssim and psnr

from tensorboardX import SummaryWriter
from torch import optim
from torch.backends import cudnn

from C2R import C2R
from data_utils import *
from models.C2PNet import *
from option import model_name, log_dir

warnings.filterwarnings('ignore')

print('log_dir :', log_dir)
print('model_name:', model_name)

models_ = {
    'C2PNet': C2PNet(gps=opt.gps, blocks=opt.blocks),
}

# Removed loaders that reference negative samples
loaders_ = {
    'its_train': ITS_train_loader,
    'its_test': ITS_test_loader,
    'ots_test': OTS_test_loader,
}

start_time = time.time()
T = opt.steps
cl_lambda = opt.cl_lambda

def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr

def clcr_train(train_model, train_loader, test_loader, optim, criterion):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    best_psnr = 0
    ssims = []
    psnrs = []
    initial_loss_weight = opt.loss_weight

    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_step:{start_step} start training ---')
    else:
        print('train from scratch *** ')

    for step in range(start_step + 1, opt.steps + 1):
        train_model.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        x, y = next(iter(train_loader))  # Only using clear and hazy images
        x = x.to(opt.device)
        y = y.to(opt.device)

        out = train_model(x)
        pixel_loss = criterion[0](out, y)
        loss = pixel_loss  # Removed additional loss components

        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())

        print(
            f'\rpixel loss : {pixel_loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(train_model, test_loader)
                print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            with SummaryWriter(logdir=log_dir, comment=log_dir) as writer:
                writer.add_scalar('data/ssim', ssim_eval, step)
                writer.add_scalar('data/psnr', psnr_eval, step)
                writer.add_scalar('data/loss', loss.item(), step)

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)

            torch.save({
                'step': step,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': train_model.state_dict(),
            }, opt.latest_model_dir)

            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'ps nrs': psnrs,
                    'losses': losses,
                    'model': train_model.state_dict(),
                }, opt.model_dir)
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)

def test(test_model, loader_test):
    test_model.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred = test_model(inputs)
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    return np.mean(ssims), np.mean(psnrs)

if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    pytorch_total_params = sum(p.nelement() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params / 1e6))
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = [nn.L1Loss().to(opt.device)]
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    clcr_train(net, loader_train, loader_test, optimizer, criterion)