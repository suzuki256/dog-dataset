import numpy as np
import os
import argparse
import json
import itertools
import logging
import warnings

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import MultiDomain_Dataset, collate_fn
import net

def makedirs_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def comb(N,r):
    iterable = list(range(0,N))
    return list(itertools.combinations(iterable,2))

def Train(models, epochs, train_dataset, train_loader, optimizers, device, model_dir, log_path, config, snapshot=100, resume=0):
    fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    datafmt = '%m/%d/%Y %I:%M:%S'
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO, format=fmt, datefmt=datafmt)
    writer = SummaryWriter(os.path.dirname(log_path))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for tag in ['gen', 'dis']:
        checkpointpath = os.path.join(model_dir, '{}.{}.pt'.format(resume,tag))
        if os.path.exists(checkpointpath):
            checkpoint = torch.load(checkpointpath, map_location=device)
            models[tag].load_state_dict(checkpoint['model_state_dict'])
            optimizers[tag].load_state_dict(checkpoint['optimizer_state_dict'])
            print('{} loaded successfully.'.format(checkpointpath))

    w_adv = config['w_adv']
    w_grad = config['w_grad']
    w_cls = config['w_cls']
    w_cyc = config['w_cyc']
    w_rec = config['w_rec']
    gradient_clip = config['gradient_clip']

    print("===================================Training Started===================================")
    n_iter = 0
    for epoch in range(resume+1, epochs+1):
        b = 0
        for X_list in train_loader:
            n_spk = len(X_list)
            xin = []
            for s in range(n_spk):
                xin.append(torch.tensor(X_list[s]).to(device, dtype=torch.float))

            # List of speaker pairs
            spk_pair_list = comb(n_spk,2)
            n_spk_pair = len(spk_pair_list)

            gen_loss_mean = 0
            dis_loss_mean = 0
            advloss_d_mean = 0
            gradloss_d_mean = 0
            advloss_g_mean = 0
            clsloss_d_mean = 0
            clsloss_g_mean = 0
            cycloss_mean = 0
            recloss_mean = 0
            # Iterate through all speaker pairs
            for m in range(n_spk_pair):
                s0 = spk_pair_list[m][0]
                s1 = spk_pair_list[m][1]

                AdvLoss_g, ClsLoss_g, CycLoss, RecLoss = models['stargan'].calc_gen_loss(xin[s0], xin[s1], s0, s1)
                gen_loss = (w_adv * AdvLoss_g + w_cls * ClsLoss_g + w_cyc * CycLoss + w_rec * RecLoss)

                models['gen'].zero_grad()
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(models['gen'].parameters(), gradient_clip)
                optimizers['gen'].step()
                
                AdvLoss_d, GradLoss_d, ClsLoss_d = models['stargan'].calc_dis_loss(xin[s0], xin[s1], s0, s1)
                dis_loss = w_adv * AdvLoss_d + w_grad * GradLoss_d + w_cls * ClsLoss_d

                models['dis'].zero_grad()
                dis_loss.backward()
                torch.nn.utils.clip_grad_norm_(models['dis'].parameters(), gradient_clip)
                optimizers['dis'].step()

                gen_loss_mean += gen_loss.item()
                dis_loss_mean += dis_loss.item()
                advloss_d_mean += AdvLoss_d.item()
                gradloss_d_mean += GradLoss_d.item()
                advloss_g_mean += AdvLoss_g.item()
                clsloss_d_mean += ClsLoss_d.item()
                clsloss_g_mean += ClsLoss_g.item()
                cycloss_mean += CycLoss.item()
                recloss_mean += RecLoss.item()

            gen_loss_mean /= n_spk_pair
            dis_loss_mean /= n_spk_pair
            advloss_d_mean /= n_spk_pair
            gradloss_d_mean /= n_spk_pair
            advloss_g_mean /= n_spk_pair
            clsloss_d_mean /= n_spk_pair
            clsloss_g_mean /= n_spk_pair
            cycloss_mean /= n_spk_pair
            recloss_mean /= n_spk_pair

            logging.info('epoch {}, mini-batch {}: AdvLoss_d={:.4f}, AdvLoss_g={:.4f}, GradLoss_d={:.4f}, ClsLoss_d={:.4f}, ClsLoss_g={:.4f}'
                        .format(epoch, b+1, w_adv*advloss_d_mean, w_adv*advloss_g_mean, w_grad*gradloss_d_mean, w_cls*clsloss_d_mean, w_cls*clsloss_g_mean))
            logging.info('epoch {}, mini-batch {}: CycLoss={:.4f}, RecLoss={:.4f}'.format(epoch, b+1, w_cyc*cycloss_mean, w_rec*recloss_mean))
            writer.add_scalars('Loss/Total_Loss',  {'adv_loss_d': w_adv*advloss_d_mean,
                                                    'adv_loss_g': w_adv*advloss_g_mean,
                                                    'grad_loss_d': w_grad*gradloss_d_mean,
                                                    'cls_loss_d': w_cls*clsloss_d_mean,
                                                    'cls_loss_g': w_cls*clsloss_g_mean,
                                                    'cyc_loss': w_cyc*cycloss_mean,
                                                    'rec_loss': w_rec*recloss_mean}, n_iter)
            n_iter += 1
            b += 1

        if epoch % snapshot == 0:
            for tag in ['gen', 'dis']:
                print('save {} at {} epoch'.format(tag, epoch))
                torch.save({'epoch': epoch,
                            'model_state_dict': models[tag].state_dict(),
                            'optimizer_state_dict': optimizers[tag].state_dict()},
                            os.path.join(model_dir, '{}.{}.pt'.format(epoch, tag)))

    print("===================================Training Finished===================================")

def main():
    parser = argparse.ArgumentParser(description='StarGAN-VC')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-ddir', '--data_rootdir', type=str, default='./dump/arctic/norm_feat/train',
                        help='root data folder that contains the normalized features')
    parser.add_argument('--epochs', '-epoch', default=2000, type=int, help='number of epochs to learn')
    parser.add_argument('--snapshot', '-snap', default=200, type=int, help='snapshot interval')
    parser.add_argument('--batch_size', '-batch', type=int, default=12, help='Batch size')
    parser.add_argument('--num_mels', '-nm', type=int, default=80, help='number of mel channels')
    parser.add_argument('--arch_type', '-arc', default='conv', type=str, help='generator architecture type (conv or rnn)')
    parser.add_argument('--loss_type', '-los', default='wgan', type=str, help='type of adversarial loss (cgan, wgan, or lsgan)')
    parser.add_argument('--zdim', '-zd', type=int, default=16, help='dimension of bottleneck layer in generator')
    parser.add_argument('--hdim', '-hd', type=int, default=64, help='dimension of middle layers in generator')
    parser.add_argument('--mdim', '-md', type=int, default=32, help='dimension of middle layers in discriminator')
    parser.add_argument('--sdim', '-sd', type=int, default=16, help='dimension of speaker embedding')
    parser.add_argument('--lrate_g', '-lrg', default='0.0005', type=float, help='learning rate for G')
    parser.add_argument('--lrate_d', '-lrd', default='5e-6', type=float, help='learning rate for D/C')
    parser.add_argument('--gradient_clip', '-gclip', default='1.0', type=float, help='gradient clip')
    parser.add_argument('--w_adv', '-wa', default='1.0', type=float, help='Weight on adversarial loss')
    parser.add_argument('--w_grad', '-wg', default='1.0', type=float, help='Weight on gradient penalty loss')
    parser.add_argument('--w_cls', '-wcl', default='1.0', type=float, help='Weight on classification loss')
    parser.add_argument('--w_cyc', '-wcy', default='1.0', type=float, help='Weight on cycle consistency loss')
    parser.add_argument('--w_rec', '-wre', default='1.0', type=float, help='Weight on reconstruction loss')
    parser.add_argument('--normtype', '-norm', default='IN', type=str, help='normalization type: LN, BN and IN')
    parser.add_argument('--src_conditioning', '-srccon', default=0, type=int, help='w or w/o source conditioning')
    parser.add_argument('--resume', '-res', type=int, default=0, help='Checkpoint to resume training')
    parser.add_argument('--model_rootdir', '-mdir', type=str, default='./model/arctic/', help='model file directory')
    parser.add_argument('--log_dir', '-ldir', type=str, default='./logs/arctic/', help='log file directory')
    parser.add_argument('--experiment_name', '-exp', default='experiment1', type=str, help='experiment name')
    args = parser.parse_args()

    # Set up GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Configuration for StarGAN
    num_mels = args.num_mels
    arch_type = args.arch_type
    loss_type = args.loss_type
    zdim = args.zdim
    hdim = args.hdim
    mdim = args.mdim
    sdim = args.sdim
    w_adv = args.w_adv
    w_grad = args.w_grad
    w_cls = args.w_cls
    w_cyc = args.w_cyc
    w_rec = args.w_rec
    lrate_g = args.lrate_g
    lrate_d = args.lrate_d
    gradient_clip = args.gradient_clip
    epochs = args.epochs
    batch_size = args.batch_size
    snapshot = args.snapshot
    resume = args.resume
    normtype = args.normtype
    src_conditioning = bool(args.src_conditioning)

    data_rootdir = args.data_rootdir
    spk_list = sorted(os.listdir(data_rootdir))
    n_spk = len(spk_list)
    melspec_dirs = [os.path.join(data_rootdir,spk) for spk in spk_list]

    model_config = {
        'num_mels': num_mels,
        'arch_type': arch_type,
        'loss_type': loss_type,
        'zdim': zdim,
        'hdim': hdim,
        'mdim': mdim,
        'sdim': sdim,
        'w_adv': w_adv,
        'w_grad': w_grad,
        'w_cls': w_cls,
        'w_cyc': w_cyc,
        'w_rec': w_rec,
        'lrate_g': lrate_g,
        'lrate_d': lrate_d,
        'gradient_clip': gradient_clip,
        'normtype': normtype,
        'epochs': epochs,
        'BatchSize': batch_size,
        'n_spk': n_spk,
        'spk_list': spk_list,
        'src_conditioning': src_conditioning
    }

    model_dir = os.path.join(args.model_rootdir, args.experiment_name)
    makedirs_if_not_exists(model_dir)
    log_path = os.path.join(args.log_dir, args.experiment_name, 'train_{}.log'.format(args.experiment_name))
    
    # Save configuration as a json file
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as outfile:
        json.dump(model_config, outfile, indent=4)

    if arch_type=='conv':
        gen = net.Generator1(num_mels, n_spk, zdim, hdim, sdim, normtype, src_conditioning)
    elif arch_type=='rnn':
        net.Generator2(num_mels, n_spk, zdim, hdim, sdim, src_conditioning=src_conditioning)
    dis = net.Discriminator1(num_mels, n_spk, mdim, normtype)
    models = {
        'gen' : gen,
        'dis' : dis
    }
    models['stargan'] = net.StarGAN(models['gen'], models['dis'],n_spk,loss_type)

    optimizers = {
        'gen' : optim.Adam(models['gen'].parameters(), lr=lrate_g, betas=(0.9,0.999)),
        'dis' : optim.Adam(models['dis'].parameters(), lr=lrate_d, betas=(0.5,0.999))
    }

    for tag in ['gen', 'dis']:
        models[tag].to(device).train(mode=True)

    train_dataset = MultiDomain_Dataset(*melspec_dirs)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              #num_workers=os.cpu_count(),
                              drop_last=True,
                              collate_fn=collate_fn)
    Train(models, epochs, train_dataset, train_loader, optimizers, device, model_dir, log_path, model_config, snapshot, resume)


if __name__ == '__main__':
    main()
