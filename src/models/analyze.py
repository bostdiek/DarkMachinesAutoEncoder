import argparse
import datetime
import numpy as np
import os
import pickle
import subprocess
import sys
import tqdm
from types import SimpleNamespace

import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import tensorboard as tb
import tensorflow as tf
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


from models import make_model_with_FSPoool
from utilities import TrainingDataset
from utilities import ChamferLossWithClassifierAndCount as MyChamfer

plt.rcParams.update({'font.family': 'cmr10',
                     'font.size': 12,
                     'axes.unicode_minus': False,
                     'axes.labelsize': 12,
                     'axes.labelsize': 12,
                     'figure.figsize': (4, 4),
                     'figure.dpi': 80,
                     'mathtext.fontset': 'cm',
                     'mathtext.rm': 'serif',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True
                     })

todaysDate = datetime.date.today();
print(f'Experiment run on {todaysDate}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running experiment on the {device}')

obj_convert = {
    1: 'jet',
    2: 'b-jet',
    3: 'e+',
    4: 'e-',
    5: 'm+',
    6: 'm-',
    7: 'gamma',
    8: 'met',
}

def main():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        '--name',
        default=datetime.datetime.now().strftime('%Y-%m-%d'),
        help='Name to store the log file as',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4, help='Number of threads for data loader'
    )

    args = parser.parse_args()
    print(args)

    dataset, model_args = args.name.split('/')
    print(dataset, model_args)
    aetype = model_args.split('-')[3]

    ddir = '/n/home02/bostdiek/DarkMachines/data/'
    if dataset == 'chan1':
        print(f'Experiment on {dataset}')
        ffile = ddir +  'interim/chan1/*.npz'
        if not os.path.isdir(f'/scratch/bostdiek/{dataset}'):
            os.makedirs(f'/scratch/bostdiek/{dataset}')
        subprocess.call(f'rsync -v --progress {ffile} /scratch/bostdiek/{dataset}',
                        shell=True
                        )

    elif dataset == 'chan2a':
        print(f'Experiment on {dataset}')
        ffile = ddir +  'interim/chan2a/*.npz'
        if not os.path.isdir(f'/scratch/bostdiek/{dataset}'):
            os.makedirs(f'/scratch/bostdiek/{dataset}')
        subprocess.call(f'rsync -v --progress {ffile} /scratch/bostdiek/{dataset}',
                        shell=True
                        )
    elif dataset == 'chan2b':
        print(f'Experiment on {dataset}')
        ffile = ddir +  'interim/chan2b/*.npz'
        if not os.path.isdir(f'/scratch/bostdiek/{dataset}'):
            os.makedirs(f'/scratch/bostdiek/{dataset}')
        subprocess.call(f'rsync -v --progress {ffile} /scratch/bostdiek/{dataset}',
                        shell=True
                        )
    elif dataset == 'chan3':
        ffile = ddir +  'interim/chan3/*.npz'
        if not os.path.isdir(f'/scratch/bostdiek/{dataset}'):
            os.makedirs(f'/scratch/bostdiek/{dataset}')
        subprocess.call(f'rsync -v --progress {ffile} /scratch/bostdiek/{dataset}',
                        shell=True
                        )
    else:
        sys.exit('Bad dataset')


    model_dir = '/n/home02/bostdiek/DarkMachines/model/logs'
    model_name = f'{model_dir}/{dataset}/{model_args}'

    model_dict_updt = torch.load(model_name,
                                 map_location=torch.device(device)
                                 )
    print('args')
    args = model_dict_updt['args']
    args = SimpleNamespace(**args)
    print(args)

    model = make_model_with_FSPoool(args)
    print(model)

    model.load_state_dict(model_dict_updt['weights'])

    if torch.cuda.device_count() > 0:
        print('...Use', torch.cuda.device_count(), 'GPUs!')
        model = nn.DataParallel(model)

    model.to(device)

    def run(net, loader):
        net.eval()
        prefix = 'test'
        torch.set_grad_enabled(False)

        iters_per_epoch = len(loader)
        loader = tqdm.tqdm(
            loader,
            ncols=0,
            desc=f'{prefix} batch{iters_per_epoch:02d}'
        )

        loss_matrix = torch.zeros(len(loader) * args.batch_size)
        latent_matrix = torch.zeros([len(loader) * args.batch_size, args.latent_size])
        embedding_labels = []
        for i_batch, x in enumerate(loader):
            x, label = x
            x = x.to(device)
            batch_len = x.shape[0]

            mask = (x[:, 0] > 0).float()
            predicted_set, (latent_mu, latent_var) = net(x, x, mask)

            latent_matrix[(i_batch * batch_len):(i_batch * batch_len) + batch_len] = latent_mu
            vpred, cpred = predicted_set
            v_true = x[:, 1:, :]
            c_true = x[:, 0, :]

            chamfer_loss, mse_loss = MyChamfer(vpred, cpred, v_true, c_true, args)

            kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim = 1)

            kld_loss = kld_loss.to('cpu')

            if args.vae:
                total_loss = (1 - args.beta) * (chamfer_loss + mse_loss) + args.beta * (kld_loss)
            else:
                total_loss = chamfer_loss + mse_loss

            loss_matrix[(i_batch * batch_len):(i_batch * batch_len) + batch_len] = total_loss

            label = list([l for l in label])
            embedding_labels += label

        return  loss_matrix, latent_matrix, embedding_labels
    # End of the Run function

    datafiles = os.listdir(f'/scratch/bostdiek/{dataset}')
    print(datafiles)
    torch.backends.cudnn.benchmark = True

    results_dict = {}
    for datafile in datafiles:
        print(f'Evaluating {datafile}')
        tmp_dict = {}
        dataset_np = TrainingDataset(f'/scratch/bostdiek/{dataset}/{datafile}',
                                     types=True,
                                     init_frac=0.9 if 'background' in datafile else 0,
                                     final_frac=1)

        data_loader = DataLoader(dataset_np,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers
                                 )


        loss_matrix, latent_matrix, embedding_labels = run(model, data_loader)

        tmp_dict['loss_matrix'] = tmp_dict['loss_matrix'] = loss_matrix.detach().cpu().numpy()[:len(embedding_labels)]
        tmp_dict['latent_matrix'] = latent_matrix.detach().cpu().numpy()[:len(embedding_labels)]
        tmp_dict['embedding_labels'] = embedding_labels
        results_dict[datafile] = tmp_dict

    savedir = os.path.join(ddir, 'processed', dataset)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    saveloc = os.path.join(savedir, model_args[:-4] + '.pkl')
    with open(saveloc, 'wb') as f:
        pickle.dump(results_dict, f)



if __name__ == '__main__':
    main()
