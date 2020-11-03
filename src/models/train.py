import argparse
import datetime
import numpy as np
import os
import subprocess
import sys
import tqdm

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
        '--time',
        default=datetime.datetime.now().strftime('%Y-%m-%d'),
        help='Name to store the log file as',
    )
    parser.add_argument('--resume', help='Path to log file to resume from')
    parser.add_argument(
        '--num_workers', type=int, default=4, help='Number of threads for data loader'
    )
    parser.add_argument(
        '--eval_only', action='store_true', help='Only run evaluation, no training'
    )
    parser.add_argument(
        '--train_only', action='store_true', help='Only run training, no evaluation'
    )
    parser.add_argument(
        "--show", action="store_true", help="Plot generated samples in Tensorboard"
    )
    parser.add_argument(
        "--show_number", type=int, default=5000, help="Plot generated samples in Tensorboard"
    )
    parser.add_argument(
        '--reduce_lr_patience', type=int, default=2, help='How many epochs of stalled validation loss before lowering learning rate'
    )
    parser.add_argument(
        '--stop_patience', type=int, default=4, help='How many epochs of stalled validation loss before early stopping'
    )
    # model params
    parser.add_argument(
        "--vae", action="store_true", help="Use a beta variational auto encoder, otherwise use vanilla auto encoder"
    )
    parser.add_argument(
        '--beta', type=float, default=0.5, help='The loss is: (1-beta)*ReconLoss + beta*KLD'
    )
    parser.add_argument(
        '--latent_size', type=int, default=32, help='Dimensionality of latent space'
    )
    parser.add_argument(
        '--encoder_width', type=int, default=64,
        help='Dimensionality of hidden layers for the encoder'
    )
    parser.add_argument(
        '--decoder_width', type=int, default=64,
        help='Dimensionality of hidden layers for the decoder'
    )
    parser.add_argument(
        '--class_pred_weight', type=float, default=1.0, help='Multiplier for class loss'
    )
    parser.add_argument(
        '--total_masked_weight', type=float, default=0.1, help='Multiplier for mse loss on masked count'
    )
    # training params
    parser.add_argument(
        '--epochs', type=int, default=40, help='Number of epochs to train with'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help='Batch size to train with'
    )

    parser.add_argument(
        '--dataset',
        choices=['chan1', 'chan2a', 'chan2b', 'chan3'],
        default='chan1',
        help='Which dataset to use',
    )
    args = parser.parse_args()
    print(args)


    ddir = '/n/home02/bostdiek/DarkMachines/data/'
    if args.dataset == 'chan1':
        print(f'Experiment on {args.dataset}')
        ffile = ddir +  'interim/chan1/background_chan1_7.79.npz'
        subprocess.call(f'rsync -v --progress {ffile} /scratch/background_chan1_7.79.npz',
                        shell=True
                        )
        Background = TrainingDataset('/scratch/background_chan1_7.79.npz',
                                     init_frac=0,
                                     final_frac=0.8)
        if not args.train_only:
            Background_eval = TrainingDataset('/scratch/background_chan1_7.79.npz',
                                              types=True,
                                              init_frac=0.8,
                                              final_frac=0.9)
    elif args.dataset == 'chan2a':
        print(f'Experiment on {args.dataset}')
        ffile = ddir +  'interim/chan2a/background_chan2a_309.6.npz'
        subprocess.call(f'rsync -v --progress {ffile} /scratch/background_chan2a_309.6.npz',
                        shell=True
                        )
        Background = TrainingDataset('/scratch/background_chan2a_309.6.npz',
                                     init_frac=0,
                                     final_frac=0.8)
        if not args.train_only:
            Background_eval = TrainingDataset('/scratch/background_chan2a_309.6.npz',
                                              types=True,
                                              init_frac=0.8,
                                              final_frac=0.9)

    elif args.dataset == 'chan2b':
        print(f'Experiment on {args.dataset}')
        ffile = ddir +  'interim/chan2b/background_chan2b_7.8.npz'
        subprocess.call(f'rsync -v --progress {ffile} /scratch/background_chan2b_7.8.npz',
                        shell=True
                        )
        Background = TrainingDataset('/scratch/background_chan2b_7.8.npz',
                                     init_frac=0,
                                     final_frac=0.8)
        if not args.train_only:
            Background_eval = TrainingDataset('/scratch/background_chan2b_7.8.npz',
                                              types=True,
                                              init_frac=0.8,
                                              final_frac=0.9)
    elif args.dataset == 'chan3':
        print(f'Experiment on {args.dataset}')
        ffile = ddir +  'interim/chan3/background_chan3_8.02.npz'
        subprocess.call(f'rsync -v --progress {ffile} /scratch/background_chan3_8.02.npz',
                        shell=True
                        )
        Background = TrainingDataset('/scratch/background_chan3_8.02.npz',
                                     init_frac=0,
                                     final_frac=0.8)
        if not args.train_only:
            Background_eval = TrainingDataset('/scratch/background_chan3_8.02.npz',
                                              types=True,
                                              init_frac=0.8,
                                              final_frac=0.9)
    else:
        sys.exit('Bad dataset')

    if args.vae:
        vae_name = f'VariationalAutoEncoderBeta-{args.beta}'
    else:
        vae_name = 'autoencoder'
    run_name = f'{args.dataset}/{args.time}-{vae_name}-{args.latent_size}-{args.encoder_width}-{args.decoder_width}-{args.class_pred_weight}-{args.total_masked_weight}'
    if args.resume:
        run_name = f'{args.resume}'

    train_writer = SummaryWriter(f'runs/{run_name}', purge_step=0)

    background_loader = DataLoader(Background,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True
                                   )
    if not args.train_only:
        background_eval = DataLoader(Background_eval,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True
                                     )

    model = make_model_with_FSPoool(args)

    model_dir = '/n/home02/bostdiek/DarkMachines/model'
    model_name = f'{model_dir}/{run_name}.tar'
    if os.path.isfile(model_name):
        model_dict_updt = torch.load(model_name,
                                     map_location=torch.device(device)
                                    )
        MyModel.load_state_dict(model_dict_updt['model_state_dict'][-1])
    if torch.cuda.device_count() > 0:
        print('...Use', torch.cuda.device_count(), 'GPUs!')
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    Scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     patience=args.reduce_lr_patience,
                                                     verbose=True,
                                                     min_lr=1e-6)
    if os.path.isfile(model_name):
        optimizer.load_state_dict(model_dict_updt['optimizer_state_dict'])
        epoch_start = model_dict_updt['epoch']
    else:
        model_dict_updt={'model_state_dict': [],
                         'TrainingLoss': [],
                         }
        epoch_start=0


    def run(net, loader, optimizer, train=False, epoch=0):
        writer = train_writer

        if train:
            net.train()
            prefix = 'train'
            torch.set_grad_enabled(True)
        else:
            net.eval()
            prefix = 'test'
            torch.set_grad_enabled(False)

        total_train_steps = args.epochs * len(loader)

        iters_per_epoch = len(loader)
        loader = tqdm.tqdm(
            loader,
            ncols=0,
            desc=f'{prefix} E{epoch:02d}'
        )

        epoch_loss = 0
        epoch_chamfer_loss = 0
        epoch_mse_loss = 0
        total_len = 0
        embedding_labels = []
        embedding_matrix = 0
        for i_batch, x in enumerate(loader, start=epoch*iters_per_epoch):
            if not train:
                x, label = x
            x = x.to(device)
            batch_len = x.shape[0]

            mask = (x[:, 0] > 0).float()
            predicted_set, (latent_mu, latent_var) = net(x, x, mask)
            vpred, cpred = predicted_set
            v_true = x[:, 1:, :]
            c_true = x[:, 0, :]

            chamfer_loss, mse_loss = MyChamfer(vpred, cpred, v_true, c_true, args)
            chamfer_loss_summed = torch.sum(chamfer_loss)
            mse_loss_summed = torch.sum(mse_loss)

            kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim = 1)
            kld_loss_summed = torch.sum(kld_loss)
            if args.vae:
                total_loss = (1 - args.beta) * (chamfer_loss_summed + mse_loss_summed) + args.beta * (kld_loss_summed)
            else:
                total_loss = chamfer_loss_summed + mse_loss_summed
            epoch_loss += total_loss.item()
            total_len += batch_len

            if train:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                writer.add_scalar('metric/train-total-loss', total_loss.item() / batch_len, i_batch)
                writer.add_scalar('metric/train-chamfer-loss', chamfer_loss_summed.item() / batch_len, i_batch)
                writer.add_scalar('metric/train-mse-loss', mse_loss_summed.item() / batch_len / args.total_masked_weight, i_batch)
                if args.vae:
                    writer.add_scalar('metric/train-KLD-loss', kld_loss_summed.item() / batch_len, i_batch)

            # Plot predictions in Tensorboard
            if args.show and not train:
                label = list([l for l in label])
                embedding_labels += label
                latent_mu = latent_mu.detach().cpu().numpy()

                if isinstance(embedding_matrix, int):
                    embedding_matrix = latent_mu
                    if args.vae:
                        recon_loss = chamfer_loss+mse_loss.detach().cpu().numpy()
                        kl_loss = kld_loss.detach().cpu().numpy()
                        loss_matrix_i = ((1 - args.beta) * recon_loss + args.beta * kl_loss)
                        loss_matrix = loss_matrix_i
                    else:
                        loss_matrix = (chamfer_loss+mse_loss).detach().cpu().numpy()
                else:
                    embedding_matrix = np.vstack([embedding_matrix, latent_mu])
                    if args.vae:
                        recon_loss = chamfer_loss+mse_loss.detach().cpu().numpy()
                        kl_loss = kld_loss.detach().cpu().numpy()
                        loss_matrix_i = ((1 - args.beta) * recon_loss + args.beta * kl_loss)
                        loss_matrix = np.hstack([loss_matrix, loss_matrix_i])
                    else:
                        loss_matrix = np.hstack([loss_matrix, (chamfer_loss+mse_loss).detach().cpu().numpy()])
                if embedding_matrix.shape[0] >= args.show_number:
                    writer.add_embedding(mat=embedding_matrix,
                                         global_step=epoch,
                                         metadata=embedding_labels
                                         )
                    writer.add_histogram(tag='TotalLoss', values=loss_matrix, global_step=epoch)
                    return 0
            if (args.dataset == 'chan3') and (i_batch % 10 == 0):
                results = {
                    'name': args.time if not args.resume else args.resume,
                    'weights': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
                    'hash': git_hash,
                    'epoch': epoch
                }
                torch.save(results, os.path.join(model_dir,
                                                 'logs',
                                                 f'{run_name}.tar'
                                                 )
                          )
            torch.cuda.empty_cache()

        loader.set_postfix(loss=f'{total_loss.item() / batch_len:.2f}')
        return  epoch_loss / total_len



    # End of the Run function


    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])

    torch.backends.cudnn.benchmark = True

    min_loss = np.inf
    stopping_patience = args.stop_patience

    for epoch in range(epoch_start+1, args.epochs):
        if (epoch == 0) and (not args.train_only) and (not args.eval_only):
            with torch.no_grad():
                run(model, background_eval, optimizer, train=False, epoch=-1)
        if not args.eval_only:
            total_loss = run(model, background_loader, optimizer, train=True, epoch=epoch)
            Scheduler.step(total_loss)

        results = {
            'name': args.time if not args.resume else args.resume,
            'weights': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
            'hash': git_hash,
            'epoch': epoch
        }
        torch.save(results, os.path.join(model_dir,
                                         'logs',
                                         f'{run_name}.tar'
                                         )
                  )
        if not args.train_only:
            run(model, background_eval, optimizer, train=False, epoch=epoch)
        if args.eval_only:
            break

        if total_loss < min_loss:
            min_loss = total_loss
            stopping_patience = args.stop_patience
        else:
            stopping_patience -= 1

        if stopping_patience == 0:
            print(f'Model has not improved in {args.stop_patience} epochs...stopping early')
            break

if __name__ == '__main__':
    main()
