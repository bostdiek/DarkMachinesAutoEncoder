import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, npz_file, types=False, init_frac=0, final_frac=1):
        """
        Args:
            npz_file (string): Path to data.
        """
        with np.load(npz_file) as f:
            len_f = f['Objects'].shape[0]
            start_i, end_i = int(init_frac * len_f), int(final_frac * len_f)
            self.data = f['Objects'][start_i:end_i]
            if types:
                self.types = f['Procs'][start_i:end_i]
            else:
                self.types = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[[idx]]
        sample = sample.transpose(0, 2, 1)

        # scale Energy
        sample[:, 1, :] = np.log10(sample[:, 1, :] / 1000, # MeV -> GeV
                                   out=np.zeros_like(sample[:, 1, :]),
                                   where=(sample[:, 1, :] != 0)
                                   )
        # scale Transverse Momentum
        sample[:, 2, :] = np.log10(sample[:, 2, :] / 1000, # MeV -> GeV
                                   out=np.zeros_like(sample[:, 2, :]),
                                   where=(sample[:, 2, :] != 0)
                                   )

        sample = torch.from_numpy(sample[0])

        if self.types is not None:
            return sample.float(), self.types[idx]
        return sample.float()

def ChamferLossWithClassifierAndCount(vpred, cpred, v_true, c_true, loss_args):
    """
    Computes the loss between the input set and the output set
    Arguments:
        vpred: pytorch tensor of the vector predictions
        cpred: pytorch tensor of the class predictions
        v_true: pytorch tensor of the true four vectors
        c_true: pytorch tensor of the true classes
        loss_args: dictionary with the following keys
            class_pred_weight: multiplier for the loss to get the class correct
            total_masked_weight: multiplier for mean squared error loss to get
                the correct number of predicted particles
    Returns:
        loss: pytorch tensor with the loss for each event
    """
    sm = nn.LogSoftmax(dim=0)
    sm2 = nn.LogSoftmax(dim=1)

    class_pred_weight = loss_args.class_pred_weight
    total_masked_weight = loss_args.total_masked_weight

    chamfer_loss = torch.zeros(len(vpred), dtype=torch.float)
    mse_loss = torch.zeros(len(vpred), dtype=torch.float)
    for i in range(len(vpred)):
        VPRED = vpred[i].transpose(0, 1).contiguous()
        CPRED = cpred[i].transpose(0, 1)

        VTRUE = v_true[i].transpose(0, 1).contiguous()
        CTRUE = c_true[i]


        dist = torch.cdist(VPRED, VTRUE[CTRUE > 0])
        # distance for each true vector to nearest predicted vector
        d1, ind1 = torch.min(dist, 0)
        inds = CTRUE[CTRUE>0].long()
        # add in the classifier prediction loss
        softmaxes = sm(CPRED)[ind1]
        d1 -= class_pred_weight * softmaxes[torch.arange(len(inds)),inds] # negative for the log
        d1sum = torch.sum(d1)

        # distance for each predicted vector to nearest real vector
        d2, ind2 = torch.min(dist, 1)
        softmaxes = sm2(CPRED)
        #  only compute the loss for the particles predicted to not be masked
        non_masked = torch.argmax(softmaxes, 1) > 0
        class_distances = softmaxes[torch.arange(len(ind2)), CTRUE[ind2].long()]
        d2sum = torch.sum((d2 - class_pred_weight * class_distances)[non_masked])

        # mean squared error for number of each node type?
        mv, inds_max = torch.max(softmaxes, 1)
        count_loss = torch.sum(torch.tensor([torch.square(torch.sum(inds_max==cval) - torch.sum(CTRUE==cval))
                                             for cval in range(1, 9)],
                                            dtype=torch.float)
                              )
        chamfer_loss[i] = d1sum + d2sum
        mse_loss[i] = total_masked_weight * count_loss

    return chamfer_loss, mse_loss
