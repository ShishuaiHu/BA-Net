import torch
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
from scipy.ndimage import distance_transform_edt
from skimage import segmentation as skimage_seg
import numpy as np
from nnunet.training.loss_functions.dice_loss import RobustCrossEntropyLoss, softmax_helper, \
    SoftDiceLoss, SoftDiceLossSquared


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):  # channel
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary == 1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf


class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        net_output = softmax_helper(net_output)
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
            gt_sdf = compute_sdf(y_onehot.cpu().numpy(), net_output.shape)

        phi = torch.from_numpy(gt_sdf)
        if phi.device != net_output.device:
            phi = phi.to(net_output.device).type(torch.float32)
        # pred = net_output[:, 1:, ...].type(torch.float32)
        # phi = phi[:,1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", net_output[:, 1:, ...], phi[:, 1:, ...])
        bd_loss = multipled.mean()

        return bd_loss


class DC_and_CE_and_BD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        super(DC_and_CE_and_BD_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.bd = BDLoss()

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, current_epoch=0, max_epoch=1000):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        bd_loss = 0
        flag_epoch = (max_epoch // 8 * 7)
        if current_epoch < flag_epoch:
            weight = 1
        else:
            bd_loss = self.bd(net_output, target)
            weight = 1 - (current_epoch - flag_epoch) / (max_epoch - flag_epoch)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = weight * (self.weight_ce * ce_loss + self.weight_dice * dc_loss) + (1 - weight) * bd_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

    #####################################################


def compute_gt_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in ground gruth.
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):  # class; exclude the background class
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance_transform_edt(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


def compute_pred_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in prediction.
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):  # class; exclude the background class
            posmask = img_gt[b][c] > 0.5
            if posmask.any():
                posdis = distance_transform_edt(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


class HDLoss(nn.Module):
    def __init__(self):
        """
        compute haudorff loss for binary segmentation
        https://arxiv.org/pdf/1904.10030v1.pdf
        """
        super(HDLoss, self).__init__()

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, c, x,y,z)
        target: ground truth, shape: (batch_size, c, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        # print('hd loss.py', net_output.shape, y_onehot.shape)

        with torch.no_grad():
            pc_dist = compute_pred_dtm(net_output.cpu().numpy(), net_output.shape)
            gt_dist = compute_gt_dtm(y_onehot.cpu().numpy(), net_output.shape)
            dist = pc_dist ** 2 + gt_dist ** 2  # \alpha=2 in eq(8)
            # print('pc_dist.shape: ', pc_dist.shape, 'gt_dist.shape', gt_dist.shape)

        pred_error = (net_output - y_onehot) ** 2

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pred_error[:, 1:, ...], dist[:, 1:, ...])
        hd_loss = multipled.mean()

        return hd_loss


class DC_and_CE_and_HD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        super(DC_and_CE_and_HD_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.hd = HDLoss()

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, current_epoch=0, max_epoch=1000):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        hd_loss = 0
        alpha = 0
        flag_epoch = (max_epoch // 8 * 7)
        if current_epoch < flag_epoch:
            weight = 1
        else:
            hd_loss = self.hd(net_output, target)
            weight = 0
            with torch.no_grad():
                alpha = hd_loss / (dc_loss + 1 + 1e-5)

        if self.aggregate == "sum":
            result = weight * (self.weight_ce * ce_loss + self.weight_dice * dc_loss) + (1 - weight) * (alpha * (dc_loss + 1) + hd_loss)
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result
