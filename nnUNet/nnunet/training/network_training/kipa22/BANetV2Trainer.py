# -*- coding:utf-8 -*-
from nnunet.training.network_training.kipa22.nnUNetTrainerV2_4Fold import nnUNetTrainerV2_4Fold
from nnunet.training.data_augmentation.bdr_augmentation import get_BDR_augmentation
from nnunet.network_architecture.generic_BANetV2 import generic_BANetV2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
import numpy as np
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.boundary import gen_all_boundary_from_seg


class BANetV2Trainer_1000Epoch(nnUNetTrainerV2_4Fold):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.bdr_loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.cons_loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            self.bdr_loss = MultipleOutputLoss2(self.bdr_loss, self.ds_loss_weights)
            self.cons_loss = MultipleOutputLoss2(self.cons_loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_BDR_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = generic_BANetV2(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        bdr_target = data_dict['bdr']
        bdr_target = [(i > 0) * 1.0 for i in bdr_target]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        bdr_target = maybe_to_torch(bdr_target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            bdr_target = to_cuda(bdr_target)

        self.optimizer.zero_grad()

        # cons_cond = self.epoch > self.max_num_epochs / 2
        cons_cond = False
        if self.fp16:
            with autocast():
                output = self.network(data)
                bdr_out = self.network.get_bdr_outputs
                del data
                seg_loss = self.loss(output, target)
                bdr_loss = self.bdr_loss(bdr_out, bdr_target)

                if cons_cond:
                    output_bdr = gen_all_boundary_from_seg(output)
                    output_bdr = [(i > 0) * 1.0 for i in output_bdr]
                    output_bdr = maybe_to_torch(output_bdr)
                    if torch.cuda.is_available():
                        output_bdr = to_cuda(output_bdr)
                    cons_loss = self.cons_loss(bdr_out, output_bdr)
                    l = seg_loss + bdr_loss + cons_loss
                else:
                    l = seg_loss + bdr_loss

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            bdr_out = self.network.get_bdr_outputs
            del data
            seg_loss = self.loss(output, target)
            bdr_loss = self.bdr_loss(bdr_out, bdr_target)

            if cons_cond:
                output_bdr = gen_all_boundary_from_seg(output)
                output_bdr = maybe_to_torch(output_bdr)
                if torch.cuda.is_available():
                    output_bdr = to_cuda(output_bdr)
                cons_loss = self.cons_loss(bdr_out, output_bdr)
                l = seg_loss + bdr_loss + cons_loss
            else:
                l = seg_loss + bdr_loss

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        del bdr_target

        return seg_loss.detach().cpu().numpy()


