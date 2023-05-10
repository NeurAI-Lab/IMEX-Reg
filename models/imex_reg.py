import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.ecr import ECR
from copy import deepcopy
import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' IMEX-Reg')
    parser.add_argument('--ecr_weight', type=float, default=0.1,
                        help='multitask weight for rotation')
    parser.add_argument('--crl_weight', type=float, default=1,
                        help='multitask weight for rotation')
    parser.add_argument('--img_size', type=int, required=True,
                        help='Input image size')
    parser.add_argument('--reg_weight', type=float, default=0.2,
                        help='EMA regularization weight')
    parser.add_argument('--ema_update_freq', type=float, default=0.5,
                        help='EMA update frequency')
    parser.add_argument('--ema_alpha', type=float, default=0.999,
                        help='EMA alpha')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ImexReg(ContinualModel):
    NAME = 'imex_reg'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ImexReg, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.ema_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = self.args.reg_weight
        # set parameters for ema model
        self.ema_update_freq = self.args.ema_update_freq
        self.ema_alpha = self.args.ema_alpha
        self.consistency_loss = torch.nn.MSELoss(reduction='none')
        self.global_step = 0
        # Additional models
        self.addit_models = ['ema_model']

        self.ecr = ECR()

    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def compute_loss(self, inputs, not_aug_inputs, labels, buf='_'):
        outputs_m, zm, output_proj = self.net(inputs, returnt="all")
        
        # Create the augmented view for SupCon and run through the model
        inputs_x = self.CRL_transform(not_aug_inputs)
        outputs_x, zx, _ = self.net(inputs_x, returnt="all")
        
        # Compute the losses
        loss_proj_m = self.CRL_loss(zm, zx, labels=labels)
        loss_ce_m = self.loss(outputs_m, labels)
        loss_ecr = self.ecr.update(output_proj, zm.detach())

        loss = loss_ce_m + self.args.crl_weight * loss_proj_m + self.args.ecr_weight * loss_ecr

        return loss, outputs_m, zm, outputs_x, zx

    def transform_inputs(self, inputs):
        return torch.stack([self.transform(ee.cpu()) for ee in inputs]).to(self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        real_batch_size = inputs.shape[0]

        loss, _, _, _, _ = self.compute_loss(inputs, not_aug_inputs, labels)

        if not self.buffer.is_empty():
            # Load buffer data without any augmentation
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size)
            aug_buf_inputs = self.transform_inputs(buf_inputs)
            loss_m, outputs_m, zm, outputs_x, zx = self.compute_loss(aug_buf_inputs, buf_inputs, buf_labels, buf='m')
            loss += loss_m

            # Apply consistency regularization w.r.t to EMA output
            outputs_ema, z_ema, _ = self.ema_model(aug_buf_inputs, returnt="all")
            loss_cr = F.mse_loss(outputs_m, outputs_ema.detach()) + F.mse_loss(zm, z_ema.detach())
            loss += self.args.reg_weight * loss_cr

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss.item()
