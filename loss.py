import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class LossCalulcator(nn.Module):
    def __init__(self, temperature, alpha):
        super().__init__()

        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        Reference: https://github.com/peterliht/knowledge-distillation-pytorch
        """

        # Distillation Loss
        teacher_prob = nn.functional.softmax(teacher_outputs / self.temperature, dim=-1)
        student_prob = nn.functional.log_softmax(outputs / self.temperature, dim=-1)
        soft_targets_loss = self.kl_div(student_prob, teacher_prob) * (self.temperature ** 2)

        hard_targets_loss = self.ce_loss(outputs, labels)

        total_loss = self.alpha * soft_targets_loss + (1 - self.alpha) * hard_targets_loss
        return total_loss

        # Logging
        # if self.distillation_weight > 0:
        #     self.loss_log['soft-target_loss'].append(soft_target_loss.item())

        # if self.distillation_weight < 1:
        #     self.loss_log['hard-target_loss'].append(hard_target_loss.item())

        # self.loss_log['total_loss'].append(total_loss.item())

        # return total_loss

    def get_log(self, length=100):
        log = []
        # calucate the average value from lastest N losses
        for key in self.loss_log.keys():
            if len(self.loss_log[key]) < length:
                length = len(self.loss_log[key])
            log.append("%s: %2.3f"%(key, sum(self.loss_log[key][-length:]) / length))
        return ", ".join(log)
