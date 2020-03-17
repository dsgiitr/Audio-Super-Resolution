import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def avg_sqrt_l2_loss(X, Y):
	sqrt_l2_loss = torch.sqrt(torch.mean((X-Y)**2 + 1e-6, dim=[1,2]))
	sqrn_l2_norm = torch.sqrt(torch.mean((Y**2), dim=[1,2]))
	snr = 20 * torch.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / tf.log(10.)

	avg_sqrt_l2 = torch.mean(sqrt_l2_loss, dim=0)
	avg_snr = torch.mean(snr, dim=0)

	return avg_sqrt_l2, avg_snr
