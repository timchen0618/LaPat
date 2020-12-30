from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from dialog0.TagSampler.Trainer import Statistics
import dialog0.TagSampler.IO as IO

class SamplerLossCompute(nn.Module):
    """
    Tag Sapmler Loss Computation.
    """
    def __init__(self,output_size,padding_idx,device):
        super(SamplerLossCompute, self).__init__()
        self.padding_idx = padding_idx
        self.weight = Variable(torch.ones(output_size)).cuda()
        self.weight = self.weight.float()
        self.weight.data[padding_idx] = 0.0
        self.weight = self.weight.squeeze(0)
        self.device = device
    def compute_loss(self,output,target):

        masked_output = self.weight * output
        masked_output = masked_output.to(self.device)
        bow = torch.gather(masked_output,dim=1,index=target)
        loss = -bow.sum(-1).mean()
        loss_data = loss.data.clone()
        stats = self.stats(loss_data)
        # print(loss_data)
        return loss, stats
    def monolithic_compute_loss(self, batch, output):
        """
        Compute the loss monolithically, not dividing into shards.
        """
        target = batch.tag.transpose(0,1)
        log_probs = output

        loss, batch_stats = self.compute_loss(log_probs, target)
        loss.backward()
        return batch_stats

    def stats(self, loss):

        return Statistics(loss)


    def bottle(self, v):
        return v.view(-1, v.size(2))
