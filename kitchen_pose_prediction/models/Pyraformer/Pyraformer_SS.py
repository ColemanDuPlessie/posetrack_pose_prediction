# -*- coding: utf-8 -*-
"""
For those who, like myself, wonder what SS stands for, it stands for Single
Step. The only modifications we have made to the Pyraformer are to slightly
tweak it (primarily in the embedding) to work with a multivariate time series
(instead of a univariate time series with covariates) and to initialize it
with separate arguments, instead of passing them all in as 'opt', although that
is just a quality-of-life change. All the logic remains true to the original.
"""
from argparse import Namespace
import torch
import torch.nn as nn
from .Layers import EncoderLayer, Predictor
from .Layers import Bottleneck_Construct
from .Layers import get_mask, refer_points, get_k_q, get_q_k
from .embed import SingleStepEmbedding


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.window_size = opt.window_size
        self.num_heads = opt.n_head
        self.mask, self.all_size = get_mask(opt.input_size, opt.window_size, opt.inner_size, opt.device)
        self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        if opt.use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            q_k_mask = get_q_k(opt.input_size, opt.inner_size, opt.window_size[0], opt.device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in range(opt.n_layer)
                ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False) for i in range(opt.n_layer)
                ])

        self.embedding = SingleStepEmbedding(opt.d_input, opt.num_seq, opt.d_model, opt.input_size, opt.device)

        self.conv_layers = Bottleneck_Construct(opt.d_model, opt.window_size, opt.d_k)

    def forward(self, sequence):

        seq_enc = self.embedding(sequence)
        mask = self.mask.repeat(len(seq_enc), self.num_heads, 1, 1).to(sequence.device)

        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        all_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        return all_enc


class Model(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.encoder = Encoder(opt)

        # convert hidden vectors into two scalar
        self.mean_hidden = Predictor(4 * opt.d_model, opt.d_input) # The Predictor layers are literally just wrappers of Linear layers.
        self.var_hidden = Predictor(4 * opt.d_model, opt.d_input) # I don't know why they multiply d_model by 4. Maybe it's supposed to be the number of heads?

        self.softplus = nn.Softplus()

    def forward(self, data):
        enc_output = self.encoder(data)

        mean_pre = self.mean_hidden(enc_output)
        var_hid = self.var_hidden(enc_output)
        var_pre = self.softplus(var_hid)
        mean_pre = self.softplus(mean_pre)

        return mean_pre.squeeze(2), var_pre.squeeze(2)

    def test(self, data, v):
        mu, sigma = self(data)

        sample_mu = mu[:, -1] * v
        sample_sigma = sigma[:, -1] * v
        return sample_mu, sample_sigma

def pyraformer_params(d_input=153, d_model=512, window_size=[4, 4, 4], n_head=4, n_layer=4, batch_size=1024, inner_size=3, device='cuda', use_tvm=False, inner_hid=512, d_k=128, d_v=128, dropout=0.1): # TODO I don't know if batch_size is actually the batch size or if it's something else.
    return Namespace(d_input=d_input, d_model=d_model, window_size=window_size, n_head=n_head, n_layer=n_layer, input_size=batch_size, inner_size=inner_size, device=device, use_tvm=use_tvm, d_inner_hid=inner_hid, d_k=d_k, d_v=d_v, num_seq=d_input, dropout=dropout)