import string

import numpy as np
import torch as th

from ttools.training import ModelInterface

from . import utils


class VectorizerInterface(ModelInterface):
    def __init__(self, model, lr, n_primitives, canvas_size, w_surface, w_alignment, cuda=True):
        self.model = model
        self.cuda = cuda
        self.n_primitives = n_primitives
        self.canvas_size = canvas_size
        self.w_surface = w_surface
        self.w_alignment = w_alignment
        self._step = 0

        if self.cuda:
            self.model.cuda()

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, batch):
        target_distance_fields = batch['distance_fields']
        if self.cuda:
            target_distance_fields = target_distance_fields.cuda()

        params = self.model(target_distance_fields[:,None])
        params = params.view(params.size(0), self.n_primitives, -1)
        params = th.cat([0.3*th.sigmoid(params[...,:3])+0.05,
                         params[...,3:6],
                         0.8*th.sigmoid(params[...,6:10])+0.1,
                         th.sigmoid(params[...,10:])], dim=-1)

        distance_fields = utils.compute_distance_fields(params, self.canvas_size,
                                                        df=utils.distance_to_rounded_cuboids).abs()
        distance_fields = distance_fields.min(1)[0]
        alignment_fields = utils.compute_alignment_fields(distance_fields)
        distance_fields = distance_fields[...,1:-1,1:-1,1:-1]
        occupancy_fields = utils.compute_occupancy_fields(distance_fields)

        return {
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields
        }

    def _compute_lossses(self, batch, fwd_data):
        ret = {}

        target_distance_fields = batch['distance_fields']
        target_alignment_fields = batch['alignment_fields']
        target_occupancy_fields = batch['occupancy_fields']
        if self.cuda:
            target_distance_fields = target_distance_fields.cuda()
            target_alignment_fields = target_alignment_fields.cuda()
            target_occupancy_fields = target_occupancy_fields.cuda()

        distance_fields = fwd_data['distance_fields']
        alignment_fields = fwd_data['alignment_fields']
        occupancy_fields = fwd_data['occupancy_fields']

        surfaceloss = th.mean(target_occupancy_fields*distance_fields + target_distance_fields*occupancy_fields)
        alignmentloss = th.mean(1 - th.sum(target_alignment_fields*alignment_fields, dim=-1)**2)
        ret['surfaceloss'] = surfaceloss
        ret['alignmentloss'] = alignmentloss

        loss = self.w_surface*surfaceloss + self.w_alignment*alignmentloss
        ret['loss'] = loss

        return ret

    def backward(self, batch, fwd_data):
        self.optimizer.zero_grad()

        losses_dict = self._compute_lossses(batch, fwd_data)
        loss = losses_dict['loss']

        loss.backward()
        self.optimizer.step()
        self._step += 1

        return { k: v.item() for k, v in losses_dict.items() }

    def init_validation(self):
        losses = ['loss', 'surfaceloss', 'alignmentloss']
        ret = { l: 0 for l in losses }
        ret['count'] = 0
        return ret

    def update_validation(self, batch, fwd_data, running_data):
        n = batch['distance_fields'].shape[0]
        losses_dict = self._compute_lossses(batch, fwd_data)
        loss = losses_dict['loss']
        surfaceloss = losses_dict['surfaceloss']
        alignmentloss = losses_dict['alignmentloss']
        return {
            'loss': running_data['loss'] + loss.item()*n,
            'surfaceloss': running_data['surfaceloss'] + surfaceloss.item()*n,
            'alignmentloss': running_data['alignmentloss'] + alignmentloss.item()*n,
            'count': running_data['count'] + n
        }

    def finalize_validation(self, running_data):
        losses = ['loss', 'surfaceloss', 'alignmentloss']
        return { l: running_data[l] / running_data['count'] for l in losses }