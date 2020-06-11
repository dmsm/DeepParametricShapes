import string

import numpy as np
import torch as th
from ttools.training import ModelInterface

from . import utils, templates


class VectorizerInterface(ModelInterface):
    def __init__(self, model, simple_templates, lr, max_stroke, canvas_size, chamfer, n_samples_per_curve, w_surface,
                 w_template, w_alignment, cuda=True):
        self.model = model
        self.simple_templates = simple_templates
        self.max_stroke = max_stroke
        self.canvas_size = canvas_size
        self.chamfer = chamfer
        self.n_samples_per_curve = n_samples_per_curve
        self.w_surface = w_surface
        self.w_template = w_template
        self.w_alignment = w_alignment
        self.cuda = cuda
        self._step = 0

        self.curve_templates = th.Tensor(templates.simple_templates if self.simple_templates
                else templates.letter_templates)

        if self.cuda:
            self.model.cuda()
            self.curve_templates = self.curve_templates.cuda()

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, batch):
        im = batch['im']
        n_loops = batch['n_loops']
        letter_idx = batch['letter_idx']
        if self.cuda:
            im = im.cuda()
            n_loops = n_loops.cuda()
            letter_idx = letter_idx.cuda()

        z = im.new_zeros(im.size(0), len(string.ascii_uppercase)).scatter_(1, letter_idx[:,None], 1)
        out = self.model(im, z)
        curves = out['curves']

        if not self.chamfer:
            strokes = out['strokes'] * self.max_stroke
            distance_fields = utils.compute_distance_fields(curves, n_loops, templates.topology, self.canvas_size)
            distance_fields = th.max(distance_fields-strokes[...,None,None], th.zeros_like(distance_fields)).min(1)[0]
            distance_fields = distance_fields ** 2
            alignment_fields = utils.compute_alignment_fields(distance_fields)
            distance_fields = distance_fields[...,1:-1,1:-1]
            occupancy_fields = utils.compute_occupancy_fields(distance_fields)

            ret = {
                'curves': curves,
                'distance_fields': distance_fields,
                'alignment_fields': alignment_fields,
                'occupancy_fields': occupancy_fields,
            }
        else:
            ret = { 'curves': curves }
        return ret

    def _compute_lossses(self, batch, fwd_data):
        ret = {}

        if not self.chamfer:
            target_distance_fields = batch['distance_fields']
            target_alignment_fields = batch['alignment_fields']
            target_occupancy_fields = batch['occupancy_fields']
        else:
            target_points = batch['points']
        letter_idx = batch['letter_idx']
        n_loops = batch['n_loops']
        if self.cuda:
            if not self.chamfer:
                target_distance_fields = target_distance_fields.cuda()
                target_alignment_fields = target_alignment_fields.cuda()
                target_occupancy_fields = target_occupancy_fields.cuda()
            else:
                target_points = target_points.cuda()
            letter_idx = letter_idx.cuda()
            n_loops = n_loops.cuda()

        loss = 0
        curves = fwd_data['curves']
        if not self.chamfer:
            distance_fields = fwd_data['distance_fields']
            alignment_fields = fwd_data['alignment_fields']
            occupancy_fields = fwd_data['occupancy_fields']

            surfaceloss = th.mean(target_occupancy_fields*distance_fields + target_distance_fields*occupancy_fields)
            alignmentloss = th.mean(1 - th.sum(target_alignment_fields*alignment_fields, dim=-1)**2)
            ret['surfaceloss'] = surfaceloss
            ret['alignmentloss'] = alignmentloss
            loss += self.w_surface*surfaceloss + self.w_alignment*alignmentloss
        else:
            chamferloss = utils.compute_chamfer_distance(
                    utils.sample_points_from_curves(curves, n_loops, templates.topology, self.n_samples_per_curve),
                    target_points)
            ret['chamferloss'] = chamferloss
            loss += chamferloss

        templateloss = 0
        b = curves.size(0)
        curve_templates = self.curve_templates.index_select(0, n_loops-1 if self.simple_templates else letter_idx)
        template_loops = th.split(curve_templates.view(b, -1, 2), [2*n for n in templates.topology], dim=1)
        loops = th.split(curves.view(b, -1, 2), [2*n for n in templates.topology], dim=1)
        for i, (template_loop, loop) in enumerate(zip(template_loops, loops)):
            idxs = (n_loops>i).nonzero().squeeze()
            if idxs.numel() == 0:
               break
            templateloss += th.mean((loop.index_select(0, idxs) - template_loop.index_select(0, idxs)) ** 2)
        ret['templateloss'] = templateloss

        w_template = self.w_template*np.exp(-max(self._step-1500, 0)/500)
        loss += w_template*templateloss
        ret['loss'] = loss

        return ret

    def training_step(self, batch):
        self.model.train()
        fwd_data = self.forward(batch)

        self.optimizer.zero_grad()

        losses_dict = self._compute_lossses(batch, fwd_data)
        loss = losses_dict['loss']

        loss.backward()
        self.optimizer.step()
        self._step += 1

        fwd_data.update({ k: v.item() for k, v in losses_dict.items() })
        return fwd_data

    def init_validation(self):
        losses = ['loss', 'chamferloss', 'templateloss'] if self.chamfer \
            else ['loss', 'surfaceloss', 'alignmentloss', 'templateloss']
        ret = { l: 0 for l in losses }
        ret['count'] = 0
        return ret

    def validation_step(self, batch, running_data):
        self.model.eval()
        n = batch['im'].shape[0]
        count = running_data['count']

        fwd_data = self.forward(batch)
        losses_dict = self._compute_lossses(batch, fwd_data)
        loss = losses_dict['loss']
        templateloss = losses_dict['templateloss']

        ret = {
            'loss': (running_data['loss']*count + loss.item()*n) / (count+n),
            'templateloss': (running_data['templateloss']*count + templateloss.item()*n) / (count+n),
        }
        if not self.chamfer:
            surfaceloss = losses_dict['surfaceloss']
            alignmentloss = losses_dict['alignmentloss']
            ret['surfaceloss'] = (running_data['surfaceloss']*count + surfaceloss.item()*n) / (count+n)
            ret['alignmentloss'] = (running_data['alignmentloss']*count + alignmentloss.item()*n) / (count+n)
        else:
            chamferloss = losses_dict['chamferloss']
            ret['chamferloss'] = (running_data['chamferloss']*count + chamferloss.item()*n) / (count+n)

        ret['count'] = count+n
        ret = ret.update(fwd_data)
        return ret
