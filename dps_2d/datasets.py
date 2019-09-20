import os
import string

import numpy as np
from PIL import Image
import torch as th
from torchvision.transforms.functional import to_tensor

from . import utils, templates


class FontsDataset(th.utils.data.Dataset):
    def __init__(self, args, val=False):
        self.args = args
        self.root = args.data
        self.files = sorted([f[:-4] for f in os.listdir(os.path.join(self.root, 'pngs')) if f.endswith('.png')])
        cutoff = int(0.9*len(self.files))
        if val:
            self.files = self.files[cutoff:]
        else:
            self.files = self.files[:cutoff]
        self.n_loops_dict = templates.n_loops

    def __repr__(self):
        return "FontsDataset | {} entries".format(len(self))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(os.path.join(self.root, 'pngs', fname + '.png')).convert('L')
        distance_fields = th.from_numpy(
                np.load(os.path.join(self.root, 'distances', fname + '.npy'))[31:-31,31:-31].astype(np.float32))
        alignment_fields = utils.compute_alignment_fields(distance_fields)
        distance_fields = distance_fields[1:-1,1:-1]
        occupancy_fields = utils.compute_occupancy_fields(th.max(distance_fields-0.01, th.zeros_like(distance_fields)))
        points = th.Tensor([])
        if self.args.chamfer is not None:
            points = th.from_numpy(np.load(os.path.join(self.root, 'points', fname + '.npy')).astype(np.float32))
            points = points[:self.args.n_samples_per_curve*sum(templates.topology)]

        return {
            'fname': fname,
            'im': to_tensor(im),
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields,
            'points': points,
            'letter_idx': string.ascii_uppercase.index(fname[0]),
            'n_loops': self.n_loops_dict[fname[0]]
        }
