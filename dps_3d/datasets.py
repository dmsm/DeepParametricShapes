import os
import string

import numpy as np
from PIL import Image
import torch as th
from torchvision.transforms.functional import to_tensor

from . import utils


class ShapenetDataset(th.utils.data.Dataset):
    def __init__(self, args, val=False):
        self.args = args
        self.root = args.data
        self.files = sorted([f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))])
        cutoff = int(0.9*len(self.files))
        if val:
            self.files = self.files[cutoff:]
        else:
            self.files = self.files[:cutoff]

    def __repr__(self):
        return "ShapenetDataset | {} entries".format(len(self))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        distance_fields = th.from_numpy(np.load(os.path.join(self.root, fname, 'df.npy')).astype(np.float32))
        alignment_fields = th.from_numpy(np.load(os.path.join(self.root, fname, 'af.npy')).astype(np.float32))

        # jitter
        x, y, z = np.random.randint(distance_fields.size(0)-self.args.canvas_size+1, size=3)
        distance_fields = distance_fields[x:x+self.args.canvas_size,y:y+self.args.canvas_size,z:z+self.args.canvas_size]
        alignment_fields = \
            alignment_fields[x:x+self.args.canvas_size,y:y+self.args.canvas_size,z:z+self.args.canvas_size]

        occupancy_fields = utils.compute_occupancy_fields(th.max(distance_fields-0.02, th.zeros_like(distance_fields)),
                                                          eps=1e-6)

        return {
            'fname': fname,
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields,
        }
