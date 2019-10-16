import os

import numpy as np
import torch as th

from . import utils


class ShapenetDataset(th.utils.data.Dataset):
    def __init__(self, root, canvas_size,  val=False, jitter=True):
        self.jitter = jitter
        self.root = root
        self.canvas_size = canvas_size
        self.files = sorted([f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))])
        np.random.shuffle(self.files)
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

        if self.jitter:
            x, y, z = np.random.randint(distance_fields.size(0)-self.canvas_size+1, size=3)
            distance_fields = distance_fields[x:x+self.canvas_size,y:y+self.canvas_size,z:z+self.canvas_size]
            alignment_fields = \
                alignment_fields[x:x+self.canvas_size,y:y+self.canvas_size,z:z+self.canvas_size]
        else:
            distance_fields = distance_fields[5:-5,5:-5,5:-5]
            alignment_fields = alignment_fields[5:-5,5:-5,5:-5]

        occupancy_fields = utils.compute_occupancy_fields(distance_fields)

        return {
            'fname': fname,
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields,
        }
