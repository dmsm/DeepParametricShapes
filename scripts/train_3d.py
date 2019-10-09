import datetime
import os
import logging

import torch as th
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import ttools

from dps_3d import datasets
from dps_3d.interfaces import VectorizerInterface
from dps_3d.models import PrimsModel


LOG = logging.getLogger(__name__)

th.manual_seed(123)
th.backends.cudnn.deterministic = True
np.random.seed(123)


def _worker_init_fn(worker_id):
    np.random.seed(worker_id)


def main(args):
    data = datasets.ShapenetDataset(args.data, args.canvas_size)
    dataloader = DataLoader(data, batch_size=args.bs, num_workers=args.num_worker_threads,
                            worker_init_fn=_worker_init_fn, shuffle=True, drop_last=True)
    LOG.info(data)

    val_data = datasets.ShapenetDataset(args.data, args.canvas_size, val=True)
    val_dataloader = DataLoader(val_data)

    model = PrimsModel(output_dim=(11 if args.rounded else 10)*args.n_primitives)

    checkpointer = ttools.Checkpointer(args.checkpoint_dir, model)
    checkpointer.load_latest()

    interface = VectorizerInterface(model, args.lr, args.n_primitives, args.canvas_size, args.w_surface,
                                    args.w_alignment, args.csg, args.rounded, cuda=args.cuda)

    keys = ['loss', 'surfaceloss', 'alignmentloss']

    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                        datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')), flush_secs=1)
    val_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                            datetime.datetime.now().strftime('val-%m%d%y-%H%M%S')), flush_secs=1)

    trainer = ttools.Trainer(interface)
    trainer.add_callback(ttools.callbacks.TensorBoardLoggingCallback(keys=keys, writer=writer,
                                                                     val_writer=val_writer, frequency=5))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer, max_files=1))
    trainer.train(dataloader, num_epochs=args.num_epochs, val_dataloader=val_dataloader)


if __name__ == '__main__':
    parser = ttools.BasicArgumentParser()
    parser.add_argument("--w_surface", type=float, default=1)
    parser.add_argument("--w_alignment", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--canvas_size", type=int, default=64)
    parser.add_argument("--n_primitives", type=int, default=16)
    parser.add_argument("--csg", default=False, dest='csg', action='store_true')
    parser.add_argument("--rounded", default=False, dest='rounded', action='store_true')
    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
