import datetime
import os
import logging

import torch as th
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ttools

from dps_2d import callbacks, datasets, templates
from dps_2d.interfaces import VectorizerInterface
from dps_2d.models import CurvesModel


LOG = logging.getLogger(__name__)

th.manual_seed(123)
th.backends.cudnn.deterministic = True
np.random.seed(123)


def _worker_init_fn(worker_id):
    np.random.seed(worker_id)


def main(args):
    data = datasets.FontsDataset(args.data, args.chamfer, args.n_samples_per_curve)
    dataloader = DataLoader(data, batch_size=args.bs, num_workers=args.num_worker_threads,
                            worker_init_fn=_worker_init_fn, shuffle=True, drop_last=True)
    LOG.info(data)

    val_data = datasets.FontsDataset(args.data, args.chamfer, args.n_samples_per_curve, val=True)
    val_dataloader = DataLoader(val_data)

    model = CurvesModel(n_curves=sum(templates.topology))

    checkpointer = ttools.Checkpointer(args.checkpoint_dir, model)
    checkpointer.load_latest()

    interface = VectorizerInterface(model, args.simple_templates, args.lr, args.max_stroke, args.canvas_size,
                                    args.chamfer, args.n_samples_per_curve, args.w_surface, args.w_template,
                                    args.w_alignment, cuda=args.cuda)

    keys = ['loss', 'chamferloss', 'templateloss'] if args.chamfer \
        else ['loss', 'surfaceloss', 'alignmentloss', 'templateloss']

    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                        datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')), flush_secs=1)
    val_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                            datetime.datetime.now().strftime('val-%m%d%y-%H%M%S')), flush_secs=1)

    trainer = ttools.Trainer(interface)
    trainer.add_callback(ttools.callbacks.TensorBoardLoggingCallback(keys=keys, writer=writer,
                                                                     val_writer=val_writer, frequency=5))
    trainer.add_callback(callbacks.InputImageCallback(writer=writer, val_writer=val_writer, frequency=100))
    trainer.add_callback(callbacks.CurvesCallback(writer=writer, val_writer=val_writer, frequency=100))
    if not args.chamfer:
        trainer.add_callback(callbacks.RenderingCallback(writer=writer, val_writer=val_writer, frequency=100))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer, max_files=1))
    trainer.train(dataloader, num_epochs=args.num_epochs, val_dataloader=val_dataloader)


if __name__ == '__main__':
    parser = ttools.BasicArgumentParser()
    parser.add_argument("--w_surface", type=float, default=1)
    parser.add_argument("--w_alignment", type=float, default=0.01)
    parser.add_argument("--w_template", type=float, default=10)
    parser.add_argument("--eps", type=float, default=0.04)
    parser.add_argument("--max_stroke", type=float, default=0.04)
    parser.add_argument("--canvas_size", type=int, default=128)
    parser.add_argument("--n_samples_per_curve", type=int, default=120)
    parser.add_argument("--chamfer", default=False, dest='chamfer', action='store_true')
    parser.add_argument("--simple_templates", default=False, dest='simple_templates', action='store_true')
    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
