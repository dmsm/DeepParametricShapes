import cairo
import numpy as np
import torch as th
from torchvision.transforms.functional import to_tensor
import ttools.callbacks as cb

from . import viz


class InputImageCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'input'

    def visualized_image(self, batch, fwd_result):
        return batch['im']


class RenderingCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'rendering'

    def visualized_image(self, batch, fwd_result):
        return fwd_result['occupancy_fields'].unsqueeze(1)


class CurvesCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'curves'

    def visualized_image(self, batch, fwd_result):
        with th.no_grad():
            curves = fwd_result['curves'].cpu()
            n_loops = batch['n_loops'].cpu()
        data = []
        for curve, n_loop in zip(curves, n_loops):
            surface = cairo.ImageSurface(cairo.Format.RGB24, 128, 128)
            ctx = cairo.Context(surface)
            ctx.scale(128, 128)
            ctx.rectangle(0, 0, 1, 1)
            ctx.set_source_rgb(1, 1, 1)
            ctx.fill()
            viz.draw_curves(curve, n_loop, ctx)

            buf = surface.get_data()
            data.append(to_tensor(np.frombuffer(buf, np.uint8).reshape(128, 128, 4))[:3])
        return th.stack(data)
