import argparse
import string

import cairo
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch as th
import ttools

from dps_2d import templates
from dps_2d.models import CurvesModel
from dps_2d.viz import draw_curves

def main(args):
    device = "cuda" if th.cuda.is_available() and args.cuda else "cpu"

    model = CurvesModel(sum(templates.topology))
    model.to(device)
    model.eval()
    checkpointer = ttools.Checkpointer(f'models/{args.model}', model)
    extras, _ = checkpointer.load_latest()
    if extras is not None:
        print(f"Loaded checkpoint (epoch {extras['epoch']})")
    else:
        print("Unable to load checkpoint")

    im = to_tensor(Image.open(args.image).convert('L').resize((128, 128))).to(device)
    z = th.zeros(len(string.ascii_uppercase)).scatter_(0,
            th.tensor(string.ascii_uppercase.index(args.letter)), 1).to(device)

    print(f"Processing image {args.image} (letter {args.letter})")

    curves = model(im[None], z[None])['curves'][0].detach().cpu()

    surface = cairo.PDFSurface(args.out, 128, 128)
    ctx = cairo.Context(surface)
    ctx.scale(128, 128)
    ctx.rectangle(0, 0, 1, 1)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill()

    ctx.save()
    im = cairo.ImageSurface.create_from_png(args.image)
    ctx.scale(1/128, 1/128)
    ctx.set_source_surface(im)
    ctx.paint()
    ctx.restore()

    draw_curves(curves, templates.n_loops[args.letter], ctx)

    surface.finish()

    print(f"Output saved to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("letter", type=str)
    parser.add_argument("out", type=str, default=".")
    parser.add_argument("--model", type=str, default="dps_2d")
    parser.add_argument("--cuda", dest='cuda', action='store_true')
    parser.add_argument("--no_cuda", dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    main(args)
