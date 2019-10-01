import argparse
import math
from multiprocessing import Pool
import os
import string

import cairo
from freetype import *
import numpy as np
from svgpathtools import Line, Path, QuadraticBezier, CubicBezier, paths2svg
from tqdm import tqdm

from opt import Opt
import util

opt = Opt()

def process_letter(font_char):
    try:
        font, char = font_char
        name = "{}_{}".format(char, os.path.splitext(font)[0])
        face = Face('data/fonts/ttfs/{}'.format(font))
        face.set_char_size(48*64)
        face.load_char(char)
        outline = face.glyph.outline
        contours = [-1] + outline.contours
        segment = []
        segments = []
        paths = []
        for i in range(len(outline.points)):
            segment.append(complex(*outline.points[i]).conjugate())
            tag = int(bin(outline.tags[i])[2])
            try:
                j = contours.index(i)
                if tag == 0:
                    segment.append(complex(*outline.points[contours[j-1]+1]).conjugate())
                    tag = 2
                else:
                    tag = 3
            except ValueError:
                pass
            if tag > 0:
                if len(segment) == 1:
                    pass
                elif len(segment) == 2:
                    segments.append(Line(*segment))
                elif len(segment) == 3:
                    segments.append(QuadraticBezier(*segment))
                elif len(segment) == 4:
                    segments.append(CubicBezier(*segment))
                else:
                    for k in range(len(segment)-1):
                        A, C = segment[k:k+2]
                        B = (A+C) / 2
                        segments.append(QuadraticBezier(A, B, C))

                if tag == 1:
                    segment = [complex(*outline.points[i]).conjugate()]
                elif tag == 2:
                    paths.append(Path(*segments))
                    segments = []
                    segment = []
                else:
                    segments.append(Line(segment[-1], complex(*outline.points[contours[j-1]+1]).conjugate()))
                    paths.append(Path(*segments))
                    segments = []
                    segment = []

        xmin, xmax, ymin, ymax = paths2svg.big_bounding_box(paths)
        factor = 0.8 / max(xmax-xmin, ymax-ymin)
        for i, path in enumerate(paths):
            paths[i] = path.translated(complex(-xmin, -ymin)).scaled(factor)
        xmin, xmax, ymin, ymax = paths2svg.big_bounding_box(paths)
        xmargin = (1 - (xmax-xmin)) / 2
        ymargin = (1 - (ymax-ymin)) / 2
        for i, path in enumerate(paths):
            paths[i] = path.translated(complex(xmargin, ymargin))

        points = []
        surface = cairo.ImageSurface(cairo.Format.RGB24, opt.img_size, opt.img_size)
        ctx = cairo.Context(surface)
        ctx.scale(opt.img_size, opt.img_size)
        ctx.set_source_rgba(1, 1, 1)
        ctx.rectangle(0, 0, 1, 1)
        ctx.fill()

        ctx.set_source_rgb(0, 0, 0)
        ctx.new_path()
        ctx.set_line_width(0.02)
        for path in paths:
            ctx.move_to(path[0].bpoints()[0].real, path[0].bpoints()[0].imag)
            for seg in path:
                bpoints = seg.bpoints()
                if len(bpoints) == 2:
                    ctx.line_to(bpoints[1].real, bpoints[1].imag)
                elif len(bpoints) == 3:
                    ctx.curve_to(bpoints[0].real * 1/3 + bpoints[1].real * 2/3,
                                 bpoints[0].imag * 1/3 + bpoints[1].imag * 2/3,
                                 bpoints[1].real * 2/3 + bpoints[2].real * 1/3,
                                 bpoints[1].imag * 2/3 + bpoints[2].imag * 1/3,
                                 bpoints[2].real, bpoints[2].imag)
                elif len(bpoints) == 4:
                    ctx.curve_to(bpoints[1].real, bpoints[1].imag,
                                 bpoints[2].real, bpoints[2].imag,
                                 bpoints[3].real, bpoints[3].imag)
            for t in np.linspace(0, 1, num=opt.n_points_sampled//len(paths)+1):
                points.append(path.point(t))

        n_points = len(points)
        points = np.array(points, dtype=np.complex64).view(np.float32).reshape([-1, 2])
        np.random.shuffle(points)
        np.save('data/fonts/points/{}.npy'.format(name), points)

        grid = np.mgrid[-0.25:1.25:opt.img_size*1.5j, -0.25:1.25:opt.img_size*1.5j].T[:,:,None,:]
        distances = np.empty((grid.shape[0], grid.shape[1]))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[0]):
                distances[i,j] = np.amin(np.linalg.norm(grid[i,j]-points, axis=1))
        if not np.isnan(distances).any():
            np.save('data/fonts/distances/{}.npy'.format(name), distances)
            surface.write_to_png('data/fonts/pngs/{}.png'.format(name))
        else:
            return e
    except Exception as e:
        return e

    return None


fonts = [f for f in os.listdir('data/fonts/ttfs/') if f.endswith('.otf') or f.endswith('.ttf')]
to_process = [(font, letter) for letter in string.ascii_uppercase for font in fonts]

to_remove = []
with Pool() as workers, tqdm(total=len(to_process)) as pbar:
    for result in tqdm(workers.imap_unordered(process_letter, to_process)):
        if result is not None:
            print(result)
        pbar.update()
