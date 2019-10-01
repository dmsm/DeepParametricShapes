import math
import os
import string

import cairo
from freetype import *
import numpy as np
from svgpathtools import Line, Path, QuadraticBezier, CubicBezier, paths2svg
from skimage.util.shape import view_as_windows
from tqdm import tqdm

from opt import Opt
import util

opt = Opt()


n_points_per_unit_length = 600

def sample_points_from_font(font, char):
    try:
        face = Face('data/fonts/ttfs/{}.ttf'.format(font))
    except:
        face = Face('data/fonts/ttfs/{}.otf'.format(font))
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
    for path in paths:
        for seg in path:
            length = seg.length()
            for a in np.linspace(0, 1, num=length*n_points_per_unit_length):
                points.append(seg.point(seg.ilength(a*length)))

    return [(p.real, p.imag) for p in points]


def poly_bezier_to_bezier(x):
    x = np.reshape(x, [-1, 2])
    x = view_as_windows(x, [3, 2], step=2)
    return x.reshape([-1])


def apply_templates(curves):
    expanded_curves = []
    splits = [4*n_curves for n_curves in opt.template_topology]
    loops = np.split(curves[:4*sum(opt.template_topology)], [sum(splits[:i]) for i in range(1, len(splits))])
    expanded_loops = []
    for loop, n_curves in zip(loops, opt.template_topology):
        expanded_loops.append(poly_bezier_to_bezier(np.concatenate([loop, loop[:2]], axis=0)))
    return np.concatenate(expanded_loops, axis=0)


def sample_points_from_curves(curves, template):
    curves = apply_templates(curves).squeeze()
    curves = curves.reshape([-1, 6])[:opt.template_lengths[template]]
    segments = []
    points = []
    for curve in curves:
        segment = []
        for point in curve.reshape([-1, 2]):
            segment.append(complex(*point).conjugate())
        segments.append(QuadraticBezier(*segment))
    for seg in segments:
        length = seg.length()
        for a in np.linspace(0, 1, num=length*n_points_per_unit_length):
            points.append(seg.point(seg.ilength(a*length)))

    return [(p.real, -p.imag) for p in points]
