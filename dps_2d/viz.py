import cairo
from freetype import *
import numpy as np
from skimage.util.shape import view_as_windows
from svgpathtools import Line, Path, QuadraticBezier, CubicBezier, paths2svg

from opt import Opt


opt = Opt()
cmap = [tuple(int(x) for x in c.split('.')) for c in
                ['255.187.120', '255.127.14', '174.199.232', '44.160.44', '31.119.180', '255.152.150',
                 '23.190.207', '197.176.213', '152.223.138', '148.103.189', '247.182.210',
                 '227.119.194', '196.156.148', '140.86.75', '188.189.34']]


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


def draw_vec_fig(curves_set, template_indexes, letters, fonts, filename, opt, rows=2, colors=None, marked_verts=[]):
    n = len(curves_set) if curves_set is not None else len(fonts)
    if n > 1: w, h = n//rows, rows
    else: w, h = 1, 1

    surface = cairo.PDFSurface(filename, w*128, h*128)
    ctx = cairo.Context(surface)
    ctx.scale(opt.img_size, opt.img_size)
    ctx.rectangle(0, 0, w, h)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill()

    for c in range(n):
        x = c % w
        y = c // w
        if curves_set is not None:
            if fonts is not None:
                draw_glyph(fonts[c], letters[c], ctx, offset=(x, y))
            draw_curves(curves_set[c], template_indexes[c], ctx, offset=(x, y), marked_verts=marked_verts)
        elif fonts is not None:
            if colors is not None:
                draw_glyph(fonts[c], letters[c], ctx, offset=(x, y), color=colors[c])
            else:
                draw_glyph(fonts[c], letters[c], ctx, offset=(x, y), color=(0, 0, 0))


    surface.finish()

def draw_gan_fig(images, curves_set, template_indexes, filename):
    n = len(curves_set)
    w, h = 3, n

    surface = cairo.PDFSurface(filename, w*128, h*128)
    ctx = cairo.Context(surface)
    ctx.scale(opt.img_size, opt.img_size)
    ctx.rectangle(0, 0, w, h)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill()

    for c in range(n):
        y = c
        if curves_set is not None:
            ctx.save()
            im = cairo.ImageSurface.create_from_png(images[c])
            ctx.translate(0, y)
            ctx.scale(1/128, 1/128)
            ctx.set_source_surface(im)
            ctx.paint()
            ctx.restore()

            curves = curves_set[c]
            template_index = template_indexes[c]
            curves = apply_templates(curves).squeeze()
            curves = curves.reshape([-1, 6])[:opt.template_lengths[template_index]]

            x = 1
            ctx.set_line_width(0.04)
            ctx.set_source_rgb(0.1, 0.1, 0.1)
            for i in range(len(curves)):
                a, b, c = np.split(curves[i], 3)
                ctx.move_to(a[0] + x, a[1] + y)
                ctx.curve_to(a[0] * 1/3 + b[0] * 2/3 + x, a[1] * 1/3 + b[1] * 2/3 + y,
                             b[0] * 2/3 + c[0] * 1/3 + x, b[1] * 2/3 + c[1] * 1/3 + y,
                             c[0] + x, c[1] + y)
                ctx.stroke()

            ctx.set_line_width(0.02)
            for i in range(len(curves)):
                color = (x/255 for x in cmap[i % len(cmap)])
                a, b, c = np.split(curves[i], 3)
                ctx.move_to(a[0] + x, a[1] + y)
                ctx.set_source_rgb(*color)
                ctx.curve_to(a[0] * 1/3 + b[0] * 2/3 + x, a[1] * 1/3 + b[1] * 2/3 + y,
                             b[0] * 2/3 + c[0] * 1/3 + x, b[1] * 2/3 + c[1] * 1/3 + y,
                             c[0] + x, c[1] + y)
                ctx.stroke()

            for i in range(len(curves)):
                color = (x/255 for x in cmap[i % len(cmap)])
                a, b, c = np.split(curves[i], 3)
                ctx.set_source_rgb(0, 0, 0)
                ctx.arc(a[0] + x, a[1] + y, 0.02, 0, 2*np.pi)
                ctx.fill()


            x = 2
            ctx.set_source_rgb(0, 0, 0)
            template_topologies = opt.template_topologies[template_index]
            if len(template_topologies) == 1:
                loops = [curves[:template_topologies[0]]]
            else:
                loops = np.split(curves[:sum(template_topologies)],
                                 [sum(template_topologies[:i]) for i in range(1, len(template_topologies))])

            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            for loop in loops:
                a, b, c = np.split(loop[0], 3)
                ctx.move_to(a[0] + x, a[1] + y)
                for curve in loop:
                    a, b, c = np.split(curve, 3)
                    ctx.curve_to(a[0] * 1/3 + b[0] * 2/3 + x, a[1] * 1/3 + b[1] * 2/3 + y,
                                 b[0] * 2/3 + c[0] * 1/3 + x, b[1] * 2/3 + c[1] * 1/3 + y,
                                 c[0] + x, c[1] + y)
            ctx.fill()


    surface.finish()


def draw_curves(curves, template_index, ctx, offset=(0, 0), marked_verts=[]):
    x, y = offset
    curves = apply_templates(curves).squeeze()
    curves = curves.reshape([-1, 6])[:opt.template_lengths[template_index]]

    ctx.set_line_width(0.04)
    ctx.set_source_rgb(0.1, 0.1, 0.1)
    for i in range(len(curves)):
        a, b, c = np.split(curves[i], 3)
        ctx.move_to(a[0] + x, a[1] + y)
        ctx.curve_to(a[0] * 1/3 + b[0] * 2/3 + x, a[1] * 1/3 + b[1] * 2/3 + y,
                     b[0] * 2/3 + c[0] * 1/3 + x, b[1] * 2/3 + c[1] * 1/3 + y,
                     c[0] + x, c[1] + y)
        ctx.stroke()

    ctx.set_line_width(0.02)
    for i in range(len(curves)):
        color = (x/255 for x in cmap[i % len(cmap)])
        a, b, c = np.split(curves[i], 3)
        ctx.move_to(a[0] + x, a[1] + y)
        ctx.set_source_rgb(*color)
        ctx.curve_to(a[0] * 1/3 + b[0] * 2/3 + x, a[1] * 1/3 + b[1] * 2/3 + y,
                     b[0] * 2/3 + c[0] * 1/3 + x, b[1] * 2/3 + c[1] * 1/3 + y,
                     c[0] + x, c[1] + y)
        ctx.stroke()

    v = 0
    for i in range(len(curves)):
        color = (x/255 for x in cmap[i % len(cmap)])
        a, b, c = np.split(curves[i], 3)
        if tuple(a) in marked_verts:
            ctx.set_source_rgb(214/255, 39/255, 40/255)
        else:
            ctx.set_source_rgb(0, 0, 0)
        ctx.arc(a[0] + x, a[1] + y, 0.02, 0, 2*np.pi)
        ctx.fill()
        v += 3


def draw_glyph(font, char, ctx, offset=(0, 0), color=(0.6, 0.6, 0.6)):
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

    ctx.set_source_rgb(*color)
    ctx.new_path()
    ctx.set_line_width(0.02)
    x, y = offset
    for path in paths:
        ctx.move_to(path[0].bpoints()[0].real + x, path[0].bpoints()[0].imag + y)
        for seg in path:
            bpoints = seg.bpoints()
            if len(bpoints) == 2:
                ctx.line_to(bpoints[1].real + x, bpoints[1].imag + y)
            elif len(bpoints) == 3:
                ctx.curve_to(bpoints[0].real * 1/3 + bpoints[1].real * 2/3 + x,
                             bpoints[0].imag * 1/3 + bpoints[1].imag * 2/3 + y,
                             bpoints[1].real * 2/3 + bpoints[2].real * 1/3 + x,
                             bpoints[1].imag * 2/3 + bpoints[2].imag * 1/3 + y,
                             bpoints[2].real + x, bpoints[2].imag + y)
            elif len(bpoints) == 4:
                ctx.curve_to(bpoints[1].real + x, bpoints[1].imag + y,
                             bpoints[2].real + x, bpoints[2].imag + y,
                             bpoints[3].real + x, bpoints[3].imag + y)
    ctx.fill()
