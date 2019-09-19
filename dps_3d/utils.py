import numpy as np
import torch as th


def hamilton_product(q1, q2, dim=-1):
    a1, b1, c1, d1 = th.split(q1, 1, dim=dim)
    a2, b2, c2, d2 = th.split(q2, 1, dim=dim)
    return th.cat([a1*a2-b1*b2-c1*c2-d1*d2,
                   a1*b2+b1*a2+c1*d2-d1*c2,
                   a1*c2-b1*d2+c1*a2+d1*b2,
                   a1*d2+b1*c2-c1*b2+d1*a2], dim=dim)


def distance_to_cuboids(source_points, params):
    """Compute the distance from each source point to each cuboid.

    source_points -- [n_points, 3]
    params -- [b, 10]
    """
    b, q, T = th.split(params, [3, 4, 3], dim=-1)
    q = q / th.sqrt(th.sum(q**2, dim=-1, keepdims=True) + 1e-6)

    p = source_points[:,None,None]

    p = p - T

    p = th.cat([p.new_zeros(*p.size()[:-1], 1), p], dim=-1)
    q_conj = q.new_tensor([1, -1, -1, -1]) * q
    p = hamilton_product(hamilton_product(q, p), q_conj)[...,1:]

    d = p.abs() - b
    length = th.sqrt(th.sum(th.max(d, th.zeros_like(d))**2, dim=-1) + 1e-6)
    max_d, _ = th.max(d, dim=-1)

    sdf = length + th.min(max_d, th.zeros_like(max_d))

    return sdf.permute(1, 2, 0)


def distance_to_rounded_cuboids(source_points, params):
    """Compute the distance from each source point to each rouned cuboid.

    source_points -- [n_points, 3]
    params -- [b, 11]
    """
    b, q, T, r = th.split(params, [3, 4, 3, 1], dim=-1)
    b = b - r
    q = q / th.sqrt(th.sum(q**2, dim=-1, keepdims=True) + 1e-6)

    p = source_points[:,None,None]

    p = p - T

    p = th.cat([p.new_zeros(*p.size()[:-1], 1), p], dim=-1)
    q_conj = q.new_tensor([1, -1, -1, -1]) * q
    p = hamilton_product(hamilton_product(q, p), q_conj)[...,1:]

    d = p.abs() - b
    length = th.sqrt(th.sum(th.max(d, th.zeros_like(d))**2, dim=-1) + 1e-6)
    max_d, _ = th.max(d, dim=-1)

    sdf = length + th.min(max_d, th.zeros_like(max_d)) - r.squeeze(-1)

    return sdf.permute(1, 2, 0)


def distance_to_spheres(source_points, params):
    """Compute the distance from each source point to each sphere.

    source_points -- [n_points, 3]
    params -- [b, 4]
    """
    c, r = th.split(params, [3, 1], dim=-1)
    p = source_points[:,None,None]
    p = p - c
    sdf = th.sqrt(th.sum(p**2, dim=-1) + 1e-6)
    sdf = sdf - r.squeeze(-1)

    return sdf.permute(1, 2, 0)


def compute_distance_fields(params, canvas_size, df=distance_to_cuboids):
    """Compute distance fields of size (canvas_size+2)^3 to specified primitives.

    params -- [b, *]
    canvas_size
    df -- distance_to_*
    """
    grid_pts = th.stack(th.meshgrid([th.linspace(-1/(canvas_size-1),
                                                 1+1/(canvas_size-1),
                                                 canvas_size+2)]*3), dim=-1).permute(0, 2, 1, 3).reshape(-1, 3)
    if params.is_cuda:
        grid_pts = grid_pts.cuda()

    return df(grid_pts, params).view(params.size(0), params.size(1), 66, 66, 66)


def compute_alignment_fields(distance_fields):
    """Compute alignment unit vector fields from distance fields."""
    dx = distance_fields[...,2:,1:-1,1:-1] - distance_fields[...,:-2,1:-1,1:-1]
    dy = distance_fields[...,1:-1,2:,1:-1] - distance_fields[...,1:-1,:-2,1:-1]
    dz = distance_fields[...,1:-1,1:-1,2:] - distance_fields[...,1:-1,1:-1,:-2]
    alignment_fields = th.stack([dx, dy, dz], dim=-1)
    return alignment_fields / th.sqrt(th.sum(alignment_fields**2, dim=-1, keepdims=True) + 1e-6)


def compute_occupancy_fields(distance_fields, eps=0.03):
    """Compute smooth occupancy fields from distance fields."""
    distance_fields = th.abs(distance_fields)
    occupancy_fields = 1 - th.clamp(distance_fields / eps, 0, 1)
    return occupancy_fields**2 * (3 - 2*occupancy_fields)
