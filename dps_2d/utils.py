import numpy as np
import torch as th


def make_safe(x):
    """Set small entries to one to avoid div by zero."""
    return th.where(x.abs() < 1e-6, th.ones_like(x), x)


def distance_to_curves(source_points, curves):
    """Compute the distance from each source point to each quaratic Bezier curve.

    source_points -- [n_points, 2]
    curves -- [..., 3, 2]
    """
    p0, p1, p2 = th.split(curves, 1, dim=-2)  # [..., 1, 2]

    X = p1 - p0  # [..., 1, 2]
    Y = p2 - p1 - X  # [..., 1, 2]
    Z = p0 - source_points  # [..., n_points, 2]

    a = th.sum(Y*Y, dim=-1)  # [..., 1]
    a = make_safe(a)

    b = 3*th.sum(X*Y, dim=-1)  # [..., 1]
    c = 2*th.sum(X*X, dim=-1) + th.sum(Y*Z, dim=-1)  # [..., n_points]
    d = th.sum(X*Z, dim=-1)  # [..., n_points]

    A = b/a
    B = c/a
    C = d/a

    Q = (A**2 - 3*B) / 9
    sqrt_Q = th.sqrt(Q.abs() + 1e-6)
    sqrt_Q = make_safe(sqrt_Q)
    R = (2*A**3 - 9*A*B + 27*C) / 54

    theta = th.acos(th.clamp(R / sqrt_Q**3, -1+1e-6, 1-1e-6))
    t1 = -2 * sqrt_Q * th.cos(theta/3) - A/3
    t2 = -2 * sqrt_Q * th.cos((theta+2*np.pi)/3) - A/3
    t3 = -2 * sqrt_Q * th.cos((theta+4*np.pi)/3) - A/3

    alpha = -R.sign() * (R.abs() + th.sqrt(th.abs(R**2-Q**3) + 1e-6)) ** (1/3)
    alpha = make_safe(alpha)
    beta = Q/alpha

    t4 = alpha + beta - A/3
    c = make_safe(c)
    t5 = -d/c

    ts = th.stack([t1, t2, t3, t4, t5], dim=-1)  # [..., n_points, 5]
    ts = th.clamp(ts, 1e-6, 1)

    ts = ts[...,None].pow(ts.new_tensor([0, 1, 2]))  # [..., n_points, 5, 3]

    A = ts.new_tensor([[1., 0, 0],
                       [-2, 2, 0],
                       [1, -2, 1]])
    points = ts @ A @ curves.unsqueeze(-3)  # [..., n_points, 5, 2]

    sizes = [-1] * (points.dim() - 3) + [points.size(-3), -1, -1]
    endpoints = th.cat([p0, p2], dim=-2).unsqueeze(-3).expand(*sizes)  # [..., n_points, 2, 2]
    points = th.cat([points, endpoints], dim=-2)  # [..., n_points, 7, 2]

    distances, _ = th.min(th.sqrt(th.sum((points-source_points[:,None,:])**2, dim=-1) + 1e-6), dim=-1) # [..., n_points]

    return distances


def unroll_curves(curves, topology):
    """Unroll curve parameters into loops as defined by the topology.

    curves -- [b, 2*max_n_curves, 2]
    topology -- [n_loops] list of curves per loop (should sum to max_n_curves)
    """
    b = curves.size(0)
    curves = curves.view(b, -1, 2)
    loops = th.split(curves, [2*n for n in topology], dim=1)
    unrolled_loops = []
    for loop in loops:
        loop = th.cat([loop, loop[:,0,None,:]], dim=1)
        loop = loop.unfold(1, 3, 2).permute(0, 1, 3, 2).view(b, -1, 3, 2)
        unrolled_loops.append(loop)
    return unrolled_loops  # n_loops x [b, n_curves, 3, 2]


def compute_distance_fields(curves, n_loops, topology, canvas_size):
    """Compute distance fields of size (canvas_size+2)^2. Distances corresponding to unused curves are set to 10.

    curves -- [b, 2*max_n_curves, 2]
    n_loops -- [b] number of loops per batch example
    topology -- [n_loops] list of curves per loop (should sum to max_n_curves)
    canvas_size
    """
    grid_pts = th.stack(th.meshgrid([th.linspace(-1/(canvas_size-1), 1+1/(canvas_size-1), canvas_size+2)]*2),
                        dim=-1).permute(1, 0, 2).reshape(-1, 2).type(curves.dtype)
    if curves.is_cuda:
        grid_pts = grid_pts.cuda()

    loops = unroll_curves(curves, topology)
    distance_fields = []
    for i, loop in enumerate(loops):
        idxs = (n_loops>i).nonzero().squeeze()
        n_curves = loop.size(1)
        padded_distances = 10*loop.new_ones(loop.size(0), n_curves, canvas_size+2, canvas_size+2)
        if idxs.numel() > 0:
            distances = distance_to_curves(grid_pts, loop.index_select(0, idxs)).view(-1, n_curves,
                                                                                      canvas_size+2, canvas_size+2)
            padded_distances[idxs] = distances
        distance_fields.append(padded_distances)

    return th.cat(distance_fields, dim=1)


def compute_alignment_fields(distance_fields):
    """Compute alignment unit vector fields from distance fields."""
    dx = distance_fields[...,2:,1:-1] - distance_fields[...,:-2,1:-1]
    dy = distance_fields[...,1:-1,2:] - distance_fields[...,1:-1,:-2]
    alignment_fields = th.stack([dx, dy], dim=-1)
    return alignment_fields / th.sqrt(th.sum(alignment_fields**2, dim=-1, keepdims=True) + 1e-6)


def compute_occupancy_fields(distance_fields, eps=0.04):
    """Compute smooth occupancy fields from distance fields."""
    occupancy_fields = 1 - th.clamp(distance_fields / eps, 0, 1)
    return occupancy_fields**2 * (3 - 2*occupancy_fields)


def sample_points_from_curves(curves, n_loops, topology, n_samples_per_curve):
    """Sample points from Bezier curves.

    curves -- [b, 2*max_n_curves, 2]
    n_loops -- [b] number of loops per batch example
    topology -- [n_loops] list of curves per loop (should sum to max_n_curves)
    n_samples_per_curve
    """
    A = curves.new_tensor([[1., 0, 0],
                           [-2, 2, 0],
                           [1, -2, 1]])

    loops = unroll_curves(curves, topology)
    all_points = th.empty(curves.size(0), 0, 2)
    if curves.is_cuda:
        all_points = all_points.cuda()
    for i, loop in enumerate(loops):
        idxs = (n_loops>i).nonzero().squeeze()
        loop = loop.index_select(0, idxs)  # [?, n_curves, 3, 2]
        n_curves = loop.size(1)

        ts = th.empty(n_curves, n_samples_per_curve).uniform_(0, 1)
        ts = ts[...,None].pow(ts.new_tensor([0, 1, 2]))  # [n_points, 3]
        if curves.is_cuda:
            ts = ts.cuda()

        points = ts @ A @ loop  # [?, n_curves, n_points, 2]
        points = points.view(-1, n_samples_per_curve*n_curves, 2)

        if i > 0:
            pad_idxs = th.randperm(all_points.size(1))[:n_samples_per_curve*n_curves]
            padded_points = all_points[:,pad_idxs]

            padded_points[idxs] = points
        else:
            padded_points = points
        all_points = th.cat([all_points, padded_points], dim=1)

    return all_points


def compute_chamfer_distance(a, b):
    """Compute Chamfer distance between two point sets.

    a -- [b, n, 2]
    b -- [b, m, 2]
    """
    D = th.sqrt(th.sum((a.unsqueeze(1) - b.unsqueeze(2))**2, dim=-1) + 1e-6)  # [b, m, n]
    return th.mean(th.sum(D.min(1)[0], dim=1) / a.size(1) + th.sum(D.min(2)[0], dim=1) / b.size(1))
