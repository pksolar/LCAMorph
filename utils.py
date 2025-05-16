import numpy as np
import pystrum.pynd.ndutils as nd
import torch.nn.functional as F
import torch


def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def align_img(grid, x):
    return F.grid_sample(
        x, grid=grid, mode="bilinear", padding_mode="border", align_corners=False
    )


def split_seg_global(seg, labels, downsize=1):
    full_classes = int(np.max(labels) + 1)
    valid_mask = np.isin(np.arange(full_classes), labels)
    shape = seg.shape[:-1]
    one_hot_seg = np.eye(full_classes)[seg.reshape(-1).astype(int)].reshape(*shape, -1)
    return one_hot_seg[:, ::downsize, ::downsize, ::downsize, valid_mask]


def minmax_norm(x, axis=None):
    """
    Min-max normalize array using a safe division.

    Arguments:
        x: Array to be normalized.
        axis: Dimensions to reduce during normalization. If None, all axes will be considered,
            treating the input as a single image. To normalize batches or features independently,
            exclude the respective dimensions.

    Returns:
        Normalized array.
    """
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.divide(x - x_min, x_max - x_min, out=np.zeros_like(x - x_min), where=x_max != x_min)


import copy
import itertools


def prod_n(lst):
    prod = copy.deepcopy(lst[0])
    for p in lst[1:]:
        prod *= p
    return prod


def sub2ind(siz, subs, **kwargs):
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = copy.deepcopy(subs[-1])
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx


def interpn(vol, loc, interp_method='linear'):
    if isinstance(loc, (list, tuple)):
        loc = torch.stack(loc, -1)

    nb_dims = loc.shape[-1]

    if nb_dims != len(vol.shape[:-1]):
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = torch.unsqueeze(vol, -1)

    loc = loc.type(torch.FloatTensor)

    if isinstance(vol.shape, (torch.Size,)):
        volshape = list(vol.shape)
    else:
        volshape = vol.shape

    if interp_method == "linear":
        loc0 = torch.floor(loc)

        max_loc = [d - 1 for d in list(vol.shape)]

        clipped_loc = [torch.clamp(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [torch.clamp(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.type(torch.IntTensor) for f in loc0lst], [f.type(torch.IntTensor) for f in loc1]]

        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]

        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            idx = sub2ind(vol.shape[:-1], subs)
            idx = torch.as_tensor(idx, dtype=torch.long)
            vol_val = torch.reshape(vol, (-1, volshape[-1]))[idx]

            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            wt = prod_n(wts_lst)

            wt = torch.unsqueeze(wt, -1).cuda()

            interp_vol += wt * vol_val

    else:
        assert interp_method == "nearest"
        loc = torch.round(loc)
        roundloc = loc.type(torch.IntTensor)

        max_loc = [(d - 1).type(torch.IntTensor) for d in vol.shape]
        roundloc = [torch.clamp(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = torch.reshape(vol, (-1, vol.shape[-1]))[idx]

    return interp_vol


def point_spatial_transformer(x, sdt_vol_resize=1):
    surface_points, trf = x
    trf = trf * sdt_vol_resize
    surface_pts_D = surface_points.shape[-1]
    trf_D = trf.shape[-1]
    assert surface_pts_D in [trf_D, trf_D + 1]

    if surface_pts_D == trf_D + 1:
        li_surface_pts = torch.unsqueeze(surface_points[..., -1], -1)
        surface_points = surface_points[..., :-1]

    fn = lambda x: interpn(x[0], x[1])

    diff = fn([trf, surface_points])
    # diff = x.map_(x, fn)
    ret = surface_points + diff

    if surface_pts_D == trf_D + 1:
        ret = torch.cat((ret, li_surface_pts), -1)
    return ret
