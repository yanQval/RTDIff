
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc


def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))

def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)

def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
        **kwargs)

def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)

def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1],
        **kwargs)

def rotate2d(theta, **kwargs):
    return matrix(
        [torch.cos(theta), torch.sin(-theta), 0],
        [torch.sin(theta), torch.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)

def rotate3d(v, theta, **kwargs):
    vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
    s = torch.sin(theta); c = torch.cos(theta); cc = 1 - c
    return matrix(
        [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
        [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
        [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
        [0,             0,             0,             1],
        **kwargs)

def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)

def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)

def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)

class Augment:
    def __init__(self, p=1,
        xflip=0, yflip=0, rotate_int=0, translate_int=0, translate_int_max=0.125,
        scale=0, rotate_frac=0, aniso=0, translate_frac=0, scale_std=0.2, rotate_frac_max=1, aniso_std=0.2, aniso_rotate_prob=0.5, translate_frac_std=0.125,
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1,
    ):
        super().__init__()
        self.p                  = float(p)                  # Overall multiplier for augmentation probability.

        # Pixel blitting.
        self.xflip              = float(xflip)              # Probability multiplier for x-flip.
        self.yflip              = float(yflip)              # Probability multiplier for y-flip.
        self.rotate_int         = float(rotate_int)         # Probability multiplier for integer rotation.
        self.translate_int      = float(translate_int)      # Probability multiplier for integer translation.
        self.translate_int_max  = float(translate_int_max)  # Range of integer translation, relative to image dimensions.

        # Geometric transformations.
        self.scale              = float(scale)              # Probability multiplier for isotropic scaling.
        self.rotate_frac        = float(rotate_frac)        # Probability multiplier for fractional rotation.
        self.aniso              = float(aniso)              # Probability multiplier for anisotropic scaling.
        self.translate_frac     = float(translate_frac)     # Probability multiplier for fractional translation.
        self.scale_std          = float(scale_std)          # Log2 standard deviation of isotropic scaling.
        self.rotate_frac_max    = float(rotate_frac_max)    # Range of fractional rotation, 1 = full circle.
        self.aniso_std          = float(aniso_std)          # Log2 standard deviation of anisotropic scaling.
        self.aniso_rotate_prob  = float(aniso_rotate_prob)  # Probability of doing anisotropic scaling w.r.t. rotated coordinate frame.
        self.translate_frac_std = float(translate_frac_std) # Standard deviation of frational translation, relative to image dimensions.

        # Color transformations.
        self.brightness         = float(brightness)         # Probability multiplier for brightness.
        self.contrast           = float(contrast)           # Probability multiplier for contrast.
        self.lumaflip           = float(lumaflip)           # Probability multiplier for luma flip.
        self.hue                = float(hue)                # Probability multiplier for hue rotation.
        self.saturation         = float(saturation)         # Probability multiplier for saturation.
        self.brightness_std     = float(brightness_std)     # Standard deviation of brightness.
        self.contrast_std       = float(contrast_std)       # Log2 standard deviation of contrast.
        self.hue_max            = float(hue_max)            # Range of hue rotation, 1 = full circle.
        self.saturation_std     = float(saturation_std)     # Log2 standard deviation of saturation.

    def __call__(self, images):
        N, C, H, W = images.shape
        device = images.device
        labels = [torch.zeros([images.shape[0], 0], device=device)]


        if self.xflip > 0:
            w = torch.randint(2, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1, 1], device=device) < self.xflip * self.p, w, torch.zeros_like(w))
            images = torch.where(w == 1, images.flip(3), images)
            labels += [w]

        if self.yflip > 0:
            w = torch.randint(2, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1, 1], device=device) < self.yflip * self.p, w, torch.zeros_like(w))
            images = torch.where(w == 1, images.flip(2), images)
            labels += [w]

        if self.rotate_int > 0:
            w = torch.randint(4, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1, 1], device=device) < self.rotate_int * self.p, w, torch.zeros_like(w))
            images = torch.where((w == 1) | (w == 2), images.flip(3), images)
            images = torch.where((w == 2) | (w == 3), images.flip(2), images)
            images = torch.where((w == 1) | (w == 3), images.transpose(2, 3), images)
            labels += [(w == 1) | (w == 2), (w == 2) | (w == 3)]

        if self.translate_int > 0:
            w = torch.rand([2, N, 1, 1, 1], device=device) * 2 - 1
            w = torch.where(torch.rand([1, N, 1, 1, 1], device=device) < self.translate_int * self.p, w, torch.zeros_like(w))
            tx = w[0].mul(W * self.translate_int_max).round().to(torch.int64)
            ty = w[1].mul(H * self.translate_int_max).round().to(torch.int64)
            b, c, y, x = torch.meshgrid(*(torch.arange(x, device=device) for x in images.shape), indexing='ij')
            x = W - 1 - (W - 1 - (x - tx) % (W * 2 - 2)).abs()
            y = H - 1 - (H - 1 - (y + ty) % (H * 2 - 2)).abs()
            images = images.flatten()[(((b * C) + c) * H + y) * W + x]
            labels += [tx.div(W * self.translate_int_max), ty.div(H * self.translate_int_max)]

        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        if self.scale > 0:
            w = torch.randn([N], device=device)
            w = torch.where(torch.rand([N], device=device) < self.scale * self.p, w, torch.zeros_like(w))
            s = w.mul(self.scale_std).exp2()
            G_inv = G_inv @ scale2d_inv(s, s)
            labels += [w]

        if self.rotate_frac > 0:
            w = (torch.rand([N], device=device) * 2 - 1) * (np.pi * self.rotate_frac_max)
            w = torch.where(torch.rand([N], device=device) < self.rotate_frac * self.p, w, torch.zeros_like(w))
            G_inv = G_inv @ rotate2d_inv(-w)
            labels += [w.cos() - 1, w.sin()]

        if self.aniso > 0:
            w = torch.randn([N], device=device)
            r = (torch.rand([N], device=device) * 2 - 1) * np.pi
            w = torch.where(torch.rand([N], device=device) < self.aniso * self.p, w, torch.zeros_like(w))
            r = torch.where(torch.rand([N], device=device) < self.aniso_rotate_prob, r, torch.zeros_like(r))
            s = w.mul(self.aniso_std).exp2()
            G_inv = G_inv @ rotate2d_inv(r) @ scale2d_inv(s, 1 / s) @ rotate2d_inv(-r)
            labels += [w * r.cos(), w * r.sin()]

        if self.translate_frac > 0:
            w = torch.randn([2, N], device=device)
            w = torch.where(torch.rand([1, N], device=device) < self.translate_frac * self.p, w, torch.zeros_like(w))
            G_inv = G_inv @ translate2d_inv(w[0].mul(W * self.translate_frac_std), w[1].mul(H * self.translate_frac_std))
            labels += [w[0], w[1]]


        if G_inv is not I_3:
            cx = (W - 1) / 2
            cy = (H - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device) # [idx, xyz]
            cp = G_inv @ cp.t() # [batch, xyz, idx]
            Hz = np.asarray(wavelets['sym6'], dtype=np.float32)
            Hz_pad = len(Hz) // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(misc.constant([0, 0] * 2, device=device))
            margin = margin.min(misc.constant([W - 1, H - 1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            conv_weight = misc.constant(Hz[None, None, ::-1], dtype=images.dtype, device=images.device).tile([images.shape[1], 1, 1])
            conv_pad = (len(Hz) + 1) // 2
            images = torch.stack([images, torch.zeros_like(images)], dim=4).reshape(N, C, images.shape[2], -1)[:, :, :, :-1]
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(2), groups=images.shape[1], padding=[0,conv_pad])
            images = torch.stack([images, torch.zeros_like(images)], dim=3).reshape(N, C, -1, images.shape[3])[:, :, :-1, :]
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(3), groups=images.shape[1], padding=[conv_pad,0])
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)

            shape = [N, C, (H + Hz_pad * 2) * 2, (W + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = torch.nn.functional.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            conv_weight = misc.constant(Hz[None, None, :], dtype=images.dtype, device=images.device).tile([images.shape[1], 1, 1])
            conv_pad = (len(Hz) - 1) // 2
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(2), groups=images.shape[1], stride=[1,2], padding=[0,conv_pad])[:, :, :, Hz_pad : -Hz_pad]
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(3), groups=images.shape[1], stride=[2,1], padding=[conv_pad,0])[:, :, Hz_pad : -Hz_pad, :]


        I_4 = torch.eye(4, device=device)
        M = I_4
        luma_axis = misc.constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device)

        if self.brightness > 0:
            w = torch.randn([N], device=device)
            w = torch.where(torch.rand([N], device=device) < self.brightness * self.p, w, torch.zeros_like(w))
            b = w * self.brightness_std
            M = translate3d(b, b, b) @ M
            labels += [w]

        if self.contrast > 0:
            w = torch.randn([N], device=device)
            w = torch.where(torch.rand([N], device=device) < self.contrast * self.p, w, torch.zeros_like(w))
            c = w.mul(self.contrast_std).exp2()
            M = scale3d(c, c, c) @ M
            labels += [w]

        if self.lumaflip > 0:
            w = torch.randint(2, [N, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1], device=device) < self.lumaflip * self.p, w, torch.zeros_like(w))
            M = (I_4 - 2 * luma_axis.ger(luma_axis) * w) @ M
            labels += [w]

        if self.hue > 0:
            w = (torch.rand([N], device=device) * 2 - 1) * (np.pi * self.hue_max)
            w = torch.where(torch.rand([N], device=device) < self.hue * self.p, w, torch.zeros_like(w))
            M = rotate3d(luma_axis, w) @ M
            labels += [w.cos() - 1, w.sin()]

        if self.saturation > 0:
            w = torch.randn([N, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1], device=device) < self.saturation * self.p, w, torch.zeros_like(w))
            M = (luma_axis.ger(luma_axis) + (I_4 - luma_axis.ger(luma_axis)) * w.mul(self.saturation_std).exp2()) @ M
            labels += [w]


        if M is not I_4:
            images = images.reshape([N, C, H * W])
            if C == 3:
                images = M[:, :3, :3] @ images + M[:, :3, 3:]
            elif C == 1:
                M = M[:, :3, :].mean(dim=1, keepdims=True)
                images = images * M[:, :, :3].sum(dim=2, keepdims=True) + M[:, :, 3:]
            else:
                raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
            images = images.reshape([N, C, H, W])

        labels = torch.cat([x.to(torch.float32).reshape(N, -1) for x in labels], dim=1)
        return images, labels
