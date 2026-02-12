"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import jittor as jt
import contextlib

from os.path import join

# Note: Jittor doesn't need explicit device specification
# It automatically handles CUDA if available

# Note: Jittor doesn't need explicit device specification
# It automatically handles CUDA if available

# Simplified Interp1d for Jittor - using numpy-based implementation
# For production, consider using jittor's native interpolation functions
def interp1d_jittor(x, y, xnew):
    """
    Simplified 1D interpolation for Jittor.
    For complex cases, you may need to implement custom interpolation.
    """
    # Convert to numpy for interpolation, then back to jittor
    if isinstance(x, jt.Var):
        x = x.numpy()
    if isinstance(y, jt.Var):
        y = y.numpy()
    if isinstance(xnew, jt.Var):
        xnew_np = xnew.numpy()
    else:
        xnew_np = xnew
    
    from scipy import interpolate
    f = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
    ynew = f(xnew_np)
    
    return jt.array(ynew)

class Interp1d:
    """Wrapper class for interpolation compatibility"""
    def __call__(self, x, y, xnew, out=None):
        return interp1d_jittor(x, y, xnew)

def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    N, C, _, _ = bayer_images.shape
    outs = bayer_images * wbs.view(N, C, 1, 1)
    return outs

def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(
        0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    outs = jt.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = jt.clamp(images, 1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    outs = jt.clamp((outs*255).int(), 0, 255).float() / 255
    return outs


def gamma_compression_grad(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = jt.clamp(images, 1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    return outs


def binning(bayer_images):
    """RGBG -> RGB"""
    lin_rgb = jt.stack([
        bayer_images[:,0,...],
        jt.mean(bayer_images[:, [1,3], ...], dim=1),
        bayer_images[:,2,...]], dim=1)

    return lin_rgb


def process(bayer_images, wbs, cam2rgbs, gamma=2.2, CRF=None):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    orig_img = bayer_images
    # White balance.
    bayer_images = apply_gains(orig_img, wbs)
    # Binning
    bayer_images = jt.clamp(bayer_images, 0.0, 1.0)
    images = binning(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = jt.clamp(images, 0.0, 1.0)
    if CRF is None:
        images = gamma_compression(images, gamma)
    else:
        images = camera_response_function(images, CRF)

    return images


def process_grad(bayer_images, wbs, cam2rgbs, gamma=2.2, CRF=None):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    orig_img = bayer_images
    # White balance.
    bayer_images = apply_gains(orig_img, wbs)
    # Binning
    bayer_images = jt.clamp(bayer_images, 0.0, 1.0)
    images = binning(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = jt.clamp(images, 0.0, 1.0)
    if CRF is None:
        images = gamma_compression_grad(images, gamma)
    else:
        images = camera_response_function_grad(images, CRF)

    return images


def camera_response_function(images, CRF):
    E, fs = CRF # unpack CRF data

    outs = torch.zeros_like(images)
    device = images.device

    for i in range(images.shape[0]):
        img = images[i].view(3, -1)
        out = Interp1d()(E.to(device), fs.to(device), img)
        outs[i, ...] = out.view(3, images.shape[2], images.shape[3])

    outs = jt.clamp((outs*255).int(), 0, 255).float() / 255
    return outs

def camera_response_function_grad(images, CRF):
    E, fs = CRF # unpack CRF data

    outs = torch.zeros_like(images)
    device = images.device

    for i in range(images.shape[0]):
        img = images[i].view(3, -1)
        out = Interp1d()(E.to(device), fs.to(device), img)
        outs[i, ...] = out.view(3, images.shape[2], images.shape[3])

    return outs

def raw2rgb(packed_raw, raw, CRF=None, gamma=2.2):
    """Raw2RGB pipeline (preprocess version)"""
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    if isinstance(packed_raw, np.ndarray):
        packed_raw = jt.array(packed_raw).float()

    wb = jt.array(wb).float()
    cam2rgb = jt.array(cam2rgb).float()

    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=gamma, CRF=CRF)[0, ...].numpy()

    return out


def raw2rgb_v2(packed_raw, wb, ccm, CRF=None, gamma=2.2): # RGBG
    packed_raw = jt.array(packed_raw).float()
    wb = jt.array(wb).float()
    cam2rgb = jt.array(ccm).float()
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=gamma, CRF=CRF)[0, ...].numpy()
    return out


def raw2rgb_torch(packed_raw, wb, ccm, CRF=None, gamma=2.2, batch=False): # RGBG
    if batch:
        out = process(packed_raw, wbs=wb, cam2rgbs=ccm, gamma=gamma, CRF=CRF)
    else:
        out = process(packed_raw[None], wbs=wb[None], cam2rgbs=ccm[None], gamma=gamma, CRF=CRF)
    return out

def raw2rgb_torch_grad(packed_raw, wb, ccm, CRF=None, gamma=2.2): # RGBG
    out = process_grad(packed_raw, wbs=wb, cam2rgbs=ccm, gamma=gamma, CRF=CRF)
    return out

def raw2rgb_postprocess(packed_raw, raw, CRF=None):
    """Raw2RGB pipeline (postprocess version)"""
    assert packed_raw.ndimension() == 4 and packed_raw.shape[0] == 1
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    wb = jt.array(wb[None]).float()
    cam2rgb = jt.array(cam2rgb[None]).float()
    out = process(packed_raw, wbs=wb, cam2rgbs=cam2rgb, gamma=2.2, CRF=CRF)
    return out

def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.rgb_camera_matrix[:3, :3].astype(np.float32)
    return wb, ccm


def read_emor(address):
    def _read_curve(lst):
        curve = [l.strip() for l in lst]
        curve = ' '.join(curve)
        curve = np.array(curve.split()).astype(np.float32)
        return curve

    with open(address) as f:
        lines = f.readlines()
        k = 1
        E = _read_curve(lines[k:k+256])
        k += 257
        f0 = _read_curve(lines[k:k+256])
        hs = []
        for _ in range(25):
            k += 257
            hs.append(_read_curve(lines[k:k+256]))

        hs = np.array(hs)

        return E, f0, hs


def read_dorf(address):
    with open(address) as f:
        lines = f.readlines()
        curve_names = lines[0::6]
        Es = lines[3::6]
        Bs = lines[5::6]

        Es = [np.array(E.strip().split()).astype(np.float32) for E in Es]
        Bs = [np.array(B.strip().split()).astype(np.float32) for B in Bs]

    return curve_names, Es, Bs


def load_CRF(EMoR_path):
    # init CRF function
    fs = np.loadtxt(join(EMoR_path, 'CRF_SonyA7S2_5.txt'))
    E, _, _ = read_emor(join(EMoR_path, 'emor.txt'))
    E = jt.array(E).repeat(3, 1)
    fs = torch.from_numpy(fs)
    CRF = (E, fs)
    return CRF
