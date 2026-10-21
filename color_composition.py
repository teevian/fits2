"""Display-only RGB/LRGB composition; calibrated masters remain unchanged.

References: https://docs.astropy.org/en/stable/visualization/rgb.html
https://arxiv.org/abs/astro-ph/0312483
"""
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_erosion
from scipy.optimize import least_squares

from imageProcessing import Frame, align_science_frames, read_frame, source_filter, source_target, _science_stars


@dataclass
class ColorComposite(Frame):
    def display(self):
        return np.where(self.mask[..., None], np.nan, self.data)

    def rgb(self, contrast=False):
        values = np.where(self.mask[..., None], 0., self.data)
        if contrast:
            rgb = make_lupton_rgb(*np.moveaxis(values, -1, 0), minimum=0, stretch=.3, Q=8)
        else:
            # One scale per pixel preserves channel ratios, even in bright stars.
            rgb = np.uint8(255 * values / np.maximum(1., values.max(axis=-1))[..., None])
        rgb[self.mask] = 0
        return rgb

    def png(self, contrast=False, thumbnail=False):
        image = Image.fromarray(np.flipud(self.rgb(contrast)))
        if thumbnail:
            image.thumbnail((480, 320), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (480, 320))
            canvas.paste(image, ((480-image.width)//2, (320-image.height)//2))
            image = canvas
        output = BytesIO()
        image.save(output, format='PNG')
        return output.getvalue()


def background_plane(image, mask):
    """Fit a clipped sky plane to cell medians, excluding bright source cells."""
    h, w = image.shape
    samples = []
    for ys in np.array_split(np.arange(h), 8):
        for xs in np.array_split(np.arange(w), 8):
            block = image[np.ix_(ys, xs)]
            good = ~mask[np.ix_(ys, xs)]
            if good.sum() < max(16, block.size * .6):
                continue
            _, level, _ = sigma_clipped_stats(block[good], sigma=2.5)
            samples.append((1., xs.mean()/w, ys.mean()/h, level))
    if len(samples) < 8:
        _, level, noise = sigma_clipped_stats(image[~mask])
        return np.full_like(image, level), float(noise)
    samples = np.array(samples)
    design, values = samples[:, :3], samples[:, 3]
    good = np.ones(len(values), dtype=bool)
    for _ in range(5):
        fit = np.linalg.lstsq(design[good], values[good], rcond=None)[0]
        residual = values - design @ fit
        scatter = max(1.4826*np.median(np.abs(residual[good]-np.median(residual[good]))), 1e-10)
        new = (residual > -3*scatter) & (residual < 2*scatter)
        if new.sum() < 8 or np.array_equal(new, good):
            break
        good = new
    yy, xx = np.indices(image.shape)
    sky = fit[0] + fit[1]*xx/w + fit[2]*yy/h
    _, _, noise = sigma_clipped_stats((image-sky)[~mask], sigma=3.)
    return sky, float(noise)


def stellar_profiles(planes, mask):
    """Estimate Gaussian widths and aperture fluxes from common isolated stars."""
    reference = Frame(planes[0], fits.Header(), mask, 'reference')
    points = _science_stars(reference, reference, 1)
    rows = []
    radius = min(24, min(mask.shape)//8)
    yy, xx = np.indices((2*radius+1, 2*radius+1))
    rr = np.hypot(yy-radius, xx-radius)
    for y, x in points[:60]:
        if len(rows) >= 24:
            break
        if len(points) > 1 and np.any((np.linalg.norm(points-[y,x], axis=1) > 1) &
                                      (np.linalg.norm(points-[y,x], axis=1) < 2*radius)):
            continue
        y, x = int(round(y)), int(round(x))
        if y < radius or x < radius or y+radius >= mask.shape[0] or x+radius >= mask.shape[1]:
            continue
        section = np.s_[y-radius:y+radius+1, x-radius:x+radius+1]
        if mask[section].any():
            continue
        measurements = []
        for plane in planes:
            patch = plane[section]
            bg = float(np.median(patch[rr > radius*.8]))
            peak = float(patch.max()-bg)
            noise = max(1.4826*np.median(np.abs(patch[rr > radius*.8]-bg)), peak*.002)
            if peak <= 10*noise:
                break
            # Fitting is only for measuring widths; output never synthesizes stars.
            def residual(p):
                amp, cy, cx, sigma, base = p
                model = base+amp*np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*sigma*sigma))
                return ((model-patch)/noise).ravel()[::2]
            fit = least_squares(residual, [peak,radius,radius,3.,bg],
                                bounds=([0,radius-3,radius-3,.6,bg-peak],
                                        [peak*3,radius+3,radius+3,radius/2,bg+peak]),
                                loss='soft_l1', max_nfev=80)
            sigma = fit.x[3]
            if not fit.success or sigma >= radius*.45 or np.median(np.abs(residual(fit.x))) > 5:
                break
            flux = float(np.sum(patch[rr < radius*.75]-bg))
            if flux <= 0:
                break
            measurements.append((sigma,flux))
        if len(measurements) == len(planes):
            rows.append(measurements)
    if len(rows) < 3:
        raise ValueError('Generate needs at least three common, isolated stars to match sharpness and color reliably.')
    measurements = np.array(rows)
    return np.median(measurements[:,:,0],axis=0), measurements[:,:,1]


def match_color_planes(planes, mask):
    widths, fluxes = stellar_profiles(planes, mask)
    target = float(max(widths[:3]))
    matched = []
    largest_kernel = 0.
    for plane, width in zip(planes[:3], widths[:3]):
        kernel = float(np.sqrt(max(0., target*target-width*width)))
        largest_kernel = max(largest_kernel, kernel)
        if kernel > .1:
            weights = gaussian_filter((~mask).astype(float), kernel)
            values = gaussian_filter(np.where(mask,0.,plane), kernel)
            plane = np.divide(values, weights, out=np.zeros_like(values), where=weights > .99)
        matched.append(plane)
    # One multiplicative gain per channel, using integrated stellar flux rather
    # than unrelated image percentiles. The ensemble has a neutral median color.
    gains = np.median(fluxes[:,1,None]/fluxes, axis=0)
    rgb = np.stack(matched, axis=-1)*gains[:3]
    if largest_kernel > .1:
        mask = ~binary_erosion(~mask, iterations=int(np.ceil(3*largest_kernel)), border_value=0)
    return rgb, mask, widths, gains


def generate_color(masters, max_shift, use_luminance=False):
    """Register channels and balance their backgrounds/ranges for visualization.

    Stellar ensemble gains provide visual balance. Catalog zero points are
    not inferred. Luminance must be explicitly identified by the caller.
    """
    channels = {}
    targets = set()
    for source in masters.values():
        image = read_frame(source)
        key = (source_filter(image) or '').casefold()
        targets.add(source_target(image).casefold())
        if key in channels:
            raise ValueError('Choose one master per filter and one target for Generate.')
        channels[key] = image
    if len(targets) != 1:
        raise ValueError('Generate requires masters of one target only.')
    order = ['red', 'green', 'blue'] + (['black'] if use_luminance else [])
    missing = set(order) - channels.keys()
    if missing:
        raise ValueError('Missing science masters: ' + ', '.join(sorted(missing)))
    selected = [channels[key] for key in order]
    for image in selected:
        kind = str(image.header.get('IMAGETYP', '')).upper()
        if ('SCIENCE' not in kind or image.header.get('SCICOR') is not True
                or image.header.get('FLATNORM') is not True):
            raise ValueError(f'{image.name}: Generate requires calibrated science masters, not flats or raw images.')
    for image in selected[1:]:
        for key in ('XPIXSIZE', 'YPIXSIZE', 'FOCALLEN'):
            a, b = selected[0].header.get(key), image.header.get(key)
            if a is not None and b is not None and not np.isclose(float(a), float(b)):
                raise ValueError(f'{image.name}: incompatible pixel scale ({key}).')
    # Registration operates on corrected data and never applies calibration again.
    aligned, offsets = align_science_frames(selected, selected[0], channels, max_shift,
                                            corrected=True, shared_reference=True)
    mask = np.logical_or.reduce([image.mask for image in aligned])
    if np.count_nonzero(~mask) < 16:
        raise ValueError('Not enough common valid pixels across the color channels.')
    planes, noises = [], []
    for image in aligned:
        sky, noise = background_plane(image.data, mask)
        planes.append(np.where(mask, 0., image.data-sky))
        noises.append(noise)
    rgb, mask, widths, gains = match_color_planes(planes, mask)
    rgb = np.maximum(rgb, 0.)
    intensity = rgb.mean(axis=-1)
    if use_luminance:
        luminance = np.maximum(planes[3]*gains[3], 0.)
        noise = max(float(np.sqrt(np.sum((np.array(noises[:3])*gains[:3])**2))/3), 1e-12)
        # Regularize chromaticity in low-S/N pixels instead of dividing by a
        # near-zero RGB intensity and amplifying isolated channel noise.
        strength = intensity**2/(intensity**2+(3*noise)**2)
        chroma = np.divide(rgb, intensity[...,None], out=np.ones_like(rgb), where=intensity[...,None]>0)
        rgb = luminance[...,None]*(1.+strength[...,None]*(chroma-1.))
    scale = float(np.percentile(rgb[~mask].max(axis=-1), 99.9))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('No usable signal in the common image area.')
    rgb /= scale
    rgb[mask] = 0.
    header = aligned[0].header.copy()
    header['IMAGETYP'] = 'COLOR_PREVIEW'
    header['COLORMOD'] = 'LRGB' if use_luminance else 'RGB'
    header['BUNIT'] = '1'
    for key in ('FILTER', 'EXPTIME', 'EXPOSURE'):
        header.remove(key, ignore_missing=True)
    header['HISTORY'] = 'Sky-plane subtraction, matched RGB PSF, ensemble stellar color balance.'
    header['HISTORY'] = 'Display product; no catalog zero-point calibration supplied.'
    for key, width, gain in zip(order, widths, gains):
        header['HISTORY'] = f'{key}: measured Gaussian sigma={width:.4f} px; flux gain={gain:.6g}'
    for key, image, offset in zip(order, selected, offsets):
        header['HISTORY'] = f'{key}: {image.name}; shift={offset}'
    return ColorComposite(rgb, header, mask, f'final_{source_target(selected[0])}_{header["COLORMOD"].lower()}.png')
