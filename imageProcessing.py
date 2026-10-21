"""Operações de tratamento, leitura e exportação de imagens FITS.

As funções trabalham com cópias dos dados e preservam os ficheiros originais.
O contraste pertence à visualização e à exportação PNG, não aos cálculos científicos.
"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import tempfile
import warnings
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy import units as u
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter, rotate as ndi_rotate, shift as ndi_shift
from scipy.optimize import minimize
from scipy.spatial import cKDTree



# Estrutura da imagem, leitura FITS e validação

@dataclass
class Frame:
    data: np.ndarray
    header: fits.Header
    mask: np.ndarray
    name: str = 'image.fits'

    # Apresenta os píxeis válidos e marca os excluídos como NaN.
    def display(self):
        return np.where(self.mask, np.nan, self.data)


# Remove do cabeçalho os valores de escala e limites que ficaram desatualizados.
def clean_header(header):
    header = header.copy()
    for key in ('BSCALE', 'BZERO', 'BLANK', 'DATAMIN', 'DATAMAX', 'CHECKSUM', 'DATASUM'):
        header.remove(key, remove_all=True, ignore_missing=True)
    return header


# Cria uma imagem em ponto flutuante com uma máscara dos píxeis inválidos.
def frame(data, header=None, mask=None, name='image.fits'):
    values = np.array(data, dtype=float, copy=True)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError('Processing requires a non-empty 2D image.')
    bad = ~np.isfinite(values)
    if mask is not None:
        if np.shape(mask) != values.shape:
            raise ValueError('Mask dimensions do not match the image.')
        bad |= np.asarray(mask, dtype=bool)
    return Frame(values, clean_header(header if header is not None else fits.Header()), bad, name)


# Lê uma imagem FITS e a sua máscara sem alterar o ficheiro original.
def read_frame(source):
    if isinstance(source, Frame):
        return frame(source.data, source.header, source.mask, source.name)
    name = Path(source).name if isinstance(source, (str, Path)) else 'carregado.fits'
    with fits.open(BytesIO(source) if isinstance(source, bytes) else source, memmap=False) as hdus:
        candidates = [h for h in hdus if isinstance(h, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                      and h.header.get('NAXIS') == 2 and h.name.upper() != 'MASK']
        if not candidates:
            raise ValueError(f'{name}: does not contain a 2D image.')
        hdu = candidates[0]
        header = hdu.header.copy()
        mask = hdus['MASK'].data if 'MASK' in hdus else None
        return frame(hdu.data, header, mask, name)


# Regista no cabeçalho FITS a operação aplicada à imagem.
def add_history(header, text):
    # FITS headers are ASCII; names can contain Portuguese characters.
    header.add_history(text.encode('ascii', 'replace').decode('ascii'))


# Verifica a compatibilidade das dimensões, aquisição, filtro e unidades de duas imagens.
def compatible(reference, other, check_filter=False, check_unit=True):
    if reference.data.shape != other.data.shape:
        raise ValueError(f'{other.name}: dimensions {other.data.shape} differ from {reference.data.shape}.')
    for key in ('XBINNING', 'YBINNING', 'GAIN', 'INSTRUME'):
        a, b = reference.header.get(key), other.header.get(key)
        if a is not None and b is not None and a != b:
            raise ValueError(f'{other.name}: {key} incompatible ({b} / {a}).')
    if check_filter:
        a, b = reference.header.get('FILTER'), other.header.get('FILTER')
        if a is not None and b is not None and str(a).strip().casefold() != str(b).strip().casefold():
            raise ValueError(f'{other.name}: FILTER {b} differs from {a}.')
    if check_unit:
        # Missing BUNIT is conventional ADU for the local camera files.
        # Interpreta a unidade dos píxeis indicada no cabeçalho FITS.
        def unit(header):
            value = str(header.get('BUNIT', 'adu')).strip()
            return u.Unit('adu' if value.casefold() == 'adu' else value)
        a, b = unit(reference.header), unit(other.header)
        if a != b:
            raise ValueError(f'{other.name}: incompatible units ({b} / {a}).')


# Obtém o tempo de exposição positivo de uma imagem.
def exposure(image):
    value = image.header.get('EXPTIME', image.header.get('EXPOSURE'))
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{image.name}: falta EXPTIME/EXPOSURE em segundos.') from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{image.name}: exposure must be positive.')
    return value


# Obtém a exposição usada para agrupar imagens antes da combinação.
def combination_exposure(header):
    """Exposure in seconds for combination; None means metadata is absent."""
    raw = header.get('EXPTIME', header.get('EXPOSURE'))
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'Invalid exposure value: {raw}.') from None
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'Invalid exposure value: {raw}.')
    return value


# Alinhamento por translação

# Prepara uma cópia da imagem para medir deslocamentos por correlação.
def _registration_array(image):
    """Create a background-suppressed, tapered image for phase correlation."""
    values = np.asarray(image.data, dtype=float)
    valid = ~image.mask & np.isfinite(values)
    if valid.sum() < 16:
        raise ValueError(f'{image.name}: too few valid pixels for alignment.')
    sample = values[valid]
    # Keep sparse star fields useful while limiting isolated detector outliers.
    low, high = np.percentile(sample, [0.01, 99.99])
    filled = np.where(valid, np.clip(values, low, high), np.median(sample))
    sigma = max(2., min(values.shape) / 128.)
    prepared = filled - gaussian_filter(filled, sigma=sigma, mode='nearest')
    prepared[~valid] = 0.
    window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    prepared *= window
    norm = np.linalg.norm(prepared)
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError(f'{image.name}: sem estrutura suficiente para estimar o alinhamento.')
    return prepared / norm


# Mede o deslocamento em Y e X necessário para sobrepor duas imagens.
def estimate_translation(reference, moving, max_shift=100, subpixel=True, phase=True):
    """Estimate the (dy, dx) shift to apply to moving via phase correlation."""
    reference = read_frame(reference)
    moving = read_frame(moving)
    compatible(reference, moving, check_filter=True)
    limit = min(reference.data.shape) // 2 - 2
    if not isinstance(max_shift, (int, np.integer)) or not 1 <= max_shift <= limit:
        raise ValueError(f'Maximum shift must be between 1 and {limit} pixels.')
    a, b = _registration_array(reference), _registration_array(moving)
    product = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    if phase:
        magnitude = np.abs(product)
        product /= np.where(magnitude > np.finfo(float).eps, magnitude, 1.)
    correlation = np.fft.fftshift(np.abs(np.fft.ifft2(product)))
    cy, cx = np.array(correlation.shape) // 2
    area = correlation[cy-max_shift:cy+max_shift+1, cx-max_shift:cx+max_shift+1]
    py, px = np.unravel_index(np.argmax(area), area.shape)
    py, px = py + cy - max_shift, px + cx - max_shift
    dy, dx = float(py - cy), float(px - cx)
    if abs(dy) == max_shift or abs(dx) == max_shift:
        raise ValueError(f'{moving.name}: shift reached the {max_shift}-pixel limit; increase it and retry.')
    if subpixel:
        # Refina a posição do máximo de correlação entre píxeis.
        def offset(before, center, after):
            denominator = before - 2*center + after
            value = 0.5 * (before - after) / denominator if abs(denominator) > 1e-15 else 0.
            return float(np.clip(value, -0.5, 0.5))
        dy += offset(correlation[py-1, px], correlation[py, px], correlation[py+1, px])
        dx += offset(correlation[py, px-1], correlation[py, px], correlation[py, px+1])
        # The parabolic peak gives a good initial value. Refine it by maximizing
        # normalized correlation after bilinear translation.
        margin = min(max_shift + 2, min(a.shape) // 4)
        if min(a.shape) - 2 * margin >= 16:
            reference_crop = a[margin:-margin, margin:-margin]
            reference_power = np.sum(reference_crop * reference_crop)

            # Avalia a correlação entre duas imagens após um deslocamento candidato.
            def objective(candidate):
                shifted = ndi_shift(b, candidate, order=1, mode='constant', cval=0., prefilter=False)
                shifted = shifted[margin:-margin, margin:-margin]
                denominator = np.sqrt(reference_power * np.sum(shifted * shifted))
                return -float(np.sum(reference_crop * shifted) / denominator) if denominator else 0.

            solution = minimize(objective, (dy, dx), method='Powell',
                                bounds=((dy - .75, dy + .75), (dx - .75, dx + .75)),
                                options={'xtol': .01, 'ftol': 1e-7, 'maxiter': 20})
            if solution.success and np.all(np.isfinite(solution.x)):
                dy, dx = (float(value) for value in solution.x)
    else:
        dy, dx = float(round(dy)), float(round(dx))
    return dy, dx


# Desloca a imagem e marca os píxeis sem cobertura.
def translate_frame(image, shift, subpixel=True):
    """Resample an image by (dy, dx), masking uncovered and partial pixels."""
    image = read_frame(image)
    dy, dx = (float(value) for value in shift)
    if not subpixel:
        dy, dx = float(round(dy)), float(round(dx))
    order = 1 if subpixel else 0
    valid = (~image.mask & np.isfinite(image.data)).astype(float)
    numerator = ndi_shift(np.where(valid > 0, image.data, 0.), (dy, dx), order=order,
                          mode='constant', cval=0., prefilter=False)
    weight = ndi_shift(valid, (dy, dx), order=order, mode='constant', cval=0., prefilter=False)
    data = np.divide(numerator, weight, out=np.full_like(numerator, np.nan), where=weight > 1e-12)
    mask = weight < 1 - 1e-7
    result = frame(data, image.header, mask, image.name)
    add_history(result.header, f'Image translated dy={dy:.4f}; dx={dx:.4f}; interpolation order={order}')
    return result


# Calibração: operações individuais sobre a cópia de trabalho

# Subtrai o bias à cópia de trabalho e propaga a máscara do bias.
def _subtract_bias(result, bias):
    if result.header.get('BIASCOR') is True:
        raise ValueError('This image already has BIASCOR; do not subtract bias again.')
    compatible(result, bias)
    result.data -= bias.data
    result.mask |= bias.mask
    result.header['BIASCOR'] = True
    add_history(result.header, f'Bias subtracted: {bias.name}')


# Subtrai o dark à cópia de trabalho, ajustando a exposição quando pedido.
def _subtract_dark(result, dark, scale_dark=False, dark_contains_bias=False):
    if result.header.get('DARKCOR') is True:
        raise ValueError('This image already has DARKCOR; do not subtract dark again.')
    if dark_contains_bias and result.header.get('BIASCOR') is True:
        raise ValueError('Bias has already been subtracted; use a bias-free dark.')
    compatible(result, dark)
    science_time, dark_time = exposure(result), exposure(dark)
    if not scale_dark and not np.isclose(science_time, dark_time, rtol=1e-6, atol=1e-6):
        raise ValueError('Dark and image exposures differ. Use a matching dark or enable exposure scaling for a bias-free dark.')
    factor = science_time / dark_time if scale_dark else 1.0
    result.data -= dark.data * factor
    result.mask |= dark.mask
    result.header['DARKCOR'] = True
    if dark_contains_bias:
        result.header['BIASCOR'] = True
    add_history(result.header, f'Dark subtracted: {dark.name}; factor={factor}; includes bias={dark_contains_bias}')


# Divide a cópia de trabalho pelo flat normalizado e mascara respostas inválidas.
def _divide_by_normalized_flat(result, flat, flat_min=0.0):
    if result.header.get('FLATCOR') is True:
        raise ValueError('This image already has FLATCOR; do not apply another flat.')
    compatible(result, flat, check_filter=True, check_unit=False)
    if not np.isfinite(flat_min) or flat_min < 0:
        raise ValueError('Flat threshold must be finite and nonnegative.')
    valid = ~flat.mask & np.isfinite(flat.data) & (flat.data > 0)
    if not np.any(valid):
        raise ValueError('The flat has no valid positive values.')
    norm = float(np.median(flat.data[valid]))
    response = flat.data / norm
    invalid = ~valid | (response <= flat_min)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result.data = np.divide(result.data, response, out=np.full_like(result.data, np.nan), where=~invalid)
    result.mask |= invalid
    result.header['FLATCOR'] = True
    add_history(result.header, f'Flat divided: {flat.name}; normalization median={norm}; min response={flat_min}')


# Aplica uma máscara externa à cópia de trabalho e regista a operação.
def _apply_bad_pixel_mask(result, mask):
    if mask.shape != result.data.shape:
        raise ValueError('Mask dimensions do not match the image.')
    result.mask |= np.asarray(mask, dtype=bool)
    add_history(result.header, 'External bad-pixel mask applied')


# Subtrai bias e dark e aplica a correção de flat e a máscara quando pedidos.
def calibrate(image, bias=None, dark=None, flat=None, mask=None,
              scale_dark=False, dark_contains_bias=False, flat_min=0.0):
    result = read_frame(image)
    if dark is not None and dark_contains_bias and (bias is not None or scale_dark):
        raise ValueError('A dark containing bias requires equal exposure and cannot be used with separate bias subtraction.')
    if bias is not None:
        _subtract_bias(result, bias)
    if dark is not None:
        _subtract_dark(result, dark, scale_dark, dark_contains_bias)
    if flat is not None:
        _divide_by_normalized_flat(result, flat, flat_min)
    if mask is not None:
        _apply_bad_pixel_mask(result, mask)
    result.mask |= ~np.isfinite(result.data)
    return result


# Combinação de imagens

# Combina imagens por média ou mediana com as opções de alinhamento e rejeição escolhidas.
def combine(sources, method='mean', sigma=5., normalize=False, kind='combined',
            bias=None, dark=None, scale_dark=False, dark_contains_bias=False, progress=None,
            align=False, max_shift=100, subpixel=True, flat_normalize=True, allow_mixed_exposures=False, normalize_exposure=False):
    if len(sources) < 2:
        raise ValueError('Select at least two images.')
    if method not in ('mean', 'median') or (sigma is not None and (not np.isfinite(sigma) or sigma <= 0)):
        raise ValueError('Invalid method or sigma threshold.')
    if kind not in ('combined', 'bias', 'flat'):
        raise ValueError('Invalid combination type.')
    if align and kind != 'combined':
        raise ValueError('Alignment is available only for Combine images.')
    normalize = normalize or (kind == 'flat' and flat_normalize)
    first = read_frame(sources[0])
    first_exposure = combination_exposure(first.header)
    if normalize_exposure:
        first_exposure = exposure(first)
    height, width = first.data.shape
    headers = []
    history = []
    alignment_offsets = []
    # Disk-backed stack avoids keeping every full image in RAM. Sigma clipping
    # operates on bounded row blocks across the exposure axis, never spatially.
    with tempfile.TemporaryDirectory(prefix='fits-combine-') as temp:
        stack = np.memmap(Path(temp) / 'stack.dat', mode='w+', dtype='float64',
                          shape=(len(sources), height, width))
        try:
            for i, source in enumerate(sources):
                image = read_frame(source)
                compatible(first, image, check_filter=kind != 'bias')
                image_exposure = combination_exposure(image.header)
                if image_exposure != first_exposure and not allow_mixed_exposures:
                    raise ValueError(f'{image.name}: exposure {image_exposure} s differs from '
                                     f'{first.name} ({first_exposure} s).')
                image_type = str(image.header.get('IMAGETYP', '')).upper()
                if kind == 'bias' and image_type and 'BIAS' not in image_type and 'ZERO' not in image_type:
                    raise ValueError(f'{image.name}: IMAGETYP does not identify a bias frame.')
                if kind == 'flat' and image_type and 'FLAT' not in image_type:
                    raise ValueError(f'{image.name}: IMAGETYP does not identify a flat frame.')
                headers.append(image.header.copy())
                if kind == 'flat':
                    image = calibrate(image, bias=bias, dark=dark, scale_dark=scale_dark,
                                      dark_contains_bias=dark_contains_bias)
                dy, dx = 0., 0.
                if align and i:
                    dy, dx = estimate_translation(first, image, max_shift=max_shift, subpixel=subpixel)
                    image = translate_frame(image, (dy, dx), subpixel=subpixel)
                if kind == 'flat' and normalize:
                    image.mask |= image.data <= 0
                values = image.display()
                if normalize_exposure:
                    values *= first_exposure / exposure(image)
                if normalize:
                    valid = values[np.isfinite(values)]
                    median = float(np.median(valid)) if valid.size else np.nan
                    if not np.isfinite(median) or median <= 0:
                        raise ValueError(f'{image.name}: median is not positive; cannot normalize.')
                    values /= median
                stack[i] = values
                history.append(f'Input {i + 1}: {image.name}; alignment dy={dy:.4f} dx={dx:.4f}')
                alignment_offsets.append((dy, dx))
                if progress:
                    progress((i + 1) / len(sources) * 0.5)
            output = np.empty((height, width), dtype=float)
            rows = max(1, min(height, int(64e6 / (len(sources) * width * 8 * 5))))
            for start in range(0, height, rows):
                block = np.ma.masked_invalid(np.array(stack[:, start:start + rows]))
                if sigma is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        block = sigma_clip(block, sigma=sigma, axis=0, cenfunc='median',
                                           stdfunc='mad_std', maxiters=5, masked=True)
                reduced = np.ma.mean(block, axis=0) if method == 'mean' else np.ma.median(block, axis=0)
                output[start:start + rows] = reduced.filled(np.nan)
                if progress:
                    progress(0.5 + min(start + rows, height) / height * 0.5)
        finally:
            del stack
    header = clean_header(first.header)
    if normalize_exposure:
        header['EXPTIME'] = first_exposure
        header['EXPOSURE'] = first_exposure
        header['EXPNORM'] = True
        add_history(header, f'Inputs scaled to {first_exposure:g} s before rejection and combination.')
    elif allow_mixed_exposures and len({combination_exposure(h) for h in headers}) > 1:
        header.remove('EXPTIME', ignore_missing=True)
        header.remove('EXPOSURE', ignore_missing=True)
        add_history(header, 'Combined counts from differing exposures without exposure scaling.')
    header['COMBINED'] = True
    header['NCOMBINE'] = len(sources)
    header['COMBMETH'] = method
    header['CLIPSIG'] = float(sigma) if sigma is not None else 0.
    header['CLIPCEN'] = 'median'
    header['CLIPDEV'] = 'mad_std'
    if align:
        header['ALIGN'] = ('TRANSLATION', 'Registration before combination')
        header['ALNSUBPX'] = (bool(subpixel), 'Subpixel interpolation used')
        header['ALNMAX'] = (int(max_shift), 'Maximum allowed shift in pixels')
        # Copy shift diagnostics collected on the temporary frame headers.
        for i, (dy, dx) in enumerate(alignment_offsets[:1000]):
            header[f'ALN{i:03d}Y'] = (dy, 'Applied row shift')
            header[f'ALN{i:03d}X'] = (dx, 'Applied column shift')
    if normalize:
        header['BUNIT'] = '1'
        header['NORMALIZ'] = True
    if kind == 'bias':
        header['IMAGETYP'] = 'MASTER_BIAS'
        header['EXPTIME'] = 0.
        header['EXPOSURE'] = 0.
        header.remove('FILTER', ignore_missing=True)
    if kind == 'flat':
        if normalize:
            valid = np.isfinite(output) & (output > 0)
            if not valid.any():
                raise ValueError('Master flat has no positive response.')
            output /= np.median(output[valid])
            output[~valid] = np.nan
        header['NORMALIZ'] = bool(normalize)
        header['IMAGETYP'] = 'MASTER_FLAT'
        header['BIASCOR'] = bias is not None or (dark is not None and dark_contains_bias) or all(h.get('BIASCOR') is True for h in headers)
        header['DARKCOR'] = dark is not None or all(h.get('DARKCOR') is True for h in headers)
    add_history(header, f'Combine method={method}; sigma={sigma}; iterations=5; median normalization={normalize}; align={align}')
    for line in history:
        add_history(header, line)
    if bias is not None:
        add_history(header, f'Flat calibration bias: {bias.name}')
    if dark is not None:
        add_history(header, f'Flat calibration dark: {dark.name}; scale={scale_dark}; includes bias={dark_contains_bias}')
    return frame(output, header,
                 name=f'{"master_" if kind != "combined" else ""}{kind}_x{len(sources)}.fits')


# Máscaras e escrita FITS

# Marca os píxeis a excluir através de limites, região ou máscara externa.
def mask_image(image, lower=None, upper=None, rectangle=None, external=None):
    result = read_frame(image)
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError('Lower limit must be less than upper limit.')
    if lower is not None:
        result.mask |= result.data < lower
    if upper is not None:
        result.mask |= result.data > upper
    if rectangle is not None:
        x0, x1, y0, y1 = rectangle
        if not (0 <= x0 < x1 <= result.data.shape[1] and 0 <= y0 < y1 <= result.data.shape[0]):
            raise ValueError('Invalid mask region.')
        result.mask[y0:y1, x0:x1] = True
    if external is not None:
        if external.shape != result.data.shape:
            raise ValueError('External mask dimensions differ.')
        result.mask |= np.asarray(external, dtype=bool)
    result.header['MASKED'] = True
    add_history(result.header, f'Mask: lower={lower}; upper={upper}; rectangle={rectangle}; external={external is not None}')
    return result


# Lê de um FITS a máscara dos píxeis excluídos.
def read_mask(source):
    if isinstance(source, Frame):
        return source.mask.copy()
    with fits.open(BytesIO(source) if isinstance(source, bytes) else source, memmap=False) as hdus:
        if 'MASK' in hdus:
            data = hdus['MASK'].data
        else:
            images = [h.data for h in hdus if h.header.get('NAXIS') == 2 and h.data is not None]
            if not images:
                raise ValueError('File does not contain a 2D mask.')
            data = images[0]
        if np.ndim(data) != 2:
            raise ValueError('Mask must be 2D.')
        return (np.asarray(data) != 0) | ~np.isfinite(data)


# Converte os dados científicos e a máscara da imagem em bytes FITS.
def to_fits(image):
    header = clean_header(image.header)
    header['NMASKED'] = int(image.mask.sum())
    hdus = fits.HDUList([fits.PrimaryHDU(image.data, header=header),
                         fits.ImageHDU(image.mask.astype('uint8'), name='MASK')])
    hdus[1].header['MASKTRUE'] = '1 = excluded pixel'
    buffer = BytesIO()
    hdus.writeto(buffer, checksum=True)
    return buffer.getvalue()



# Estatísticas, escalas e recortes

# Calcula as estatísticas dos píxeis finitos de uma imagem.
def statistics(data):
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {}
    return dict(pixeis=int(values.size), minimo=float(values.min()),
                maximo=float(values.max()), media=float(values.mean()),
                mediana=float(np.median(values)), desvio=float(values.std()),
                soma=float(values.sum()))


# Converte os valores em níveis de visualização segundo a escala escolhida.
def display_values(data, low, high, scale):
    normalized = np.clip((np.asarray(data, dtype=float) - low) / max(high-low, 1e-30), 0, 1)
    if scale == 'Logarithmic':
        return np.log1p(1000 * normalized) / np.log1p(1000)
    if scale == 'Asinh':
        return np.arcsinh(10 * normalized) / np.arcsinh(10)
    if scale == 'Square root':
        return np.sqrt(normalized)
    return normalized


# Recorta a imagem, aplica a subtração e o ganho pedidos e preserva a máscara e o WCS.
def export_crop(data, header, x0, x1, y0, y1, offset=0., gain=1., mask=None):
    """Recorta os dois eixos espaciais, preservando os restantes eixos e WCS."""
    if not (0 <= x0 < x1 <= data.shape[-1] and 0 <= y0 < y1 <= data.shape[-2]):
        raise ValueError('Invalid crop bounds.')
    result = (np.array(data[..., y0:y1, x0:x1], dtype=float) - offset) * gain
    hdr = header.copy()
    for key in list(hdr):
        if key in ('BSCALE', 'BZERO', 'BLANK', 'DATAMIN', 'DATAMAX', 'CHECKSUM', 'DATASUM'):
            del hdr[key]
    # Include alternate WCS versions (CRPIX1A, CRPIX2A, ...).
    for key in list(hdr):
        if key.startswith('CRPIX1'):
            hdr[key] -= x0
        elif key.startswith('CRPIX2'):
            hdr[key] -= y0
    hdr.add_history(f'Crop x=[{x0},{x1}), y=[{y0},{y1}); (data - {offset}) * {gain}')
    buffer = BytesIO()
    output = [fits.PrimaryHDU(result, header=hdr)]
    if mask is not None:
        if np.shape(mask) != np.shape(data):
            raise ValueError('Mask dimensions do not match the image.')
        cropped_mask = np.asarray(mask, dtype=bool)[..., y0:y1, x0:x1] | ~np.isfinite(result)
        output[0].header['NMASKED'] = int(cropped_mask.sum())
        output.append(fits.ImageHDU(cropped_mask.astype('uint8'), name='MASK'))
    fits.HDUList(output).writeto(buffer, checksum=True)
    return buffer.getvalue()



# Subtração de fundo e alinhamento com rotação

# Subtrai à imagem a mediana do fundo estimada com rejeição sigma.
def subtract_background(image, sigma=3.0):
    result = read_frame(image)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Background sigma must be positive.')
    valid = result.data[~result.mask & np.isfinite(result.data)]
    if not valid.size:
        raise ValueError(f'{result.name}: no valid pixels for background estimation.')
    clipped = sigma_clip(valid, sigma=sigma, maxiters=5)
    level = float(np.ma.median(clipped))
    if not np.isfinite(level):
        raise ValueError(f'{result.name}: background estimation failed.')
    result.data -= level
    result.header['BKGDSUB'] = (True, 'Global sky background subtracted')
    result.header['BKGDLEV'] = (level, 'Subtracted sky level')
    add_history(result.header, f'Global sigma-clipped sky median subtracted: {level:.8g}')
    return result


# Roda a imagem em torno do centro e mascara os píxeis sem cobertura.
def rotate_frame(image, angle):
    """Rotate around the image center, masking uncovered or partly valid pixels."""
    image = read_frame(image)
    valid = (~image.mask & np.isfinite(image.data)).astype(float)
    numerator = ndi_rotate(np.where(valid > 0, image.data, 0.), angle, reshape=False,
                           order=1, mode='constant', cval=0., prefilter=False)
    weight = ndi_rotate(valid, angle, reshape=False, order=1, mode='constant',
                        cval=0., prefilter=False)
    values = np.divide(numerator, weight, out=np.full_like(numerator, np.nan), where=weight > 1e-12)
    result = frame(values, image.header, weight < 1 - 1e-7, image.name)
    add_history(result.header, f'Image rotated about center by {angle:.4f} degrees')
    return result


# Alinha uma imagem com a referência por translação e, quando pedida, rotação.
def align_frame(reference, moving, max_shift=100, max_angle=0.0):
    """Find a small rotation, then the X/Y translation to the reference."""
    reference, moving = read_frame(reference), read_frame(moving)
    if not np.isfinite(max_angle) or max_angle < 0 or max_angle > 15:
        raise ValueError('Maximum rotation must be between 0 and 15 degrees.')
    angle = 0.0
    if max_angle:
        step = max(1, int(np.ceil(max(reference.data.shape) / 384)))
        ref_small = frame(reference.data[::step, ::step], reference.header,
                          reference.mask[::step, ::step], reference.name)
        mov_small = frame(moving.data[::step, ::step], moving.header,
                          moving.mask[::step, ::step], moving.name)
        limit = min(ref_small.data.shape) // 2 - 2
        small_shift = min(limit, max(1, int(np.ceil(max_shift / step))))
        if limit < 1:
            raise ValueError('Image is too small for rotational alignment.')
        prepared_reference = _registration_array(ref_small)

        # Mede a correlação das imagens para avaliar uma rotação candidata.
        def score(candidate):
            try:
                rotated = rotate_frame(mov_small, candidate)
                dy, dx = estimate_translation(ref_small, rotated, small_shift, False)
                prepared = _registration_array(rotated)
                shifted = ndi_shift(prepared, (dy, dx), order=1, mode='constant', cval=0.)
                return float(np.sum(prepared_reference * shifted))
            except ValueError:
                return -np.inf

        coarse = np.linspace(-max_angle, max_angle, 9)
        angle = float(max(coarse, key=score))
        spacing = coarse[1] - coarse[0]
        fine = np.linspace(max(-max_angle, angle-spacing),
                           min(max_angle, angle+spacing), 7)
        angle = float(max(fine, key=score))
        if not np.isfinite(score(angle)):
            raise ValueError(f'{moving.name}: cannot determine rotational alignment.')
    rotated = rotate_frame(moving, angle) if angle else moving
    dy, dx = estimate_translation(reference, rotated, max_shift, True)
    aligned = translate_frame(rotated, (dy, dx))
    return aligned, (angle, dy, dx)


# Masters e alinhamento de imagens de calibração

# Produz o master bias combinando os frames de bias por mediana ou média.
def combine_bias_frames(sources, method='mean'):
    """Create the first workflow product from the selected bias exposures."""
    return combine(sources, kind='bias', method=method, sigma=5.)


# Verifica se os píxeis contêm estrutura suficiente para medir um alinhamento.
def _has_registration_structure(image):
    values = image.data[~image.mask & np.isfinite(image.data)]
    if values.size < 16:
        return False
    scale = max(1., abs(float(np.median(values))))
    return bool(np.ptp(values) > 1e-10 * scale)


# Alinha os frames de bias ou os frames de flat dentro de cada grupo.
def align_calibration_frames(sources, kind, max_shift=100, output_folder=None):
    """Register calibration frames, leaving unmeasurable shifts unchanged."""
    if kind not in ('bias', 'flat'):
        raise ValueError('Alignment is available for bias and flat frames.')
    sources = list(sources)
    if len(sources) < 2:
        raise ValueError('Select at least two frames to align.')
    groups = ({'bias': sources} if kind == 'bias' else {
        (filter_name, exposure): selected
        for filter_name, exposures in flat_exposure_groups(sources).items()
        for exposure, selected in exposures.items()})
    # Guarda uma imagem de calibração com o deslocamento medido e o estado do alinhamento.
    def persist(image, status, offset, number):
        if output_folder is None:
            return image
        image.header['ALNSTAT'] = status
        image.header['ALNMAX'] = int(max_shift)
        image.header['ALIGN'] = ('TRANSLATION' if status == 'SHIFTED' else
                                 image.header.get('ALIGN', 'NO SHIFT') if status == 'REFERENCE' else
                                 'NO SHIFT')
        add_history(image.header, f'Calibration alignment dy={offset[0]:.4f}; dx={offset[1]:.4f}')
        destination = Path(output_folder) / str(number) / Path(image.name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(to_fits(image))
        return destination

    aligned_by_id = {}
    offsets_by_id = {}
    for selected in groups.values():
        if len(selected) < 2:
            for source in selected:
                aligned_by_id[id(source)] = source
                offsets_by_id[id(source)] = (0., 0.)
            continue
        reference = read_frame(selected[0])
        limit = min(reference.data.shape) // 2 - 2
        if not isinstance(max_shift, (int, np.integer)) or not 1 <= max_shift <= limit:
            raise ValueError(f'Maximum shift must be between 1 and {limit} pixels.')
        aligned = [reference]
        offsets = [(0., 0.)]
        skipped_indices = set()
        # Estimate on a bounded image; resample the full frame just once.
        # Powell refinement on every full-resolution flat was taking minutes.
        step = max(1, int(np.ceil(max(reference.data.shape) / 384)))
        small_reference = frame(reference.data[::step, ::step], reference.header,
                                reference.mask[::step, ::step], reference.name)
        small_limit = min(small_reference.data.shape) // 2 - 2
        small_shift = min(small_limit, max(1, int(np.ceil(max_shift / step))))
        try:
            if not _has_registration_structure(small_reference) or small_limit < 1:
                raise ValueError('No measurable structure in reference frame.')
            prepared_reference = _registration_array(small_reference)
        except ValueError:
            prepared_reference = None
        smooth_reference = (gaussian_filter(prepared_reference, sigma=3)
                            if prepared_reference is not None else None)
        smooth_reference_norm = (np.linalg.norm(smooth_reference)
                                 if smooth_reference is not None else 0.)
        for source in selected[1:]:
            moving = read_frame(source)
            compatible(reference, moving, check_filter=True)
            displacement = None
            if prepared_reference is not None and (kind == 'flat' or smooth_reference_norm >= 0.15):
                try:
                    small_moving = frame(moving.data[::step, ::step], moving.header,
                                         moving.mask[::step, ::step], moving.name)
                    if not _has_registration_structure(small_moving):
                        raise ValueError('No measurable structure in moving frame.')
                    prepared_moving = _registration_array(small_moving)
                    if kind == 'flat' or np.linalg.norm(gaussian_filter(prepared_moving, 3)) >= 0.15:
                        dy, dx = estimate_translation(small_reference, small_moving, small_shift, True)
                        shifted = ndi_shift(prepared_moving, (dy, dx), order=1,
                                            mode='constant', cval=0., prefilter=False)
                        score = float(np.sum(prepared_reference * shifted))
                        smooth_shifted = gaussian_filter(shifted, sigma=3)
                        smooth_norm = smooth_reference_norm * np.linalg.norm(smooth_shifted)
                        smooth_score = (float(np.sum(smooth_reference * smooth_shifted) / smooth_norm)
                                        if smooth_norm else 0.)
                        if (score >= 0.05 and smooth_score >= 0.3 and
                                abs(dy * step) < max_shift and abs(dx * step) < max_shift):
                            displacement = (dy * step, dx * step)
                except ValueError:
                    pass
            if displacement is None:
                aligned.append(source if isinstance(source, (str, Path, bytes)) else
                               persist(moving, 'UNCHANGED', (0., 0.), len(aligned_by_id) + len(aligned)))
                offsets.append((0., 0.))
                skipped_indices.add(len(aligned) - 1)
            else:
                shifted_frame = translate_frame(moving, displacement)
                aligned.append(persist(shifted_frame, 'SHIFTED', displacement,
                                       len(aligned_by_id) + len(aligned)))
                offsets.append(displacement)
        reference.header['ALNSKIP'] = (len(skipped_indices), 'Frames kept unshifted without registration')
        reference.header['ALNCOUNT'] = (len(selected) - 1 - len(skipped_indices), 'Frames aligned to reference')
        for index, (dy, dx) in enumerate(offsets[:100]):
            reference.header[f'ALN{index:03d}Y'] = (dy, 'Applied row shift')
            reference.header[f'ALN{index:03d}X'] = (dx, 'Applied column shift')
        for index, (item, offset) in enumerate(zip(aligned, offsets)):
            if isinstance(item, Frame):
                item.header['ALIGN'] = ('TRANSLATION' if len(skipped_indices) < len(selected) - 1
                                        else 'NO SHIFT', 'Calibration registration outcome')
                item.header['ALNMAX'] = (int(max_shift), 'Maximum allowed shift in pixels')
                if index in skipped_indices:
                    item.header['ALNSTAT'] = ('UNCHANGED', 'No reliable shift could be measured')
                    add_history(item.header, 'Alignment not measurable; frame kept at original pixels')
                else:
                    item.header['ALNSTAT'] = ('REFERENCE' if index == 0 else 'SHIFTED',
                                              'Calibration registration status')
                    add_history(item.header, f'Calibration alignment dy={offset[0]:.4f}; dx={offset[1]:.4f}')
            if isinstance(item, Frame) and output_folder is not None:
                item = persist(item, item.header['ALNSTAT'], offset, len(aligned_by_id))
            aligned_by_id[id(selected[index])] = item
            offsets_by_id[id(selected[index])] = offset
    return ([aligned_by_id[id(source)] for source in sources],
            [offsets_by_id[id(source)] for source in sources])


# Seleção e agrupamento de imagens por metadados

# Lê apenas o cabeçalho da imagem para identificar os dados de aquisição.
def _source_header(source):
    """Read image metadata without loading its pixel array."""
    if isinstance(source, Frame):
        return source.header
    with fits.open(BytesIO(source) if isinstance(source, bytes) else source,
                   memmap=False) as hdus:
        image = next((hdu for hdu in hdus
                      if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU,
                                          fits.CompImageHDU))
                      and hdu.header.get('NAXIS') == 2
                      and hdu.name.upper() != 'MASK'), None)
        if image is None:
            raise ValueError(f'{source}: does not contain a 2D image.')
        return image.header.copy()


# Obtém o nome do filtro indicado no cabeçalho da imagem.
def _filter_name(source):
    value = _source_header(source).get('FILTER')
    return str(value).strip().casefold() if value is not None and str(value).strip() else None


# Obtém o tempo de exposição de um frame ou ficheiro FITS.
def source_exposure(source):
    return combination_exposure(_source_header(source))


# Seleciona os frames de flat que correspondem ao filtro da imagem de ciência.
def matching_flat_sources(sources, reference):
    """Use only flats for the science filter when a mixed batch is selected."""
    sources = list(sources)
    science_filter = _filter_name(reference)
    filters = [_filter_name(source) for source in sources]
    named = set(filters) - {None}
    if science_filter and named:
        matched = [source for source, value in zip(sources, filters)
                   if value == science_filter]
        if len(matched) < 2:
            raise ValueError(f'Only {len(matched)} flat frame(s) match science FILTER '
                             f'{science_filter}; select at least two.')
        return matched
    if not science_filter and len(named) > 1:
        raise ValueError('The science image has no FILTER; select flats from one filter '
                         'or use a science image with FILTER metadata.')
    return sources


# Combina os frames de flat e aplica o bias e a normalização apenas quando pedidos.
def combine_flat_frames(sources, bias=None, method='mean', reference=None, normalize=True):
    """Create a normalized master flat for the science filter after bias correction."""
    sources = list(sources)
    selected = matching_flat_sources(sources, reference) if reference is not None else sources
    result = combine(selected, kind='flat', bias=bias, method=method, sigma=5.,
                     flat_normalize=normalize, allow_mixed_exposures=True,
                     normalize_exposure=len({source_exposure(item) for item in selected}) > 1)
    if reference is not None and len(selected) != len(sources):
        add_history(result.header, f'Flat FILTER selection: {len(selected)} of {len(sources)} inputs used')
    return result


FILTER_NAMES = {'red': 'Red', 'r': 'Red', 'green': 'Green', 'g': 'Green',
                'blue': 'Blue', 'b': 'Blue', 'black': 'black'}


FILTER_ORDER = ('Red', 'Green', 'Blue', 'black')


# Identifica o filtro pelo nome do ficheiro e verifica a correspondência com o cabeçalho.
def source_filter(source):
    """Identify a filter from a filename token, checking the FITS header when present."""
    name = source.name if isinstance(source, Frame) else (
        Path(source).name if isinstance(source, (str, Path)) else '')
    tokens = re.split(r'[^a-z0-9]+', name.casefold())
    from_name = next((FILTER_NAMES[token] for token in reversed(tokens)
                      if token in FILTER_NAMES), None)
    header_name = _filter_name(source)
    from_header = FILTER_NAMES.get(header_name, header_name) if header_name else None
    if from_name and from_header and from_name.casefold() != from_header.casefold():
        raise ValueError(f'{name}: filename filter {from_name} differs from FITS FILTER {from_header}.')
    return from_name or from_header


# Agrupa os frames de flat pelo respetivo filtro.
def group_flat_sources(sources):
    """Group flat inputs by the filter token in their name or FITS header."""
    groups = {}
    for source in sources:
        filter_name = source_filter(source)
        if not filter_name:
            name = source.name if isinstance(source, Frame) else str(source)
            raise ValueError(f'{name}: filter is missing from the filename and FITS header.')
        groups.setdefault(filter_name, []).append(source)
    if not groups:
        raise ValueError('Select at least two flat images.')
    ordered = sorted(groups, key=lambda name: (FILTER_ORDER.index(name)
                     if name in FILTER_ORDER else len(FILTER_ORDER), name))
    return {filter_name: groups[filter_name] for filter_name in ordered}


# Agrupa os frames de flat por filtro e tempo de exposição.
def flat_exposure_groups(sources):
    """Preserve selected flats, separated by filter and exposure."""
    grouped = {}
    for filter_name, selected in group_flat_sources(sources).items():
        exposures = {}
        for source in selected:
            exposures.setdefault(source_exposure(source), []).append(source)
        grouped[filter_name] = dict(sorted(exposures.items(),
                                           key=lambda item: (item[0] is None,
                                                             item[0] if item[0] is not None else 0)))
    return grouped


# Produz um master flat por filtro, escalando exposições diferentes antes de combinar.
def combine_flat_groups(sources, bias=None, method='mean', return_sources=False):
    """Combine all exposures per filter; subtract bias before scaling each input."""
    masters = {}
    names = {}
    for filter_name, selected in group_flat_sources(sources).items():
        if len(selected) < 2:
            continue
        exposures = {source_exposure(source) for source in selected}
        master = combine_flat_frames(selected, bias=bias, method=method, normalize=bias is not None)
        master.header['FILTER'] = (filter_name, 'Filter of the combined flat frames')
        single_exposure = next(iter(exposures)) if len(exposures) == 1 else None
        suffix = f'_{single_exposure:g}s' if single_exposure is not None else ''
        master.name = f'master_flat_{filter_name.casefold()}{suffix}_x{len(selected)}.fits'
        if bias is not None:
            master.name = master.name.replace('master_flat_', 'corrected_flat_', 1)
            master.header['IMAGETYP'] = 'CORRECTED_FLAT'
            master.header['BIASFILE'] = bias.name
        add_history(master.header, f'Flat group FILTER={filter_name}; {len(selected)} inputs')
        masters[filter_name] = master
        names[filter_name] = [source.name if isinstance(source, Frame) else (
                              Path(source).name if isinstance(source, (str, Path)) else 'carregado.fits')
                              for source in selected]
    if not masters:
        raise ValueError('No flat filter group contains at least two images.')
    if return_sources:
        return masters, names
    return masters


# Subtrai o master bias a cada master flat sem normalizar os seus valores.
def correct_flat_groups(masters, bias):
    """Diagnostic subtraction only; science uses individually normalized flat inputs."""
    bias = read_frame(bias)
    corrected = {}
    for key, master in masters.items():
        result = calibrate(master, bias=bias)
        result.header['IMAGETYP'] = 'CORRECTED_FLAT'
        result.name = master.name.replace('master_flat_', 'corrected_flat_', 1)
        corrected[key] = result
    return corrected


# Escolhe o master flat corrigido que corresponde ao filtro e exposição da imagem.
def matching_master_flat(masters, reference):
    """Select a matching filter, preferring exposure, then the largest input count."""
    filter_name = source_filter(reference)
    candidates = [master for master in masters.values()
                  if source_filter(master) == filter_name]
    if not candidates:
        raise ValueError(f'{reference.name}: no master flat for FILTER {filter_name or "unknown"}.')
    matching = [master for master in candidates
                if source_exposure(master) == source_exposure(reference)]
    candidates = matching or candidates
    best_count = max(master.header.get('NCOMBINE', 1) for master in candidates)
    best = [master for master in candidates if master.header.get('NCOMBINE', 1) == best_count]
    if len(best) != 1:
        raise ValueError(f'{reference.name}: several master flats match FILTER {filter_name}; '
                         'choose one flat exposure group.')
    return best[0]


# Identifica o objeto observado pelo nome do ficheiro ou cabeçalho da imagem.
def source_target(source, header=None):
    name = source.name if isinstance(source, Frame) else Path(source).name
    if name.startswith('corrected_science_'):
        name = name[len('corrected_science_'):]
    match = re.match(r'(.+?)_\d{4}-\d{2}-\d{2}_', name)
    if match:
        return match.group(1)
    header = _source_header(source) if header is None else header
    return str(header.get('OBJECT', 'science')).strip() or 'science'


# Correção das imagens de ciência

# Subtrai o master bias e divide pela resposta de flat normalizada, preservando contagens.
def correct_science_frame(source, bias, flats):
    """Subtract bias and divide by a unit-normalized flat response, retaining counts."""
    image = read_frame(source)
    kind = str(image.header.get('IMAGETYP', '')).upper()
    if (any(token in kind for token in ('BIAS', 'ZERO', 'FLAT', 'DARK')) or
            image.name.casefold().startswith(('bias', 'flat', 'dark'))):
        raise ValueError(f'{image.name}: select science images, not calibration frames.')
    target = source_target(image)
    flat = matching_master_flat(flats, image)
    if flat.header.get('BIASCOR') is not True:
        raise ValueError(f'{image.name}: use a bias-corrected master flat.')
    if image.header.get('FLATCOR') is True:
        raise ValueError(f'{image.name}: science corrections have already been applied; select original frames.')
    result = calibrate(image, bias=bias, flat=flat)
    result.header['BIASFILE'] = bias.name
    result.header['FLATFILE'] = flat.name
    result.header['FLATCOR'] = True
    result.header['SCICOR'] = True
    result.header['FLATNORM'] = True
    result.header['BUNIT'] = image.header.get('BUNIT', 'adu')
    result.header['DARKCOR'] = False
    add_history(result.header, 'Dark correction not applied: no dark supplied by this workflow.')
    result.header['OBJECT'] = target
    result.header['FILTER'] = source_filter(image)
    result.header['IMAGETYP'] = 'CORRECTED_SCIENCE'
    result.header['ORIGFILE'] = image.name
    stem = re.sub(r'\.(fits|fit|fts)(\.gz|\.fz)?$|\.fz$', '', image.name, flags=re.IGNORECASE)
    result.name = f'corrected_science_{stem}.fits'
    add_history(result.header, f'Science divided by normalized corrected flat {flat.name}')
    return result


# Corrige individualmente todas as imagens de ciência selecionadas e guarda cada resultado.
def correct_science_frames(sources, bias, flats, output_folder=None):
    """Correct every selected frame, retaining one output per original image."""
    sources = list(sources)
    if not sources:
        raise ValueError('Select science images to correct.')
    if bias is None:
        raise ValueError('Create a master bias before correcting science images.')
    bias = read_frame(bias)
    flats = {key: read_frame(value) for key, value in flats.items()}
    if not flats:
        raise ValueError('Create corrected master flats before correcting science images.')
    corrected = []
    for index, source in enumerate(sources):
        image = correct_science_frame(source, bias, flats)
        if output_folder is None:
            corrected.append(image)
        else:
            path = Path(output_folder) / f'{index:05d}' / image.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(to_fits(image))
            corrected.append(path)
    return corrected


# Alinhamento das imagens de ciência pelas estrelas

# Mede os centros das estrelas numa cópia usada apenas para determinar o alinhamento.
def _science_stars(image, small, step):
    """Measure stellar centroids; filtering is for detection only."""
    valid = ~small.mask & np.isfinite(small.data)
    sample = small.data[valid]
    if sample.size < 16:
        return np.empty((0, 2))
    filled = np.where(valid, small.data, np.median(sample))
    signal = gaussian_filter(filled, 1.) - gaussian_filter(filled, 10.)
    noise = 1.4826 * np.median(np.abs(signal[valid] - np.median(signal[valid])))
    threshold = max(6. * noise, np.max(signal) * .002)
    peaks = valid & (signal > threshold) & (signal == maximum_filter(signal, size=9))
    peaks[:8] = peaks[-8:] = False
    peaks[:, :8] = peaks[:, -8:] = False
    points = np.argwhere(peaks)
    points = sorted(points, key=lambda p: signal[tuple(p)], reverse=True)[:100]
    centers = []
    for sy, sx in points:
        if (signal[sy-2:sy+3, sx-2:sx+3] > threshold).sum() < 2:
            continue
        y, x = int(sy * step + (step - 1) / 2.), int(sx * step + (step - 1) / 2.)
        radius = max(5, 3 * step)
        values = image.data[y-radius:y+radius+1, x-radius:x+radius+1]
        good = ~image.mask[y-radius:y+radius+1, x-radius:x+radius+1]
        if not good.any():
            continue
        patch = np.where(good, np.maximum(values - np.median(values[good]), 0.), 0.)
        yy, xx = np.indices(patch.shape)
        power = patch.sum()
        # A hot pixel can survive bias subtraction. Unlike a stellar PSF,
        # most of its local flux sits in a single pixel; do not register it.
        if power > 0 and patch.max() / power < .4:
            centers.append((y - radius + np.sum(yy * patch) / power,
                            x - radius + np.sum(xx * patch) / power))
    return np.asarray(centers).reshape(-1, 2)


# Mede a translação através de correspondências entre as estrelas de duas imagens.
def _science_star_shift(reference, moving, max_shift, shape=None):
    if min(len(reference), len(moving)) < 3:
        return None
    differences = (reference[:, None, :] - moving[None, :, :]).reshape(-1, 2)
    differences = differences[np.max(np.abs(differences), axis=1) <= max_shift]
    if not len(differences):
        return None
    bins, counts = np.unique(np.round(differences / 2.).astype(int), axis=0, return_counts=True)
    tree = cKDTree(reference)
    best = None
    for bin_index in np.argsort(counts)[-12:]:
        offset = bins[bin_index] * 2.
        for _ in range(3):
            distance, indices = tree.query(moving + offset)
            good = distance < 2.
            if good.sum() < 3 or len(np.unique(indices[good])) != good.sum():
                break
            residuals = reference[indices[good]] - moving[good]
            offset = np.median(residuals, axis=0)
        else:
            residual = np.linalg.norm(residuals - offset, axis=1)
            count = int((residual < .8).sum())
            overlap = (np.prod(np.maximum(np.asarray(shape) - np.abs(offset), 0.) / shape)
                       if shape is not None else 1.)
            minimum = 4 if min(len(reference), len(moving)) >= 8 else 3
            required = max(minimum, min(8, int(np.ceil(min(len(reference), len(moving)) * .2 * overlap))))
            if count >= required and np.max(np.abs(offset)) <= max_shift:
                if best is None or count > best[0]:
                    best = (count, tuple(float(value) for value in offset))
    return best[1] if best else None


# Alinha as imagens de ciência dentro de cada grupo de objeto, filtro e exposição.
def align_science_frames(sources, bias, flats, max_shift=100, output_folder=None, corrected=False, group_by_filter=False, shared_reference=False):
    """Register each target/filter/exposure group to its own stellar reference.

    Block averages are used only to measure shifts. Resampling is applied once
    to the full-resolution scientific data; display contrast is never involved.
    Unreliable registration fails the job rather than silently stacking it.
    """
    sources = list(sources)
    if len(sources) < 2:
        raise ValueError('Select at least two science frames to align.')
    bias = read_frame(bias)
    flats = {key: read_frame(value) for key, value in flats.items()}
    if not flats:
        raise ValueError('Create master flats before aligning science images.')
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    references, full_stars, aligned, offsets = {}, {}, [], []

    # Calcula médias em blocos apenas para medir o alinhamento sem reduzir a imagem científica.
    def registration_image(image, step):
        height, width = image.data.shape
        height, width = height // step * step, width // step * step
        valid = (~image.mask[:height, :width]).astype(float)
        shape = (height // step, step, width // step, step)
        weights = valid.reshape(shape).sum(axis=(1, 3))
        sums = np.where(valid, image.data[:height, :width], 0.).reshape(shape).sum(axis=(1, 3))
        values = np.divide(sums, weights, out=np.full_like(sums, np.nan), where=weights > 0)
        # Empty metadata lets different filters share the same sky reference.
        return frame(values, mask=weights < step * step, name=image.name)

    for index, source in enumerate(sources):
        image = read_frame(source)
        kind = str(image.header.get('IMAGETYP', '')).upper()
        if (any(token in kind for token in ('BIAS', 'ZERO', 'FLAT', 'DARK')) or
                image.name.casefold().startswith(('bias', 'flat', 'dark'))):
            raise ValueError(f'{image.name}: select science images, not calibration frames.')
        limit = max(image.data.shape) - 2
        if not isinstance(max_shift, (int, np.integer)) or not 1 <= max_shift <= limit:
            raise ValueError(f'Maximum shift must be between 1 and {limit} pixels.')
        target = (source_target(image).casefold(), source_filter(image), None if group_by_filter else source_exposure(image))
        if shared_reference:
            target = (source_target(image).casefold(), None, None)
        if corrected:
            if not (image.header.get('SCICOR') is True and image.header.get('FLATNORM') is True):
                raise ValueError(f'{image.name}: correct the science frames before alignment.')
        else:
            image = correct_science_frame(image, bias, flats)
        if not _has_registration_structure(image):
            raise ValueError(f'{image.name}: no structure for science alignment.')
        valid = ~image.mask
        background = np.median(image.data[valid])
        filled = np.where(valid, image.data, background)
        local = median_filter(filled, size=3)
        residual = filled - local
        noise = 1.4826 * np.median(np.abs(residual[valid]))
        isolated = residual > np.maximum(8. * noise, 2. * np.abs(local - background))
        # Exclude isolated detector defects only from shift measurements.
        # The full-resolution science image retains every original data pixel.
        measurement = frame(np.where(isolated, local, filled), mask=image.mask | isolated, name=image.name)
        step = max(1, int(np.ceil(max(image.data.shape) / 384)))
        small = registration_image(measurement, step)
        stars = _science_stars(measurement, small, step)
        if target not in references:
            # Validate the reference too, including targets with only one frame.
            _registration_array(small)
            references[target] = (image, small, step, stars, measurement)
            dy, dx = 0., 0.
            status = 'REFERENCE'
        else:
            reference, small_reference, reference_step, reference_stars, reference_measurement = references[target]
            compatible(reference, image)
            if step != reference_step:
                raise ValueError(f'{image.name}: image dimensions differ from the reference.')
            small_limit = min(small.data.shape) // 2 - 2
            search = min(small_limit, max(1, int(np.ceil(max_shift / step)) + 1))
            a, b = _registration_array(small_reference), _registration_array(small)

            # Avalia a correlação na região comum das imagens após um deslocamento candidato.
            def quality(offset):
                shifted = ndi_shift(b, offset, order=1, mode='constant', cval=0., prefilter=False)
                sy, sx = offset
                height, width = a.shape
                rows = slice(max(0, int(np.ceil(sy))), min(height, height + int(np.floor(sy))))
                cols = slice(max(0, int(np.ceil(sx))), min(width, width + int(np.floor(sx))))
                first, second = a[rows, cols], shifted[rows, cols]
                norm = np.linalg.norm(first) * np.linalg.norm(second)
                return float(np.sum(first * second) / norm) if norm else 0.

            stellar_shift = _science_star_shift(reference_stars, stars, max_shift, image.data.shape)
            try:
                if stellar_shift is not None:
                    sy, sx = np.asarray(stellar_shift) / step
                    # The matched constellation validates a shift even when
                    # a large pointing change leaves little image overlap.
                    score = 1.
                else:
                    sy, sx = estimate_translation(small_reference, small, search, True)
                    score = quality((sy, sx))
            except ValueError:
                score = 0.
            if score < .4 and step > 1:
                if target not in full_stars:
                    full_stars[target] = _science_stars(reference_measurement, reference_measurement, 1)
                stellar_shift = _science_star_shift(full_stars[target], _science_stars(measurement, measurement, 1),
                                                   max_shift, image.data.shape)
                if stellar_shift is not None:
                    sy, sx = np.asarray(stellar_shift) / step
                    score = 1.
            # Phase correlation may lock onto noise when exposure or filter
            # changes strongly. A stellar intensity peak gives a second estimate.
            if score < .4:
                try:
                    sy, sx = estimate_translation(small_reference, small, search, True, phase=False)
                except ValueError:
                    raise ValueError(f'{image.name}: no reliable alignment within {max_shift} pixels; '
                                     'increase the limit or select images of the same field.') from None
                score = quality((sy, sx))
            dy, dx = sy * step, sx * step
            if abs(dy) > max_shift or abs(dx) > max_shift:
                raise ValueError(f'{image.name}: shift exceeds {max_shift} pixels; increase it and retry.')
            if score < .4:
                raise ValueError(f'{image.name}: no reliable common star field for alignment.')
            image = translate_frame(image, (dy, dx))
            status = 'SHIFTED'
        image.header['ALIGN'] = 'TRANSLATION'
        image.header['ALNSTAT'] = status
        image.header['ALNREF'] = references[target][0].name
        image.header['ALNMAX'] = max_shift
        image.header['ALNSHFTY'] = dy
        image.header['ALNSHFTX'] = dx
        add_history(image.header, f'Science registration dy={dy:.4f}; dx={dx:.4f}')
        if output_folder is not None:
            path = output_folder / f'{index:05d}' / image.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(to_fits(image))
            aligned.append(path)
        else:
            aligned.append(image)
        offsets.append((dy, dx))
    return aligned, offsets


# Combinação de ciência e sequência completa de tratamento

# Combina as imagens de ciência por objeto, filtro e exposição sem repetir as correções aplicadas.
def combine_science_groups(sources, bias, flats, method='mean', aligned=False, corrected=False, group_by_filter=False):
    """Combine raw science groups and apply bias and the corrected flat response."""
    bias = read_frame(bias)
    flats = {key: read_frame(value) for key, value in flats.items()}
    if not flats:
        raise ValueError('Create master flats before combining science images.')
    groups = {}
    for source in sources:
        header = _source_header(source)
        name = source.name if isinstance(source, Frame) else Path(source).name
        kind = str(header.get('IMAGETYP', '')).upper()
        if any(token in kind for token in ('BIAS', 'ZERO', 'FLAT', 'DARK')) or name.casefold().startswith(('bias', 'flat', 'dark')):
            raise ValueError(f'{name}: select science images, not calibration frames.')
        if aligned and not (header.get('BIASCOR') is True and header.get('FLATCOR') is True
                            and header.get('ALIGN') == 'TRANSLATION'):
            raise ValueError(f'{name}: expected calibrated, aligned science frames.')
        if (corrected or aligned) and not (header.get('SCICOR') is True and header.get('FLATNORM') is True):
            raise ValueError(f'{name}: expected corrected science frames.')
        if not (aligned or corrected) and (header.get('BIASCOR') is True or header.get('FLATCOR') is True):
            raise ValueError(f'{name}: science corrections have already been applied; select original frames.')
        filter_name = source_filter(source)
        if filter_name is None:
            raise ValueError(f'{name}: filter is missing from the filename and FITS header.')
        target = source_target(source, header)
        groups.setdefault((target, filter_name, None if group_by_filter else source_exposure(source)), []).append(source)
    masters, names, skipped = {}, {}, []
    for (target, filter_name, exposure), selected in sorted(
            groups.items(), key=lambda item: (item[0][0], item[0][1],
                                             item[0][2] is None, item[0][2] or 0)):
        label = f'{target} · {filter_name} · {exposure:g} s' if exposure is not None else f'{target} · {filter_name} · unknown exposure'
        if group_by_filter:
            label = f'{target} · {filter_name}'
        if len(selected) < 2 and not group_by_filter:
            skipped.append((label, [source.name if isinstance(source, Frame) else Path(source).name for source in selected]))
            continue
        # Fixed bias subtraction and a positive flat response commute with both
        # median and mean. Applying them once avoids another full disk stack.
        master = combine(selected, method=method, sigma=5., allow_mixed_exposures=group_by_filter,
                         normalize_exposure=group_by_filter and (aligned or corrected)) if len(selected) > 1 else read_frame(selected[0])
        master.header['NCOMBINE'] = len(selected)
        master.header['COMBMETH'] = method
        master.header['FILTER'] = filter_name
        master.header['OBJECT'] = target
        flat = matching_master_flat(flats, master)
        if aligned or corrected:
            result = master
        else:
            result = correct_science_frame(master, bias, flats)
        if aligned:
            result.header['ALNSUBPX'] = True
            for index, source in enumerate(selected):
                header = _source_header(source)
                result.header[f'ALN{index:03d}Y'] = float(header['ALNSHFTY'])
                result.header[f'ALN{index:03d}X'] = float(header['ALNSHFTX'])
        result.header['IMAGETYP'] = 'MASTER_SCIENCE'
        result.header['BIASFILE'] = bias.name
        result.header['FLATFILE'] = flat.name
        token = re.sub(r'[^a-z0-9_-]+', '_', target.casefold()).strip('_') or 'science'
        suffix = f'_{exposure:g}s' if exposure is not None else ''
        result.name = f'master_science_{token}_{filter_name.casefold()}{suffix}_x{len(selected)}.fits'
        masters[label] = result
        names[label] = [source.name if isinstance(source, Frame) else Path(source).name for source in selected]
    if not masters:
        raise ValueError('No science filter/exposure group contains at least two images.')
    return masters, names, skipped


# Executa a sequência de calibração, subtração de fundo, alinhamento e combinação escolhida.
def run_workflow(current, others=(), *, bias_sources=(), bias_master=None,
                 flat_sources=(), flat_master=None, background=False,
                 background_sigma=3.0, alignment='None', max_shift=100,
                 max_angle=5.0, method='median', rejection_sigma=5.0,
                 bias_method='mean', flat_method='mean', return_intermediates=False,
                 flat_subtract_bias=True):
    """Build masters, calibrate each science frame, subtract sky, align, combine."""
    if bias_sources and bias_master is not None or flat_sources and flat_master is not None:
        raise ValueError('Choose either source frames or an existing master for each calibration.')
    if alignment not in ('None', 'Translation', 'Translation + rotation'):
        raise ValueError('Invalid alignment mode.')
    master_bias = combine_bias_frames(bias_sources, bias_method) if bias_sources else (
        read_frame(bias_master) if bias_master is not None else None)
    flat_inputs = None
    flat_versions = None
    if flat_sources:
        plain, flat_inputs = combine_flat_groups(flat_sources, method=flat_method, return_sources=True)
        flat_versions = {False: plain}
        if master_bias is not None:
            flat_versions[True] = combine_flat_groups(flat_sources, bias=master_bias, method=flat_method)
        master_flats = flat_versions.get(flat_subtract_bias, plain)
    else:
        master_flats = None
    existing_flat = read_frame(flat_master) if flat_master is not None else None
    sources = [current, *others]
    calibrated = []
    for source in sources:
        image = read_frame(source)
        if master_flats:
            filter_name = source_filter(image)
            candidates = [master for master in master_flats.values()
                          if (filter_name is None and len(master_flats) == 1 or
                              master.header.get('FILTER') == filter_name)]
            if not candidates:
                raise ValueError(f'{image.name}: no master flat for FILTER {filter_name or "unknown"}.')
            if len(candidates) > 1:
                image_exposure = combination_exposure(image.header)
                matching = [master for master in candidates
                            if combination_exposure(master.header) == image_exposure]
                candidates = matching or candidates
            best_count = max(master.header['NCOMBINE'] for master in candidates)
            best = [master for master in candidates
                    if master.header['NCOMBINE'] == best_count]
            if len(best) != 1:
                raise ValueError(f'{image.name}: several master flats match FILTER '
                                 f'{filter_name}; choose one exposure group.')
            selected_flat = best[0]
        else:
            selected_flat = existing_flat
        image = calibrate(image, bias=master_bias, flat=selected_flat,
                          flat_min=0.05)
        if background:
            image = subtract_background(image, background_sigma)
        calibrated.append(image)
    offsets = [(0.0, 0.0, 0.0)]
    if len(calibrated) > 1 and alignment != 'None':
        for index in range(1, len(calibrated)):
            calibrated[index], offset = align_frame(
                calibrated[0], calibrated[index], max_shift,
                max_angle if alignment == 'Translation + rotation' else 0.0)
            offsets.append(offset)
    result = (combine(calibrated, method=method, sigma=rejection_sigma)
              if len(calibrated) > 1 else calibrated[0])
    if len(calibrated) > 1 and alignment != 'None':
        result.header['ALIGN'] = ('ROTATION' if alignment == 'Translation + rotation'
                                  else 'TRANSLATION', 'Registration before combination')
        for index, (angle, dy, dx) in enumerate(offsets[:100]):
            result.header[f'ALN{index:03d}R'] = (angle, 'Applied rotation in degrees')
            result.header[f'ALN{index:03d}Y'] = (dy, 'Applied row shift')
            result.header[f'ALN{index:03d}X'] = (dx, 'Applied column shift')
    count_suffix = f'_x{len(calibrated)}' if len(calibrated) > 1 else ''
    result.name = Path(read_frame(current).name).stem + f'_workflow{count_suffix}.fits'
    add_history(result.header, 'Workflow: bias -> flat -> background -> alignment -> combination')
    return (result, {'bias': master_bias, 'flats': master_flats,
                     'flat_versions': flat_versions,
                     'flat_inputs': flat_inputs}) if return_intermediates else result



# Limites de visualização e contraste opcional

# Identifica a família da imagem para escolher os limites fixos de visualização.
def display_group(folder, name, header, hdu=0):
    folder = str(Path(folder).resolve())
    kind = str(header.get('IMAGETYP', '')).upper()
    lower = name.casefold()
    if 'BIAS' in kind or lower.startswith(('bias', 'master_bias')):
        return folder, 'bias'
    if header.get('NORMALIZ') is True and 'FLAT' in kind:
        return folder, 'flat_response'
    if 'FLAT' in kind or lower.startswith(('flat', 'master_flat', 'corrected_flat')):
        tokens = re.split(r'[^a-z0-9]+', lower)
        color = str(header.get('FILTER', '')).strip().casefold()
        if not color:
            color = next((token for token in tokens if token in ('blue', 'red', 'green', 'black')), '')
        exposure = header.get('EXPTIME', header.get('EXPOSURE'))
        return folder, 'flat', color, exposure
    if header.get('SCICOR') is True and header.get('FLATNORM') is False:
        return folder, 'science_ratio'
    return folder, 'image', name, hdu


# Escolhe os limites fixos de visualização adequados a bias, ciência corrigida ou outras imagens.
def default_display_limits(group):
    if group[1] == 'bias':
        return 500., 630.
    if group[1] == 'flat_response':
        return 0., 2.
    return (0., 1.) if group[1] == 'science_ratio' else (0., 65535.)


# Determina a escala de visualização e aplica contraste apenas quando está ativado.
def resolve_display_limits(values, limits=(0., 65535.), contrast=False, reference=None):
    """Use the same optional screen contrast for the viewer and PNG exports."""
    low, high = map(float, limits)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError('Display maximum must be greater than minimum.')
    if contrast:
        reference = np.asarray(values if reference is None else reference, dtype=float)
        finite = reference[np.isfinite(reference)]
        if finite.size:
            contrast_low, contrast_high = np.percentile(finite, [1, 99])
            if contrast_high > contrast_low:
                low, high = float(contrast_low), float(contrast_high)
    return low, high



# Exportação de imagens e arquivos ZIP

# Obtém o nome da imagem a usar no ficheiro exportado.
def image_name(source):
    return source[0] if isinstance(source, tuple) else source.name


# Obtém os bytes FITS do resultado preservando os seus valores científicos.
def fits_data(source):
    if isinstance(source, tuple):
        return source[1]
    return to_fits(source) if isinstance(source, Frame) else Path(source).read_bytes()


# Converte a extensão do nome da imagem para PNG.
def png_name(name):
    return re.sub(r'\.(?:fits|fit|fts)(?:\.gz|\.fz)?$|\.fz$', '',
                  Path(name).name, flags=re.IGNORECASE) + '.png'


# Converte contagens ou razões de ciência em PNG de 16 bits sem contraste automático.
def to_png(source):
    """Write counts, or fixed 0–1 science ratios, without automatic contrast."""
    image = source if isinstance(source, Frame) else read_frame(
        source[1] if isinstance(source, tuple) else source)
    valid = ~image.mask & np.isfinite(image.data)
    pixels = np.zeros(image.data.shape, dtype=np.uint16)
    if valid.any():
        scale = (65535. / 2. if image.header.get('NORMALIZ') is True and 'FLAT' in str(image.header.get('IMAGETYP', ''))
                 else 65535. if image.header.get('SCICOR') is True and image.header.get('FLATNORM') is False else 1.)
        pixels[valid] = np.clip(np.rint(image.data[valid] * scale), 0, 65535).astype(np.uint16)
    output = BytesIO()
    Image.fromarray(np.flipud(pixels)).save(output, format='PNG')
    return output.getvalue()


# Exporta a imagem para PNG com a mesma escala e contraste do visualizador.
def preview_png(source, contrast=False):
    """Match the image viewer's fixed scale or its explicitly enabled contrast."""
    image = source if isinstance(source, Frame) else read_frame(source[1] if isinstance(source, tuple) else source)
    limits = default_display_limits(display_group('.', image_name(source), image.header))
    limits = resolve_display_limits(image.display(), limits, contrast)
    return to_display_png(image, limits)


# Reúne as imagens resultantes num ZIP de FITS ou PNG.
def image_archive(images, file_format='fits', display_png=False, contrast=False):
    if file_format not in ('fits', 'png'):
        raise ValueError('Choose FITS or PNG.')
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for image in images.values():
            name = image_name(image)
            if file_format == 'png':
                archive.writestr(png_name(name), preview_png(image, contrast) if display_png else to_png(image))
            else:
                archive.writestr(name, fits_data(image))
    return output.getvalue()


# Converte a escala de visualização em píxeis PNG sem alterar os dados científicos.
def to_display_png(image, limits):
    """Export screen grayscale to PNG without changing scientific pixel arrays."""
    low, high = resolve_display_limits(image.data, limits)
    valid = ~image.mask & np.isfinite(image.data)
    pixels = np.zeros(image.data.shape, dtype=np.uint8)
    with np.errstate(over='ignore', invalid='ignore'):
        gray = np.clip((image.data[valid] - low) / (high - low), 0., 1.)
        # Match JavaScript Math.round in the viewer, including half levels.
        pixels[valid] = np.floor(gray * 255. + .5).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(np.flipud(pixels)).save(output, format='PNG')
    return output.getvalue()


def flat_channel(image):
    """Display channel only; unassigned filters retain a neutral grayscale preview."""
    return {'Red': 'R', 'Green': 'G', 'Blue': 'B', 'black': 'Gray'}.get(source_filter(image), 'Gray')


def channel_png(image, limits, thumbnail=False):
    """Place display intensity in the filter channel, preserving scientific data."""
    gray = np.asarray(Image.open(BytesIO(to_display_png(image, limits))))
    channel = flat_channel(image)
    pixels = np.zeros((*gray.shape, 3), dtype=np.uint8)
    if channel == 'Gray':
        pixels[:] = gray[..., None]
    else:
        pixels[..., 'RGB'.index(channel)] = gray
    preview = Image.fromarray(pixels)
    if thumbnail:
        preview.thumbnail((480, 320), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (480, 320))
        canvas.paste(preview, ((480-preview.width)//2, (320-preview.height)//2))
        preview = canvas
    output = BytesIO()
    preview.save(output, format='PNG')
    return output.getvalue()


# Reúne os master flats num arquivo ZIP de imagens FITS.
def flat_archive(masters):
    archive = BytesIO()
    with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
        for master in masters.values():
            bundle.writestr(master.name, to_fits(master))
    return archive.getvalue()


# Exporta masters e versões corrigidas em subpastas com e sem contraste.
def calibration_archive(versions, file_format='fits'):
    """Package masters with and without display contrast, including both flat versions.

    FITS packages contain original FITS plus contrasted PNG previews. PNG packages
    contain screen-matching previews with and without contrast. Scientific FITS are
    never contrast-scaled. Folder spelling follows the requested download layout.
    """
    if file_format not in ('fits', 'png'):
        raise ValueError('Choose FITS or PNG.')
    output = BytesIO()
    groups = dict.fromkeys(key for images in versions.values() for key in images)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('with_constrast/', b'')
        archive.writestr('no_contrast/', b'')
        for key in groups:
            sources = [images[key] for images in versions.values() if key in images]
            images = [source if isinstance(source, Frame) else read_frame(
                source[1] if isinstance(source, tuple) else source) for source in sources]
            reference = [image.display() for image in images]
            first = images[0]
            fixed_limits = default_display_limits(display_group('.', image_name(sources[0]), first.header))
            limits = resolve_display_limits(first.display(), fixed_limits, True, reference)
            mixed_normalization = any(image.header.get('NORMALIZ') is True for image in images)
            for source, image in zip(sources, images):
                if mixed_normalization:
                    fixed_limits = default_display_limits(display_group('.', image.name, image.header))
                    limits = resolve_display_limits(image.display(), fixed_limits, True)
                name = Path(image_name(source)).name
                archive.writestr(f'with_constrast/{png_name(name)}', to_display_png(image, limits))
                if 'FLAT' in str(image.header.get('IMAGETYP', '')).upper() or 'flat' in name.casefold():
                    colored_name = png_name(name)[:-4] + f'_channel_{flat_channel(image)}.png'
                    for folder, scale in [('with_constrast', limits), ('no_contrast', fixed_limits)]:
                        archive.writestr(f'{folder}/{colored_name}', channel_png(image, scale))
                if file_format == 'fits':
                    archive.writestr(f'no_contrast/{name}', fits_data(source))
                else:
                    # Match Contrast off in the viewer (bias uses 500–630),
                    # rather than making a dark count-valued 16-bit image.
                    archive.writestr(f'no_contrast/{png_name(name)}', to_display_png(image, fixed_limits))
    return output.getvalue()



# Operações completas executadas pelos processos de tratamento

# Produz o master bias e as versões dos master flats com e sem correção de bias.
def combine_flats_job(flat_sources, bias_sources, bias_method, flat_method, master_bias=None):
    """Compute both flat versions and their archives within the same time limit."""
    if master_bias is None:
        master_bias = combine_bias_frames(bias_sources, bias_method)
    plain, names = combine_flat_groups(flat_sources, method=flat_method, return_sources=True)
    versions = {False: plain, True: combine_flat_groups(flat_sources, bias=master_bias, method=flat_method)}
    archives = {mode: image_archive(masters) for mode, masters in versions.items()}
    return master_bias, versions, names, archives


# Executa o tratamento completo e prepara os masters e os seus arquivos de exportação.
def run_workflow_job(*args, **kwargs):
    result, intermediates = run_workflow(*args, **kwargs, return_intermediates=True)
    intermediates['flat_archives'] = {
        mode: image_archive(masters)
        for mode, masters in (intermediates['flat_versions'] or {}).items()}
    return result, intermediates
