"""Linear-image aperture measurements. No display transforms enter this module."""
import warnings

import numpy as np
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit, OptimizeWarning

MAG_FACTOR = 2.5 / np.log(10)


def moffat(r, amplitude, alpha, beta):
    return amplitude * (1 + (r / alpha) ** 2) ** (-beta)


def measure(image, x, y, radius=6., inner=10., outer=16., gain=None,
            calibration_error=0., saturation=None, flux_error=None):
    """Whole-pixel aperture; clipped mean sky. Coordinates are zero-based x,y.

    Errors for combined/resampled images require an externally established total
    aperture uncertainty; a single-frame Poisson model is never applied to stacks.
    """
    data = np.asarray(image.data, dtype=float)
    if data.ndim != 2 or str(image.header.get('BUNIT', 'adu')).strip().lower() not in ('adu', 'count', 'counts'):
        raise ValueError('A fotometria exige uma imagem linear 2D em ADU/counts.')
    params = [x, y, radius, inner, outer, calibration_error]
    if not np.all(np.isfinite(params)) or not 0 < radius < inner < outer or calibration_error < 0:
        raise ValueError('Usa 0 < abertura < raio interior < raio exterior e parâmetros finitos.')
    for value in (gain, saturation, flux_error):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError('Ganho, saturação e erro externo devem ser positivos.')
    h, w = data.shape
    if x - outer < 0 or y - outer < 0 or x + outer > w - 1 or y + outer > h - 1:
        raise ValueError('O anel de fundo sai da imagem; reduz os raios ou muda o centro.')
    x0, x1 = int(np.floor(x - outer)), int(np.ceil(x + outer)) + 1
    y0, y1 = int(np.floor(y - outer)), int(np.ceil(y + outer)) + 1
    cut = data[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    valid = np.isfinite(cut) & ~image.mask[y0:y1, x0:x1]
    aperture = rr <= radius
    if not np.all(valid[aperture]):
        raise ValueError('A abertura contém píxeis inválidos/mascarados; o fluxo seria incompleto.')
    sky_raw = cut[valid & (rr >= inner) & (rr <= outer)]
    sky = sigma_clip(sky_raw, sigma=3, maxiters=5, cenfunc='median', stdfunc='mad_std').compressed()
    if sky.size < 20:
        raise ValueError('O anel precisa de pelo menos 20 píxeis válidos após rejeição.')
    background, variance = float(np.mean(sky)), float(np.var(sky, ddof=1))
    n, m = int(aperture.sum()), int(sky.size)
    flux = float(np.sum(cut[aperture] - background))
    flags = []
    if saturation is None:
        flags.append('Saturação/linearidade não verificadas: indicar limite validado do detetor.')
    elif np.any(cut[aperture] >= saturation):
        raise ValueError('Fonte saturada ou fora da linearidade: fotometria rejeitada.')
    stack = (image.header.get('NCOMBINE', 1) > 1 or image.header.get('COMBINED', False)
             or image.header.get('ALNSUBPX', False) or image.header.get('ALIGN') in ('TRANSLATION', 'ROTATION'))
    error = None
    if flux_error is not None:
        error = float(flux_error)
        flags.append('Erro total externo: deve incluir covariância, ruído de fonte/fundo e calibração.')
    elif stack:
        flags.append('Stack/interpolação: S/N e erros indisponíveis sem variância/covariância ou erro total externo.')
    elif gain is not None and flux > 0:
        error = float(np.sqrt(flux / gain + n * (1 + n / m) * variance + calibration_error ** 2))
        flags.append('Erro aproximado: píxeis independentes, fundo local uniforme; clipping e flat podem introduzir sistemáticos.')
    else:
        flags.append('Ganho ausente ou fluxo não positivo: erro fotométrico indisponível.')
    exposure = image.header.get('EXPTIME', image.header.get('EXPOSURE'))
    try:
        exposure = float(exposure)
    except (TypeError, ValueError):
        exposure = np.nan
    rate = flux / exposure if np.isfinite(exposure) and exposure > 0 else None
    magnitude = float(-2.5 * np.log10(rate)) if rate is not None and rate > 0 else None
    if magnitude is None:
        flags.append('Magnitude indisponível: exposição ausente/inválida ou fluxo não positivo.')
    snr = flux / error if error is not None and error > 0 and flux > 0 else None
    mag_error = MAG_FACTOR / snr if snr is not None and snr >= 5 else None
    if snr is not None and snr < 5:
        flags.append('S/N < 5: aproximação gaussiana do erro de magnitude não apresentada.')
    radial, growth = [], []
    for lo in range(int(np.ceil(inner))):
        ring = valid & (rr >= lo) & (rr < lo + 1)
        values = cut[ring] - background
        if values.size:
            radial.append((float(np.mean(rr[ring])), float(np.mean(values)),
                           float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan))
        disk = valid & (rr <= lo + 1)
        growth.append((lo + 1, float(np.sum(cut[disk] - background))))
    radial = np.asarray(radial)
    fit = None
    if len(radial) >= 5 and radial[:, 1].max() > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                pars, _ = curve_fit(moffat, radial[:, 0], radial[:, 1],
                                    p0=[radial[:, 1].max(), 2., 2.5],
                                    bounds=([0, .1, 1.01], [np.inf, outer * 2, 20]), maxfev=5000)
            fit = dict(amplitude=float(pars[0]), alpha=float(pars[1]), beta=float(pars[2]),
                       fwhm_px=float(2 * pars[1] * np.sqrt(2 ** (1 / pars[2]) - 1)))
            if pars[2] > 19.99 or pars[2] < 1.02 or pars[1] < .11 or pars[1] > outer * 1.99:
                flags.append('Moffat no limite do ajuste: parâmetros pouco determinados; verificar perfil e resíduos.')
        except (ValueError, RuntimeError, OptimizeWarning, FloatingPointError):
            flags.append('Ajuste Moffat não convergiu.')
    return dict(x=float(x), y=float(y), radius=radius, inner=inner, outer=outer,
                flux_adu=flux, flux_error_adu=error, exposure_s=exposure if np.isfinite(exposure) else None,
                rate_adu_s=rate, m_inst=magnitude, sigma_m=mag_error, snr=snr,
                sky_adu=background, sky_std_adu=float(np.sqrt(variance)), n_aperture=n, n_sky=m,
                n_sky_rejected=int(sky_raw.size - m), fit=fit, flags=flags,
                radial=radial, growth=np.asarray(growth), sky_sample=sky, cutout=cut,
                cutout_mask=~valid, origin=(x0, y0))


def stellar_parameters(b, v, ebv, distance_pc, av, bc_v, mass=None):
    """Conditional colour-temperature approximation, not stellar classification/age."""
    values = [b, v, ebv, distance_pc, av, bc_v]
    if not np.all(np.isfinite(values)) or distance_pc <= 0 or ebv < 0 or av < 0:
        raise ValueError('Distância positiva e extinções não negativas são obrigatórias.')
    color = b - v - ebv
    if not -.4 <= color <= 2.:
        raise ValueError('Cor fora do intervalo adotado −0.4 ≤ (B−V)₀ ≤ 2.0; usar modelo atmosférico adequado.')
    temperature = 4600 * (1 / (.92 * color + 1.7) + 1 / (.92 * color + .62))
    mv = v - 5 * np.log10(distance_pc / 10) - av
    luminosity = 10 ** (.4 * (4.74 - mv - bc_v))
    if mass is not None and (not np.isfinite(mass) or mass <= 0):
        raise ValueError('A massa independente deve ser positiva.')
    return dict(color_bv0=float(color), temperature_K=float(temperature), M_V=float(mv),
                luminosity_solar=float(luminosity),
                lifetime_years=float(1e10 * mass / luminosity) if mass is not None else None)


def calibrate_with_reference(image, target, ref_x, ref_y, magnitude, magnitude_error=None,
                             gain=None, saturation=None, reference_flux_error=None,
                             correction=0., correction_error=0., passband_confirmed=False):
    """Single comparison star in the *same* image, with independent-error approximation.

    The equal exposure cancels even when EXPTIME is missing. A catalogue-anchored
    magnitude is distinct from a validated standard-system transformation.
    """
    if not np.all(np.isfinite([ref_x, ref_y, magnitude, correction, correction_error])) or correction_error < 0:
        raise ValueError('Coordenadas, magnitude e correção devem ser finitas; erro da correção não negativo.')
    if magnitude_error is not None and (not np.isfinite(magnitude_error) or magnitude_error < 0):
        raise ValueError('A incerteza tabelada deve ser não negativa ou desconhecida.')
    separation = float(np.hypot(target['x']-ref_x, target['y']-ref_y))
    if separation <= target['radius'] + target['outer']:
        raise ValueError('Alvo e referência demasiado próximos: as aberturas contaminam os anéis de fundo. Escolhe outra referência ou reduz os raios.')
    reference = measure(image, ref_x, ref_y, target['radius'], target['inner'], target['outer'],
                        gain=gain, saturation=saturation, flux_error=reference_flux_error)
    if target['flux_adu'] <= 0 or reference['flux_adu'] <= 0:
        raise ValueError('Alvo e referência precisam de fluxo líquido positivo.')
    difference = float(-2.5*(np.log10(target['flux_adu'])-np.log10(reference['flux_adu'])))
    calibrated = float(magnitude + difference + correction)
    flags = ['Fotometria diferencial: extinção cinzenta comum aproximadamente cancelada; diferenças de cor, resposta do filtro e abertura podem persistir.']
    sigma = None
    shared_sky = separation <= 2*target['outer']
    if shared_sky:
        flags.append('Anéis de fundo sobrepostos: erro diferencial não calculado sem covariância.')
    elif target['sigma_m'] is not None and reference['sigma_m'] is not None and magnitude_error is not None:
        sigma = float(np.sqrt(target['sigma_m']**2 + reference['sigma_m']**2 + magnitude_error**2 + correction_error**2))
        flags.append('Erro diferencial de primeira ordem: assume erros independentes e exclui sistemáticos não quantificados.')
    else:
        flags.append('Erro diferencial indisponível: falta erro do alvo, da referência ou da magnitude tabelada.')
    if not passband_confirmed:
        flags.append('Magnitude ancorada na referência, sem transformação para banda padrão confirmada; não utilizada automaticamente no H–R.')
    return dict(m_calibrated=calibrated, sigma_m_calibrated=sigma,
                m_std=calibrated if passband_confirmed else None,
                sigma_m_std=sigma if passband_confirmed else None,
                differential_mag=difference, derived_zero_point=float(magnitude-reference['m_inst']) if reference['m_inst'] is not None else None,
                reference=reference, reference_magnitude=float(magnitude), reference_magnitude_error=magnitude_error,
                differential_correction=float(correction), differential_correction_error=float(correction_error),
                passband_confirmed=bool(passband_confirmed), calibration_flags=flags)


def stellar_uncertainties(parameters, distance_pc, sigma_b, sigma_v, sigma_ebv,
                          sigma_distance, sigma_av, sigma_bc, mass=None, sigma_mass=0.):
    """First-order independent-input errors; no intrinsic/model scatter included.

    B and V must have calibrated errors. Correlated extinction/distance/zero
    points need a joint model instead; HR coordinates themselves are correlated.
    """
    errors = [sigma_b, sigma_v, sigma_ebv, sigma_distance, sigma_av, sigma_bc, sigma_mass]
    if not np.all(np.isfinite(errors)) or min(errors) < 0 or distance_pc <= 0:
        raise ValueError('Incertezas finitas não negativas e distância positiva são necessárias.')
    color = parameters['color_bv0']
    sigma_color = float(np.sqrt(sigma_b**2 + sigma_v**2 + sigma_ebv**2))
    derivative = -4600 * .92 * ((.92*color + 1.7)**-2 + (.92*color + .62)**-2)
    sigma_mv = float(np.sqrt(sigma_v**2 + (5/np.log(10)*sigma_distance/distance_pc)**2 + sigma_av**2))
    sigma_l = float(parameters['luminosity_solar'] * .4 * np.log(10) * np.hypot(sigma_mv, sigma_bc))
    sigma_t = None
    if parameters['lifetime_years'] is not None and mass is not None and mass > 0:
        sigma_t = float(parameters['lifetime_years'] * np.hypot(sigma_mass/mass, sigma_l/parameters['luminosity_solar']))
    return dict(sigma_color_bv0=sigma_color, sigma_temperature_K=float(abs(derivative)*sigma_color),
                sigma_M_V=sigma_mv, sigma_luminosity_solar=sigma_l, sigma_lifetime_years=sigma_t)
