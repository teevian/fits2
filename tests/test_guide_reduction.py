"""Numerical regressions for the Astropy guide calibration recipe."""
import numpy as np
import pytest
from astropy.io import fits
from imageProcessing import (frame, combine_bias_frames, combine_flat_groups,
                             correct_science_frame, combine_science_groups, calibration_archive)


def make(data, name, kind, exposure=0., color=None):
    header = fits.Header({'IMAGETYP': kind, 'EXPTIME': exposure, 'BUNIT': 'adu'})
    if color:
        header['FILTER'] = color
    return frame(data, header, name=name)


def test_bias_rejects_transient_outlier_and_retains_count_scale():
    offsets = [-2., -1., 0., 1., 2., -2., -1., 0., 1., 2., 0.]
    inputs = [make(np.full((8, 8), 500. + n), f'bias{i}.fits', 'BIAS') for i, n in enumerate(offsets)]
    inputs[-1].data[2, 3] = 60000.
    result = combine_bias_frames(inputs)
    np.testing.assert_allclose(result.data, 500.)
    assert result.header['COMBMETH'] == 'mean'
    assert result.header['CLIPSIG'] == 5.
    assert result.header['CLIPDEV'] == 'mad_std'
    assert inputs[-1].data[2, 3] == 60000.


def test_variable_flat_illumination_does_not_change_science_flux():
    y, x = np.indices((20, 20))
    response = .5 + x / 20. + y / 40.
    response /= np.median(response)
    bias_data = 500. + 3. * x
    bias = make(bias_data, 'master_bias.fits', 'MASTER_BIAS')
    inputs = [make(bias_data + level * response, f'flat_Blue_{i}.fits', 'FLAT', 2., 'Blue')
              for i, level in enumerate([1000., 2000., 5000., 10000., 15000., 20000., 12000.])]
    inputs[-1].data[2, 3] += 30000.  # transient only in one input
    for image in inputs:
        image.data[1, 1] = bias_data[1, 1]  # no usable flat response
    flats = combine_flat_groups(inputs, bias)
    flat = flats['Blue']
    valid = ~flat.mask
    assert flat.mask[1, 1]
    np.testing.assert_allclose(flat.data[valid], response[valid] / np.median(response[valid]))
    assert flat.header['NORMALIZ'] and flat.header['BIASCOR']
    assert flat.header['BUNIT'] == '1'
    assert np.median(flat.data[valid]) == pytest.approx(1.)
    sky = 100. + 500. * np.exp(-((x - 10.)**2 + (y - 10.)**2) / 5.)
    source = make(bias_data + sky * response, 'target_Blue_1.fits', 'OBJECT', 60., 'Blue')
    corrected = correct_science_frame(source, bias, flats)
    np.testing.assert_allclose(corrected.data[valid], (sky * np.median(response[valid]))[valid])
    assert corrected.header['BUNIT'] == 'adu' and corrected.header['FLATNORM']
    assert corrected.mask[1, 1]
    assert corrected.header['DARKCOR'] is False
    with pytest.raises(ValueError, match='already'):
        correct_science_frame(corrected, bias, flats)


def test_final_science_combination_clips_outlier_preserves_counts_and_exposures():
    bias = make(np.full((8, 8), 500.), 'bias.fits', 'MASTER_BIAS')
    flat = make(np.ones((8, 8)), 'flat_Blue.fits', 'CORRECTED_FLAT', 2., 'Blue')
    flat.header['BIASCOR'] = True
    corrected = []
    for exposure in (30., 60.):
        for index, offset in enumerate([-2., -1., 0., 1., 2., 0.]):
            source = make(np.full((8, 8), 500. + exposure * 10. + offset),
                          f'target_Blue_{exposure}_{index}.fits', 'OBJECT', exposure, 'Blue')
            source.header['OBJECT'] = 'target'
            if index == 5:
                source.data[3, 4] = 60000.
            corrected.append(correct_science_frame(source, bias, {'Blue': flat}))
    masters, inputs, skipped = combine_science_groups(corrected, bias, {'Blue': flat}, corrected=True)
    assert len(masters) == 2 and not skipped
    for key, result in masters.items():
        np.testing.assert_allclose(result.data, result.header['EXPTIME'] * 10.)
        assert result.header['BUNIT'] == 'adu'
        assert len(inputs[key]) == 6


def test_normalized_flat_export_uses_response_scale_separate_from_raw_counts():
    from io import BytesIO
    from zipfile import ZipFile
    from PIL import Image
    from imageProcessing import default_display_limits, display_group
    bias = make(np.full((4, 4), 500.), 'bias.fits', 'MASTER_BIAS')
    response = np.tile([.5, 1., 1., 1.5], (4, 1))
    sources = [make(500. + response * level, f'flat_Blue_{i}.fits', 'FLAT', 1., 'Blue')
               for i, level in enumerate((1000., 2000.))]
    versions = {False: combine_flat_groups(sources), True: combine_flat_groups(sources, bias)}
    master = versions[True]['Blue']
    assert default_display_limits(display_group('.', master.name, master.header)) == (0., 2.)
    with ZipFile(BytesIO(calibration_archive(versions, 'png'))) as archive:
        name = 'no_contrast/' + master.name.replace('.fits', '.png')
        with Image.open(BytesIO(archive.read(name))) as png:
            np.testing.assert_array_equal(np.asarray(png), np.flipud(np.floor(response / 2 * 255 + .5)))
        with Image.open(BytesIO(archive.read(name.replace('no_contrast', 'with_constrast')))) as png:
            assert np.asarray(png).max() == 255
