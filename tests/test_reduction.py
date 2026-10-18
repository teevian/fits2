from io import BytesIO

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import shift

from imageProcessing import (frame, read_frame, combine, calibrate, mask_image,
                       read_mask, to_fits, estimate_translation, translate_frame)
from imageProcessing import export_crop


def image(data, **metadata):
    return frame(np.asarray(data, dtype=float), fits.Header(metadata))


def test_master_bias_clips_transient_and_handles_masked_samples(tmp_path):
    sources = []
    for i in range(7):
        data = np.full((3, 4), 100.)
        data[0, 0] = 10000 if i == 6 else 100
        data[2, 3] = np.nan
        f = image(data, IMAGETYP='BIAS', EXPTIME=0.)
        path = tmp_path / f'bias{i}.fits'
        path.write_bytes(to_fits(f))
        sources.append(path)
    result = combine(sources, sigma=5, kind='bias')
    assert result.data[0, 0] == 100
    assert result.mask[2, 3]
    assert result.header['NCOMBINE'] == 7
    assert result.name == 'master_bias_x7.fits'
    assert result.header['IMAGETYP'] == 'MASTER_BIAS'
    assert result.header['EXPTIME'] == 0


def test_mean_and_median_combination_and_input_immutability():
    a = image([[1, 3], [5, 7]], FILTER='Red')
    b = image([[3, 7], [9, 11]], FILTER='Red')
    c = image([[100, 100], [100, 100]], FILTER='Red')
    np.testing.assert_allclose(combine([a, b], sigma=None).data, [[2, 5], [7, 9]])
    np.testing.assert_allclose(combine([a, b, c], method='median', sigma=None).data, b.data)
    np.testing.assert_array_equal(a.data, [[1, 3], [5, 7]])


def test_combined_filename_reports_actual_input_count():
    sources = [image(np.full((2, 2), i)) for i in range(12)]
    result = combine(sources, sigma=None)
    assert result.name == 'combined_x12.fits'
    assert result.header['NCOMBINE'] == 12


def test_combination_requires_matching_exposure():
    first = image(np.ones((2, 2)), EXPTIME=8.)
    second = image(np.ones((2, 2)), EXPTIME=10.)
    with pytest.raises(ValueError, match='exposure 10.0 s differs'):
        combine([first, second], sigma=None)
    with pytest.raises(ValueError, match='exposure None s differs'):
        combine([first, image(np.ones((2, 2)))], sigma=None)
    assert combine([first, image(np.ones((2, 2)), EXPTIME=8.)], sigma=None).header['EXPTIME'] == 8.


def test_flats_are_calibrated_and_normalized_before_combination():
    response = np.array([[0.5, 1], [1, 1.5]])
    bias = image(np.full((2, 2), 100))
    sources = [image(response*level + 100, IMAGETYP='FLAT', FILTER='Blue') for level in [1000, 3000, 7000]]
    result = combine(sources, kind='flat', bias=bias)
    np.testing.assert_allclose(result.data, response)
    assert result.header['BUNIT'] == '1'
    assert result.header['BIASCOR']


def test_science_bias_dark_flat_recovers_known_signal():
    response = np.array([[0.5, 1], [1, 1.5]])
    bias = image(np.full((2, 2), 100))
    dark = image(np.full((2, 2), 5), EXPTIME=10.)
    flat = image(response, FILTER='Blue', BUNIT='1')
    raw = image(200*response + 100 + 10, EXPTIME=20., FILTER='Blue')
    result = calibrate(raw, bias=bias, dark=dark, flat=flat, scale_dark=True)
    np.testing.assert_allclose(result.data, 200.)
    assert not result.mask.any()
    assert result.header['BIASCOR'] and result.header['DARKCOR'] and result.header['FLATCOR']


def test_dark_with_bias_cannot_be_scaled_or_double_subtracted():
    raw = image(np.full((2, 2), 250), EXPTIME=10.)
    dark = image(np.full((2, 2), 110), EXPTIME=10.)
    bias = image(np.full((2, 2), 100))
    np.testing.assert_allclose(calibrate(raw, dark=dark, dark_contains_bias=True).data, 140)
    with pytest.raises(ValueError, match='containing bias'):
        calibrate(raw, bias=bias, dark=dark, dark_contains_bias=True)
    with pytest.raises(ValueError, match='containing bias'):
        calibrate(raw, dark=dark, dark_contains_bias=True, scale_dark=True)
    with pytest.raises(ValueError, match='exposures differ'):
        calibrate(image(np.ones((2,2)), EXPTIME=20.), dark=dark)


def test_bad_flat_and_source_mask_are_propagated():
    raw = image(np.ones((2, 3))*100)
    raw.mask[1, 2] = True
    flat = image([[0, -1, np.nan], [1, 1, 1]])
    result = calibrate(raw, flat=flat)
    np.testing.assert_array_equal(result.mask, [[True, True, True], [False, False, True]])
    assert np.isfinite(result.data[1, 0])


def test_incompatible_shapes_filters_and_binning_rejected():
    a = image(np.ones((2, 2)), FILTER='Red', XBINNING=1)
    for b, message in [(image(np.ones((3, 3))), 'dimensions'),
                       (image(np.ones((2, 2)), FILTER='Blue'), 'FILTER'),
                       (image(np.ones((2, 2)), XBINNING=2), 'XBINNING')]:
        with pytest.raises(ValueError, match=message):
            calibrate(a, flat=b)
    with pytest.raises(ValueError, match='IMAGETYP'):
        combine([image(np.ones((2,2)), IMAGETYP='OBJECT')]*2, kind='bias')


def test_mask_preserves_pixels_and_fits_roundtrip_and_crop():
    original = image([[0, 10, 20], [30, 40, 50], [60, 70, 80]])
    result = mask_image(original, lower=5, upper=75, rectangle=(1, 2, 1, 2))
    assert result.mask.sum() == 3
    np.testing.assert_array_equal(result.data, original.data)
    assert not original.mask.any()
    payload = to_fits(result)
    reread = read_frame(payload)
    np.testing.assert_array_equal(reread.mask, result.mask)
    np.testing.assert_array_equal(read_mask(payload), result.mask)
    cropped = export_crop(result.data, result.header, 1, 3, 1, 3, mask=result.mask)
    with fits.open(BytesIO(cropped), checksum=True) as hdus:
        np.testing.assert_array_equal(hdus['MASK'].data, [[1, 0], [0, 1]])
        assert hdus[0].verify_checksum() == 1


def test_scaled_unsigned_fits_and_invalid_normalization():
    buffer = BytesIO()
    fits.PrimaryHDU(np.array([[0,32768],[50000,65535]],dtype='uint16')).writeto(buffer)
    result = read_frame(buffer.getvalue())
    with fits.open(BytesIO(to_fits(result))) as hdus:
        np.testing.assert_array_equal(hdus[0].data, [[0,32768],[50000,65535]])
    with pytest.raises(ValueError, match='median is not positive'):
        combine([image(np.zeros((2,2)))]*2, kind='flat')


def test_translation_alignment_integer_and_subpixel():
    y, x = np.indices((96, 128))
    data = sum(amplitude * np.exp(-((y-cy)**2 + (x-cx)**2) / (2*width**2))
               for cy, cx, amplitude, width in [(20, 30, 100, 1.5), (65, 80, 70, 2.5),
                                                 (42, 105, 50, 1.2)])
    reference = image(data, FILTER='Blue')
    integer = image(shift(data, (4, -7), order=0, mode='constant', cval=np.nan), FILTER='Blue')
    subpixel = image(shift(data, (-3.4, 5.2), order=1, mode='constant', cval=np.nan), FILTER='Blue')
    np.testing.assert_allclose(estimate_translation(reference, integer, 20, False), (-4, 7), atol=.01)
    np.testing.assert_allclose(estimate_translation(reference, subpixel, 20, True), (3.4, -5.2), atol=.15)
    aligned = combine([reference, integer, subpixel], sigma=None, align=True, max_shift=20)
    unaligned = combine([reference, integer, subpixel], sigma=None)
    core = np.s_[10:-10, 10:-10]
    assert np.mean((aligned.data[core] - data[core])**2) < np.mean((unaligned.data[core] - data[core])**2) / 20
    assert aligned.header['ALIGN'] == 'TRANSLATION'
    np.testing.assert_allclose([aligned.header['ALN001Y'], aligned.header['ALN001X']], [-4, 7], atol=.15)
    shifted = translate_frame(reference, (2.5, -3.5))
    assert shifted.mask[:, -4:].any() and shifted.mask[:3, :].any()


def test_alignment_rejects_bad_limit_and_non_science_combination():
    data = np.zeros((32, 32)); data[10, 10] = 1
    a = image(data, IMAGETYP='BIAS')
    b = image(shift(data, (8, 0)), IMAGETYP='BIAS')
    with pytest.raises(ValueError, match='reached the'):
        estimate_translation(a, b, max_shift=5)
    with pytest.raises(ValueError, match='only'):
        combine([a, b], kind='bias', align=True)
