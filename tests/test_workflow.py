from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import rotate, shift
from streamlit.testing.v1 import AppTest

from imageProcessing import frame, to_fits
from imageProcessing import (align_calibration_frames, align_frame,
                      combine_bias_frames, combine_flat_frames,
                      combine_flat_groups, flat_exposure_groups, run_workflow,
                      source_filter)


def image(values, name='image.fits', **header):
    return frame(values, fits.Header(header), name=name)


def test_workflow_calibrates_flats_before_science_then_subtracts_background():
    y, x = np.indices((40, 48))
    response = 0.7 + x / 80 + y / 120
    signal = 80 * np.exp(-((x-22)**2 + (y-17)**2) / 8)
    raw = image(100 + response * (25 + signal), FILTER='Blue')
    biases = [image(np.full(raw.data.shape, 100), name=f'bias{i}.fits', IMAGETYP='BIAS')
              for i in range(2)]
    flats = [image(100 + level * response, name=f'flat{i}.fits',
                   IMAGETYP='FLAT', FILTER='Blue') for i, level in enumerate((1000, 2000))]
    result = run_workflow(raw, bias_sources=biases, flat_sources=flats, background=True)
    assert result.header['BIASCOR'] and result.header['FLATCOR'] and result.header['BKGDSUB']
    np.testing.assert_allclose(result.data[17, 22], signal[17, 22] * np.median(response), atol=0.5)
    assert abs(np.median(result.data)) < 0.1
    np.testing.assert_array_equal(raw.data, 100 + response * (25 + signal))


def test_workflow_aligns_and_combines_science_frames():
    y, x = np.indices((96, 128))
    stars = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / 5)
                for cy, cx, peak in [(24, 30, 100), (66, 82, 60), (45, 104, 80)])
    first = image(stars, FILTER='Blue')
    second = image(shift(stars, (4, -6), order=1, mode='constant', cval=np.nan),
                   name='second.fits', FILTER='Blue')
    result = run_workflow(first, [second], alignment='Translation', max_shift=20,
                          rejection_sigma=None)
    assert result.header['NCOMBINE'] == 2
    assert result.name.endswith('_workflow_x2.fits')
    assert result.header['ALIGN'] == 'TRANSLATION'
    np.testing.assert_allclose([result.header['ALN001Y'], result.header['ALN001X']],
                               [-4, 6], atol=0.2)
    assert np.mean((result.data[12:-12, 12:-12] - stars[12:-12, 12:-12])**2) < 0.1


def test_workflow_does_not_combine_science_exposures():
    first = image(np.ones((12, 12)), EXPTIME=45., FILTER='Blue')
    second = image(np.ones((12, 12)), EXPTIME=90., FILTER='Blue')
    with pytest.raises(ValueError, match='exposure 90.0 s differs'):
        run_workflow(first, [second])


def test_rotation_alignment_handles_small_camera_rotation():
    y, x = np.indices((128, 128))
    stars = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*width**2))
                for cy, cx, peak, width in [(21, 30, 100, 1.5), (96, 45, 80, 2),
                                            (67, 103, 70, 1.3), (31, 91, 50, 1.7)])
    first = image(stars, FILTER='Blue')
    turned = rotate(stars, 2, reshape=False, order=1, mode='constant', cval=np.nan)
    second = image(shift(turned, (3, -4), order=1, mode='constant', cval=np.nan),
                   FILTER='Blue')
    aligned, (angle, dy, dx) = align_frame(first, second, max_shift=20, max_angle=5)
    assert abs(angle + 2) < 0.6
    assert abs(dy + 3) < 1 and abs(dx - 4) < 1
    valid = ~aligned.mask & ~first.mask
    assert np.mean((aligned.data[valid] - first.data[valid])**2) < 2


def test_calibration_alignment_uses_filter_groups_and_preserves_noise():
    y, x = np.indices((64, 64))
    pattern = (35 * np.exp(-((x-19)**2 + (y-27)**2) / 18)
               + 18 * np.exp(-((x-43)**2 + (y-39)**2) / 32))
    blue = [image(100 + pattern, name='flat_Blue_1.fits', IMAGETYP='FLAT', FILTER='Blue'),
            image(100 + shift(pattern, (3, -4), mode='constant', cval=0),
                  name='flat_Blue_2.fits', IMAGETYP='FLAT', FILTER='Blue')]
    red = [image(100 + pattern, name='flat_Red_1.fits', IMAGETYP='FLAT', FILTER='Red'),
           image(100 + shift(pattern, (-2, 5), mode='constant', cval=0),
                 name='flat_Red_2.fits', IMAGETYP='FLAT', FILTER='Red')]
    aligned, offsets = align_calibration_frames([*blue, *red], 'flat', 10)
    np.testing.assert_allclose(offsets[1], (-3, 4), atol=.25)
    np.testing.assert_allclose(offsets[3], (2, -5), atol=.25)
    assert aligned[0].header['ALIGN'] == 'TRANSLATION'
    assert aligned[2].header['ALN001Y'] == pytest.approx(2, abs=.25)
    for first, second in [(aligned[0], aligned[1]), (aligned[2], aligned[3])]:
        valid = ~first.mask & ~second.mask
        assert np.mean((first.data[valid] - second.data[valid])**2) < 1
    assert blue[1].header.get('ALIGN') is None

    rng = np.random.default_rng(27)
    biases = [image(rng.normal(size=(64, 64)), name=f'bias{i}.fits', IMAGETYP='BIAS')
              for i in range(2)]
    aligned_biases, bias_offsets = align_calibration_frames(biases, 'bias', 15)
    assert bias_offsets == [(0., 0.), (0., 0.)]
    np.testing.assert_array_equal(aligned_biases[1].data, biases[1].data)
    assert aligned_biases[0].header['ALIGN'] == 'NO SHIFT'
    assert aligned_biases[0].header['ALNSKIP'] == 1
    assert aligned_biases[1].header['ALNSTAT'] == 'UNCHANGED'


def test_bias_combines_without_alignment_controls(tmp_path):
    for name, level, kind in [('science.fits', 120, 'OBJECT'),
                              ('bias1.fits', 10, 'BIAS'),
                              ('bias2.fits', 12, 'BIAS')]:
        (tmp_path / name).write_bytes(to_fits(image(np.full((12, 12), level),
                                               name=name, IMAGETYP=kind)))
    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
workflow_panel(read_frame(folder / 'science.fits'), 'source', folder)
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    assert not any(button.key == 'workflow_bias_align' for button in app.button)
    app.button(key='workflow_bias_combine').click().run()
    assert not app.exception and not app.error
    np.testing.assert_allclose(app.session_state['workflow_bias_last'][1].data, 11)


def test_workflow_combines_original_bias_and_flat_frames(tmp_path):
    y, x = np.indices((64, 64))
    pattern = (25 * np.exp(-((x-18)**2 + (y-26)**2) / 18)
               + 15 * np.exp(-((x-44)**2 + (y-39)**2) / 28))
    files = [('science.fits', np.full((64, 64), 200.), 'OBJECT'),
             ('bias1.fits', 100 + pattern, 'BIAS'),
             ('bias2.fits', 100 + shift(pattern, (2, -3), mode='constant'), 'BIAS'),
             ('flat_Blue_1.fits', 1100 + pattern, 'FLAT'),
             ('flat_Blue_2.fits', 1100 + shift(pattern, (-3, 4), mode='constant'), 'FLAT')]
    for name, values, kind in files:
        metadata = {'IMAGETYP': kind}
        if kind != 'BIAS':
            metadata['FILTER'] = 'Blue'
        (tmp_path / name).write_bytes(to_fits(image(values, name=name, **metadata)))
    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
workflow_panel(read_frame(folder / 'science.fits'), 'source', folder)
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    app.text_input(key='workflow_bias_pattern').set_value('bias*').run()
    assert not any(button.key in ('workflow_bias_align', 'workflow_flat_align') for button in app.button)
    assert not any(field.key in ('workflow_bias_shift', 'workflow_flat_shift') for field in app.number_input)
    app.button(key='workflow_bias_combine').click().run()
    assert not app.exception and not app.error
    master = app.session_state['workflow_bias_last'][1]
    assert 'ALIGN' not in master.header
    np.testing.assert_allclose(master.data, (files[1][1] + files[2][1]) / 2)
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    master = app.session_state['workflow_flat_last'][1]['Blue']
    assert 'ALIGN' not in master.header
    calibrated = [(values - app.session_state['workflow_bias_last'][1].data) for values in (files[3][1], files[4][1])]
    expected = np.mean([values / np.median(values) for values in calibrated], axis=0)
    np.testing.assert_allclose(master.data, expected / np.median(expected))


def test_bias_combination_stage_preview_uses_selected_method(tmp_path):
    for name, value, kind in [('science.fits', 120, 'OBJECT'),
                              ('bias1.fits', 10, 'BIAS'),
                              ('bias2.fits', 10, 'BIAS'),
                              ('bias3.fits', 40, 'BIAS')]:
        (tmp_path / name).write_bytes(to_fits(image(np.full((12, 12), value),
                                               name=name, IMAGETYP=kind)))
    sources = [tmp_path / f'bias{i}.fits' for i in range(1, 4)]
    np.testing.assert_allclose(combine_bias_frames(sources, 'mean').data, 10)
    np.testing.assert_allclose(combine_bias_frames(sources, 'median').data, 10)
    np.testing.assert_allclose(combine_bias_frames(sources).data, 10)
    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
result = workflow_panel(read_frame(folder / 'science.fits'), 'source', folder)
st.session_state.preview = result.name if result is not None else None
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    assert not any(widget.key == 'workflow_bias_mode' for widget in app.selectbox)
    assert app.radio(key='workflow_bias_method').value == 'Median'
    assert app.text_input(key='workflow_bias_pattern').value == 'bias*'
    assert app.session_state['workflow_bias_frames'] == [path.name for path in sources]
    app.text_input(key='workflow_bias_pattern').set_value(r'^bias1\.fits$').run()
    assert app.session_state['workflow_bias_frames'] == ['bias1.fits']
    app.text_input(key='workflow_bias_pattern').set_value('[').run()
    assert app.error and not app.exception
    assert app.session_state['workflow_bias_frames'] == ['bias1.fits']
    app.text_input(key='workflow_bias_pattern').set_value('BIAS*').run()
    assert app.session_state['workflow_bias_frames'] == [path.name for path in sources]
    app.button(key='workflow_bias_combine').click().run()
    assert not app.exception and not app.error
    assert app.session_state['preview'].startswith('master_bias_')
    assert app.session_state['preview'].endswith('_x3.fits')
    assert any(app.session_state['preview'] in item.value for item in app.markdown)
    np.testing.assert_allclose(app.session_state['workflow_bias_last'][1].data, 10)
    app.radio(key='workflow_bias_method').set_value('Mean').run()
    assert app.session_state['preview'] is None
    app.button(key='workflow_bias_combine').click().run()
    np.testing.assert_allclose(app.session_state['workflow_bias_last'][1].data, 10)
    app.button(key='workflow_bias_view').click().run()
    assert app.session_state['preview'].startswith('master_bias_')
    assert any(app.session_state['preview'] in item.value for item in app.markdown)
    app.button(key='workflow_bias_master_remove').click().run()
    assert 'workflow_bias_last' not in app.session_state
    assert app.session_state['preview'] is None


def test_flat_pattern_combination_preview_and_method(tmp_path):
    left, right = np.full((12, 6), 100.), np.full((12, 6), 200.)
    regular = np.hstack((left, right))
    outlier = np.hstack((np.full((12, 6), 400.), left))
    files = [('science.fits', np.full((12, 12), 120.), 'OBJECT', 'Blue'),
             ('bias1.fits', np.full((12, 12), 10.), 'BIAS', None),
             ('bias2.fits', np.full((12, 12), 10.), 'BIAS', None),
             ('flat1.fits', regular + 10, 'FLAT', 'Blue'),
             ('flat2.fits', regular + 10, 'FLAT', 'Blue'),
             ('flat3.fits', outlier + 10, 'FLAT', 'Blue'),
             ('flat_green1.fits', outlier + 10, 'FLAT', 'Green'),
             ('flat_green2.fits', outlier + 10, 'FLAT', 'Green'),
             ('flat_001_Red_01.fits', regular + 10, 'FLAT', None),
             ('flat_002_Red_02.fits', regular + 10, 'FLAT', None),
             ('flat_001_black_01.fits', regular + 10, 'FLAT', None),
             ('flat_002_black_02.fits', regular + 10, 'FLAT', None)]
    for name, values, kind, filter_name in files:
        metadata = {'IMAGETYP': kind}
        if filter_name:
            metadata['FILTER'] = filter_name
        (tmp_path / name).write_bytes(to_fits(image(values, name=name, **metadata)))
    flats = [tmp_path / f'flat{i}.fits' for i in range(1, 4)]
    all_flats = sorted(tmp_path.glob('flat*.fits'))
    bias = combine_bias_frames([tmp_path / 'bias1.fits', tmp_path / 'bias2.fits'])
    mean = combine_flat_frames(flats, bias, 'mean')
    median = combine_flat_frames(flats, bias, 'median')
    np.testing.assert_allclose(mean.data, median.data)  # clipping rejects the transient outlier
    np.testing.assert_allclose(combine_flat_frames(all_flats, bias, 'mean',
                                                  tmp_path / 'science.fits').data, mean.data)
    with pytest.raises(ValueError, match='match science FILTER'):
        combine_flat_frames(all_flats, bias, reference=image(np.ones((12, 12)), FILTER='Red'))
    with pytest.raises(ValueError, match='science image has no FILTER'):
        combine_flat_frames(all_flats, bias, reference=image(np.ones((12, 12))))
    assert source_filter(tmp_path / 'flat_001_Red_01.fits') == 'Red'
    assert source_filter(tmp_path / 'flat_001_black_01.fits') == 'black'
    with pytest.raises(ValueError, match='filename filter Green differs'):
        source_filter(image(np.ones((12, 12)), name='flat_Green_01.fits', FILTER='Blue'))
    grouped = combine_flat_groups(all_flats, bias)
    assert list(grouped) == ['Red', 'Green', 'Blue', 'black']
    assert [grouped[name].header['NCOMBINE'] for name in grouped] == [2, 2, 3, 2]
    np.testing.assert_allclose(grouped['Blue'].data, regular / np.median(regular))
    green_science = image(np.full((12, 12), 120.), name='science_Green.fits', FILTER='Green')
    green_result = run_workflow(green_science, bias_sources=[tmp_path / 'bias1.fits',
                                                              tmp_path / 'bias2.fits'],
                                flat_sources=all_flats)
    np.testing.assert_allclose(green_result.data,
                               110 * np.median(grouped['Green'].data) / grouped['Green'].data)

    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
result = workflow_panel(read_frame(folder / 'science.fits'), 'source', folder)
st.session_state.preview = result.name if result is not None else None
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    assert not any(widget.key == 'workflow_flat_mode' for widget in app.selectbox)
    assert app.radio(key='workflow_flat_method').value == 'Median'
    assert app.text_input(key='workflow_flat_pattern').value == 'flat*'
    assert app.session_state['workflow_flat_frames'] == sorted(
        [path.name for path in all_flats], key=str.casefold)
    app.text_input(key='workflow_bias_pattern').set_value('bias*').run()
    app.text_input(key='workflow_flat_pattern').set_value(r'^flat1\.fits$').run()
    assert app.session_state['workflow_flat_frames'] == ['flat1.fits']
    app.text_input(key='workflow_flat_pattern').set_value(r'^flat[12]\.fits$').run()
    assert app.session_state['workflow_flat_frames'] == ['flat1.fits', 'flat2.fits']
    app.text_input(key='workflow_flat_pattern').set_value('[').run()
    assert app.error and not app.exception
    assert app.session_state['workflow_flat_frames'] == ['flat1.fits', 'flat2.fits']
    app.text_input(key='workflow_flat_pattern').set_value('FLAT*').run()
    assert app.session_state['workflow_flat_frames'] == sorted(
        [path.name for path in all_flats], key=str.casefold)
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    assert app.session_state['preview'] == 'corrected_flat_blue_x3.fits'
    masters = app.session_state['workflow_flat_last'][1]
    assert list(masters) == ['Red', 'Green', 'Blue', 'black']
    inputs = app.session_state['workflow_flat_last'][3]
    assert inputs['Red'] == ['flat_001_Red_01.fits', 'flat_002_Red_02.fits']
    assert inputs['Blue'] == ['flat1.fits', 'flat2.fits', 'flat3.fits']
    for filter_name in masters:
        assert app.button(key=f'workflow_flat_view_{filter_name}')
    np.testing.assert_allclose(masters['Blue'].data, regular / np.median(regular))
    with ZipFile(BytesIO(app.session_state['workflow_flat_last'][2])) as bundle:
        assert set(bundle.namelist()) == {master.name for master in masters.values()}
    app.button(key='workflow_flat_view_black').click().run()
    assert app.session_state['preview'] == 'corrected_flat_black_x2.fits'
    app.radio(key='workflow_flat_method').set_value('Mean').run()
    assert app.session_state['preview'] is None
    app.button(key='workflow_flat_combine').click().run()
    combined_mean = regular / np.median(regular)
    np.testing.assert_allclose(app.session_state['workflow_flat_last'][1]['Blue'].data, combined_mean)
    app.button(key='workflow_flat_view_Blue').click().run()
    assert app.session_state['preview'] == 'corrected_flat_blue_x3.fits'
    app.radio(key='workflow_bias_method').set_value('Mean').run()
    assert app.session_state['preview'] is None

    direct = AppTest.from_string(source, default_timeout=30).run()
    direct.text_input(key='workflow_bias_pattern').set_value('bias*').run()
    direct.text_input(key='workflow_flat_pattern').set_value('flat*').run()
    direct.button(key='workflow_flat_combine').click().run()
    assert not direct.exception and not direct.error
    assert set(direct.session_state['workflow_flat_last'][1]) == {'Red', 'Green', 'Blue', 'black'}
    assert len(direct.session_state['workflow_flat_last'][3]['Green']) == 2
    direct.button(key='workflow_flat_view_Green').click().run()
    direct.button(key='workflow_flat_master_remove_Green').click().run()
    assert not direct.exception and not direct.error
    assert 'Green' not in direct.session_state['workflow_flat_last'][1]
    assert direct.session_state['preview'] == 'corrected_flat_red_x2.fits'
    with ZipFile(BytesIO(direct.session_state['workflow_flat_last'][2])) as bundle:
        assert 'corrected_flat_green_x2.fits' not in bundle.namelist()


def test_flat_exposures_and_file_buttons(tmp_path):
    entries = [('science_Blue.fits', 120., 'OBJECT', 'Blue', 8.),
               ('other.fits', 125., 'OBJECT', 'Blue', 8.),
               ('bias1.fits', 10., 'BIAS', None, 0.),
               ('bias2.fits', 10., 'BIAS', None, 0.),
               ('flat_Blue_8_a.fits', 110., 'FLAT', 'Blue', 8.),
               ('flat_Blue_8_b.fits', 120., 'FLAT', 'Blue', 8.),
               ('flat_Blue_10_a.fits', 130., 'FLAT', 'Blue', 10.),
               ('flat_Red_35_a.fits', 140., 'FLAT', 'Red', 35.),
               ('flat_Red_40_a.fits', 150., 'FLAT', 'Red', 40.)]
    for name, value, kind, color, exposure in entries:
        metadata = {'IMAGETYP': kind, 'EXPTIME': exposure}
        if color:
            metadata['FILTER'] = color
        (tmp_path / name).write_bytes(to_fits(image(np.full((12, 12), value),
                                               name=name, **metadata)))
    flats = sorted(tmp_path.glob('flat*.fits'))
    groups = flat_exposure_groups(flats)
    assert {color: {exp: len(items) for exp, items in values.items()}
            for color, values in groups.items()} == {
                'Red': {35.: 1, 40.: 1}, 'Blue': {8.: 2, 10.: 1}}
    masters = combine_flat_groups(flats)
    assert list(masters) == ['Red', 'Blue']
    assert masters['Blue'].header['NCOMBINE'] == 3
    assert masters['Blue'].header['EXPNORM']

    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
result = workflow_panel(read_frame(folder / 'science_Blue.fits'), 'source', folder)
st.session_state.preview = result.name if result is not None else None
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    app.text_input(key='workflow_science_pattern').set_value('other.*').run()
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    assert set(app.session_state['workflow_flat_last'][1]) == {'Red', 'Blue'}
    assert app.session_state['workflow_flat_last'][3]['Blue'] == [
        'flat_Blue_10_a.fits', 'flat_Blue_8_a.fits', 'flat_Blue_8_b.fits']
    app.button(key='workflow_flat_master_remove_Blue').click().run()
    assert set(app.session_state['workflow_flat_last'][1]) == {'Red'}
    app.button(key='workflow_flat_master_remove_Red').click().run()
    assert 'workflow_flat_last' not in app.session_state
    assert not any(widget.key == 'workflow_flat_frames' for widget in app.multiselect)
    assert not any('View selected flat' in item.label for item in app.expander)


def test_flat_bias_toggle_selects_precomputed_versions_without_processing(tmp_path, monkeypatch):
    raw = np.tile([[20., 30.], [40., 50.]], (6, 6))
    for name, values, kind in [('science.fits', raw, 'OBJECT'),
                               ('flat_Blue_1.fits', raw, 'FLAT'),
                               ('flat_Blue_2.fits', raw, 'FLAT'),
                               ('bias1.fits', np.full_like(raw, 10.), 'BIAS'),
                               ('bias2.fits', np.full_like(raw, 10.), 'BIAS')]:
        metadata = {'IMAGETYP': kind, 'EXPTIME': 0. if kind == 'BIAS' else 1.}
        if kind != 'BIAS':
            metadata['FILTER'] = 'Blue'
        (tmp_path / name).write_bytes(to_fits(image(values, name=name, **metadata)))
    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
result = workflow_panel(read_frame(folder / 'science.fits'), 'source', folder)
st.session_state.preview = result.name if result is not None else None
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    assert app.session_state['workflow_flat_subtract_bias']
    app.text_input(key='workflow_bias_pattern').set_value('').run()
    app.button(key='workflow_flat_combine').click().run()
    assert app.error and not app.exception
    app.text_input(key='workflow_bias_pattern').set_value('bias*').run()
    app.button(key='workflow_bias_combine').click().run()
    saved_bias = app.session_state['workflow_bias_last'][1]
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    assert app.session_state['workflow_bias_last'][1] is saved_bias
    versions = app.session_state['workflow_flat_versions'][1]
    np.testing.assert_allclose(versions[False]['Blue'].data, raw)
    corrected = versions[True]['Blue']
    np.testing.assert_allclose(corrected.data, (raw - saved_bias.data) / np.median(raw - saved_bias.data))
    assert corrected.name == 'corrected_flat_blue_1s_x2.fits'
    assert corrected.header['BIASCOR'] is True and corrected.header['NORMALIZ'] is True

    def unexpected_processing(*args, **kwargs):
        raise AssertionError('Switching the displayed version must not run a processing job.')

    with monkeypatch.context() as patch:
        patch.setattr('workflow_ui.run_with_timeout', unexpected_processing)
        for mode in (False, True, False):
            app.button(key='workflow_flat_bias_option').click().run()
            assert not app.exception and not app.error
            assert app.session_state['workflow_flat_subtract_bias'] is mode
            shown = app.session_state['workflow_flat_last'][1]['Blue']
            assert shown is versions[mode]['Blue']
            assert app.session_state['preview'] == shown.name
            with ZipFile(BytesIO(app.session_state['workflow_flat_last'][2])) as archive:
                assert archive.namelist() == [shown.name]
                with fits.open(BytesIO(archive.read(shown.name))) as hdus:
                    np.testing.assert_allclose(hdus[0].data, shown.data)
    np.testing.assert_allclose(app.session_state['workflow_flat_last'][1]['Blue'].data, raw)
    app.button(key='workflow_flat_bias_option').click().run()
    np.testing.assert_allclose(app.session_state['workflow_flat_last'][1]['Blue'].data, (raw - 10.) / np.median(raw - 10.))
    app.button(key='workflow_flat_master_remove_Blue').click().run()
    assert 'workflow_flat_versions' not in app.session_state
    app.button(key='workflow_flat_bias_option').click().run()
    assert 'workflow_flat_last' not in app.session_state


@pytest.mark.parametrize('method', ['median', 'mean'])
def test_corrected_flat_calibrates_and_normalizes_each_input(method):
    bias_values = np.array([[30., 3.], [5., 7.]])
    bias = image(bias_values, name='master_bias_x2.fits', IMAGETYP='MASTER_BIAS', BUNIT='adu')
    bias.mask[1, 1] = True
    response = np.array([[.5, 1.], [1.5, 2.]])
    inputs = [image(bias_values + response * level, name=f'flat_Blue_{index}.fits',
                    IMAGETYP='FLAT', BUNIT='adu', EXPTIME=2.)
              for index, level in enumerate([100., 200., 400.])]
    corrected = combine_flat_groups(inputs, bias, method=method)['Blue']
    np.testing.assert_allclose(corrected.data[~corrected.mask], response[~bias.mask])
    assert corrected.mask[1, 1]
    assert corrected.header['BIASCOR'] and corrected.header['NORMALIZ']
    assert corrected.header['BUNIT'] == '1'
    assert corrected.header['NCOMBINE'] == 3
    np.testing.assert_array_equal(bias.data, bias_values)
    assert inputs[0].header.get('BIASCOR') is None
