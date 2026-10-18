from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def isolate_saved_folder(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'working_folder.json')


def test_viewer_tools_and_direct_export(tmp_path):
    fits.writeto(tmp_path / 'image.fits', np.arange(48.).reshape(6, 8))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    assert not app.exception
    assert not any(widget.key == 'workflow_bias_mode' for widget in app.selectbox)
    assert app.radio(key='workflow_bias_method').value == 'Median'
    assert not any(widget.label == 'Find FITS file to view' for widget in app.text_input)
    assert not any('View selected bias' in widget.label for widget in app.expander)
    labels = [tab.label for tab in app.tabs]
    assert labels == ['Statistics', 'Header']
    assert any(item.value == 'File' for item in app.subheader)
    assert app.session_state['workflow_flat_subtract_bias'] is True
    assert app.text_input(key='workflow_science_pattern').value == 'hd226868.*'
    assert not any(item.label == 'All file numbers' for item in app.checkbox)
    assert 'Workflow' not in labels and 'Processing' not in labels
    assert not app.sidebar.tabs
    assert [item.label for item in app.sidebar.expander if item.label[:1].isdigit()] == [
        '1. Master bias', '2. Flats', '3. Science']
    assert all(button.key != 'workflow_run' for button in app.button)
    assert app.sidebar.expander[0].proto.expanded is False
    assert next(item for item in app.sidebar.expander if item.label == '2. Flats').proto.expanded is True
    assert not any(button.label in ('Download statistics CSV', 'Download header')
                   for button in app.get('download_button'))
    assert any(button.label == 'Download FITS' for button in app.get('download_button'))
    assert not any(button.label == 'Prepare FITS' for button in app.button)


def test_bias_grid_replaces_list_and_upload(tmp_path):
    for name in ('bias1.fits', 'bias2.fits'):
        fits.writeto(tmp_path / name, np.ones((12, 12)))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    assert not app.exception
    assert not any(item.label == 'Add bias frames' for item in app.get('file_uploader'))
    assert not any(item.key == 'workflow_bias_choose' for item in app.button)
    assert not any('**Bias frames**' in item.value for item in app.markdown)
    assert app.session_state['workflow_bias_frames'] == ['bias1.fits', 'bias2.fits']


def test_working_folder_updates_all_file_selections(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    for folder, count in [(first, 2), (second, 5)]:
        fits.writeto(folder / 'science.fits', np.ones((12, 12)))
        for index in range(2):
            fits.writeto(folder / f'bias{index}.fits', np.ones((12, 12)))
        for index in range(count):
            fits.writeto(folder / f'flat{index}.fits', np.ones((12, 12)))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.session_state['folder_path'] = str(first)
    app.run()
    assert not app.exception
    assert app.sidebar.text_input[0].key == 'folder_path'
    assert app.sidebar.text_input[0].label == 'Working folder'
    assert len(app.session_state['workflow_flat_frames']) == 2
    app.sidebar.text_input(key='folder_path').set_value(str(second)).run()
    assert not app.exception
    assert len(app.session_state['workflow_flat_frames']) == 5
    assert len(app.session_state['workflow_bias_frames']) == 2
    assert app.selectbox(key='viewer_file').value.parent == second
    app.text_input(key='workflow_science_pattern').set_value('science.*').run()
    assert app.session_state['workflow_science_frames'] == ['science.fits']


def test_working_folder_is_remembered_with_all_115_flats(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    old, correct = tmp_path / 'old', tmp_path / 'share-20261001'
    old.mkdir()
    correct.mkdir()
    fits.writeto(old / 'science.fits', np.ones((12, 12)))
    for index in range(115):
        fits.writeto(correct / f'flat_Blue_{index:04d}.fits', np.ones((12, 12)),
                     header=fits.Header({'IMAGETYP': 'FLAT', 'FILTER': 'Blue'}))
    app_file = str(Path(__file__).resolve().parents[1] / 'app.py')
    app = AppTest.from_file(app_file, default_timeout=10)
    app.session_state['folder_path'] = str(old)
    app.run()
    app.text_input(key='folder_path').set_value(str(correct)).run()
    assert not app.exception
    assert len(app.session_state['workflow_flat_frames']) == 115
    fresh = AppTest.from_file(app_file, default_timeout=10).run()
    assert not fresh.exception
    assert fresh.text_input(key='folder_path').value == str(correct)
    assert len(fresh.session_state['workflow_flat_frames']) == 115


def test_flat_bias_toggle_refreshes_main_image_from_cached_versions(tmp_path, monkeypatch):
    raw = np.tile([[20., 30.], [40., 50.]], (6, 6))
    for name, values, kind in [('flat_Blue_1.fits', raw, 'FLAT'),
                               ('flat_Blue_2.fits', raw, 'FLAT'),
                               ('bias1.fits', np.full_like(raw, 10.), 'BIAS'),
                               ('bias2.fits', np.full_like(raw, 10.), 'BIAS')]:
        header = fits.Header({'IMAGETYP': kind, 'EXPTIME': 1. if kind == 'FLAT' else 0.})
        if kind == 'FLAT':
            header['FILTER'] = 'Blue'
        fits.writeto(tmp_path / name, values, header)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    assert any(item.value == 'corrected_flat_blue_1s_x2.fits' for item in app.caption)

    def unexpected_processing(*args, **kwargs):
        raise AssertionError('The display toggle must use cached images.')

    monkeypatch.setattr('workflow_ui.run_with_timeout', unexpected_processing)
    for mode, name in [(False, 'master_flat_blue_1s_x2.fits'),
                       (True, 'corrected_flat_blue_1s_x2.fits')]:
        app.button(key='workflow_flat_bias_option').click().run()
        assert not app.exception and not app.error
        assert app.session_state['workflow_flat_subtract_bias'] is mode
        assert any(item.value == name for item in app.caption)
        np.testing.assert_allclose(app.session_state['workflow_flat_last'][1]['Blue'].data,
                                   (raw - 10.) / np.median(raw - 10.) if mode else raw)


def test_raw_viewer_uses_full_array_without_automatic_image_adjustments(tmp_path):
    from base64 import b64decode
    import json

    values = np.arange(12 * 1800, dtype=float).reshape(12, 1800) - 1000.25
    path = tmp_path / 'raw.fits'
    fits.writeto(path, values)
    original = path.read_bytes()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    assert not app.exception
    html = app.get('iframe')[0].proto.srcdoc
    assert 'width="1800" height="12"' in html
    encoded = json.loads(html.split('const raw = atob(', 1)[1].split(');', 1)[0])
    np.testing.assert_array_equal(np.frombuffer(b64decode(encoded), dtype='<f8').reshape(values.shape), values)
    assert 'const displayLow = 0.0, displayHigh = 65535.0;' in html
    assert 'percentiles' not in html and 'Stretch' not in html and 'Invert' not in html
    assert all(widget.key != 'workflow_reject' for widget in app.checkbox)
    assert path.read_bytes() == original


def test_fixed_display_levels_make_bias_visible_without_changing_data(tmp_path):
    from base64 import b64decode
    import json

    values = np.tile(np.arange(511., 624.), (12, 1))
    first, second = tmp_path / 'bias1.fits', tmp_path / 'bias2.fits'
    fits.writeto(first, values)
    fits.writeto(second, values + 2.)
    originals = [first.read_bytes(), second.read_bytes()]
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    assert not app.exception
    html = app.get('iframe')[0].proto.srcdoc
    assert 'const displayLow = 500.0, displayHigh = 630.0;' in html
    encoded = json.loads(html.split('const raw = atob(', 1)[1].split(');', 1)[0])
    np.testing.assert_array_equal(np.frombuffer(b64decode(encoded), dtype='<f8').reshape(values.shape), values)
    app.selectbox(key='viewer_file').select(second).run()
    assert not app.exception
    assert 'const displayLow = 500.0, displayHigh = 630.0;' in app.get('iframe')[0].proto.srcdoc
    assert [first.read_bytes(), second.read_bytes()] == originals


def test_contrast_sidebar_toggle_preserves_raw_calculation_and_flat_comparison(tmp_path, monkeypatch):
    from base64 import b64decode
    import json
    import re
    from io import BytesIO
    from zipfile import ZipFile

    bias = np.tile(np.arange(511., 624.), (12, 1))
    flat = np.tile(np.linspace(4000., 5000., 113), (12, 1))
    files = [('science.fits', np.full_like(flat, 65535.), 'OBJECT'),
             ('bias1.fits', bias, 'BIAS'), ('bias2.fits', bias, 'BIAS'),
             ('flat_Blue_1.fits', flat, 'FLAT'), ('flat_Blue_2.fits', flat, 'FLAT')]
    originals = {}
    for name, data, kind in files:
        header = fits.Header({'IMAGETYP': kind})
        if kind != 'BIAS':
            header['FILTER'] = 'Blue'
            header['EXPTIME'] = .2
        path = tmp_path / name
        fits.writeto(path, data, header)
        originals[name] = path.read_bytes()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    assert not app.session_state['viewer_contrast']
    assert all(widget.key not in ('viewer_display_min', 'viewer_display_max') for widget in app.number_input)
    assert all(widget.key != 'viewer_fit_display' for widget in app.button)
    assert all('Display limits:' not in item.value for item in app.markdown)
    for section in ('bias', 'flat'):
        buttons = list(app.sidebar.button)
        combine = next(item for item in buttons if item.key == f'workflow_{section}_combine')
        assert not any(item.key == f'workflow_{section}_contrast_option' for item in buttons)
        assert combine.label == 'Combine'
    app.selectbox(key='viewer_file').select(tmp_path / 'science.fits').run()
    assert any('This image is uniform:' in item.value for item in app.caption)
    app.selectbox(key='viewer_file').select(tmp_path / 'bias1.fits').run()
    assert any(item.value == 'bias1.fits' for item in app.caption)
    app.selectbox(key='viewer_file').select(tmp_path / 'flat_Blue_1.fits').run()
    assert any(item.value == 'flat_Blue_1.fits' for item in app.caption)
    app.button(key='workflow_science_contrast_option').click().run()
    assert app.session_state['viewer_contrast']
    app.button(key='workflow_bias_combine').click().run()
    assert not app.exception and not app.error
    saved_bias = app.session_state['workflow_bias_last'][1]
    np.testing.assert_array_equal(saved_bias.data, bias)
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    versions = app.session_state['workflow_flat_versions'][1]
    np.testing.assert_array_equal(versions[False]['Blue'].data, flat)
    normalized = (flat - bias) / np.median(flat - bias)
    np.testing.assert_allclose(versions[True]['Blue'].data, normalized)
    comparison = np.percentile(normalized, [1, 99])

    def bounds():
        html = app.get('iframe')[0].proto.srcdoc
        match = re.search(r'const displayLow = ([^,]+), displayHigh = ([^;]+);', html)
        return tuple(map(float, match.groups()))

    np.testing.assert_allclose(bounds(), comparison)
    old_contrast = np.percentile(flat, [1, 99])
    assert comparison[0] < old_contrast[0]

    def unexpected_processing(*args, **kwargs):
        raise AssertionError('Display toggles must not run image calculations.')

    monkeypatch.setattr('workflow_ui.run_with_timeout', unexpected_processing)
    for mode in (False, True):
        app.button(key='workflow_flat_bias_option').click().run()
        assert not app.exception and not app.error
        np.testing.assert_allclose(bounds(), np.percentile(normalized if mode else flat, [1, 99]))
        shown = versions[mode]['Blue']
        assert any(item.value == shown.name for item in app.caption)
        html = app.get('iframe')[0].proto.srcdoc
        encoded = json.loads(html.split('const raw = atob(', 1)[1].split(');', 1)[0])
        np.testing.assert_array_equal(np.frombuffer(b64decode(encoded), dtype='<f8').reshape(flat.shape), shown.data)
        with ZipFile(BytesIO(app.session_state['workflow_flat_last'][2])) as archive:
            with fits.open(BytesIO(archive.read(shown.name))) as hdus:
                np.testing.assert_array_equal(hdus[0].data, normalized if mode else flat)
    app.button(key='workflow_science_contrast_option').click().run()
    assert not app.session_state['viewer_contrast']
    assert bounds() == (0., 2.)
    app.button(key='workflow_image_name_bias_master').click().run()
    assert any(item.value.startswith('master_bias_') for item in app.caption)
    app.button(key='workflow_science_contrast_option').click().run()
    np.testing.assert_allclose(bounds(), np.percentile(bias, [1, 99]))
    app.button(key='workflow_image_name_flat_master_Blue').click().run()
    assert any(item.value == 'corrected_flat_blue_0.2s_x2.fits' for item in app.caption)
    np.testing.assert_allclose(bounds(), comparison)
    assert {name: (tmp_path / name).read_bytes() for name in originals} == originals
