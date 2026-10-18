from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from streamlit.testing.v1 import AppTest

from workflow_ui import automatic_frame_selection


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    for index, value in enumerate((8., 12.)):
        fits.writeto(tmp_path / f'bias_{index}.fits', np.full((6, 8), value),
                     fits.Header({'IMAGETYP': 'BIAS'}))
    for color, value in [('Blue', 110.), ('Red', 210.)]:
        for index in range(2):
            fits.writeto(tmp_path / f'flat_{color}_{index}.fits', np.full((6, 8), value),
                         fits.Header({'IMAGETYP': 'FLAT', 'FILTER': color, 'EXPTIME': 1.}))
        for target in ('deneb', 'vega'):
            name = f'{target}_{color}_0001.fits'
            fits.writeto(tmp_path / name, np.full((6, 8), 60.),
                         fits.Header({'IMAGETYP': 'OBJECT', 'FILTER': color,
                                      'OBJECT': target, 'EXPTIME': 30.}))
    fits.writeto(tmp_path / 'dark.fits', np.ones((6, 8)), fits.Header({'IMAGETYP': 'DARK'}))
    return tmp_path


def open_app(folder):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(folder)
    return app.run()


def test_auto_runs_in_dependency_order_and_corrects_every_target_and_filter(dataset, monkeypatch):
    from processing_jobs import run_with_timeout
    from imageProcessing import read_frame
    calls = []

    def record(function, *args, **kwargs):
        calls.append(function.__name__)
        if function.__name__ == 'combine_flats_job':
            assert kwargs['timeout'] == 180.
        return run_with_timeout(function, *args, **kwargs)

    monkeypatch.setattr('workflow_ui.run_with_timeout', record)
    monkeypatch.setattr('science_ui.run_with_timeout', record)
    originals = {path.name: path.read_bytes() for path in dataset.glob('*.fits')}
    app = open_app(dataset)
    assert not app.exception
    assert app.sidebar.button[0].key == 'workflow_auto'
    assert not app.session_state['workflow_science_frames']
    app.button(key='workflow_auto').click().run()
    assert not app.exception and not app.error
    assert calls == ['combine_bias_frames', 'combine_flats_job', 'correct_science_frames']
    assert app.session_state['workflow_flat_subtract_bias'] is True
    bias = app.session_state['workflow_bias_last'][1]
    np.testing.assert_allclose(bias.data, 10.)
    flats = app.session_state['workflow_flat_last'][1]
    assert set(flats) == {'Blue', 'Red'}
    corrected = app.session_state['workflow_science_corrected'][1]
    assert len(corrected) == 4
    for name, path in corrected.items():
        image = read_frame(path)
        expected = 50.
        np.testing.assert_allclose(image.data, expected)
        assert image.header['BIASFILE'] == bias.name
        assert image.header['FLATFILE'] == flats[image.header['FILTER']].name
        assert image.header['SCICOR'] is True
    assert any('Auto complete: 4' in item.value for item in app.success)
    assert any(item.value.startswith('corrected_science_') for item in app.caption)
    assert {path.name: path.read_bytes() for path in dataset.glob('*.fits')} == originals
    # Viewing the products reuses the completed run.
    app.button(key='workflow_image_name_science_corrected_1').click().run()
    assert not app.exception and not app.error
    assert len(calls) == 3


def test_auto_replaces_previous_object_and_frame_selection(dataset):
    app = open_app(dataset)
    app.text_input(key='workflow_science_pattern').set_value('deneb.*').run()
    app.session_state['workflow_science_frames'] = []
    app.text_input(key='workflow_bias_pattern').set_value('[').run()
    app.button(key='workflow_auto').click().run()
    assert not app.exception and not app.error
    assert not any(item.label == 'All file numbers' for item in app.checkbox)
    assert len(app.session_state['workflow_science_corrected'][1]) == 4
    app.button(key='workflow_auto').click().run()
    assert not app.exception and not app.error
    assert len(app.session_state['workflow_science_corrected'][1]) == 4


@pytest.mark.parametrize('failure', ['combine_bias_frames', 'combine_flats_job', 'correct_science_frames'])
def test_auto_stops_on_failed_step_and_can_retry(dataset, monkeypatch, failure):
    from processing_jobs import run_with_timeout
    calls = []

    def fail(function, *args, **kwargs):
        calls.append(function.__name__)
        if function.__name__ == failure:
            raise TimeoutError('Timeout: operation stopped after 30 seconds.')
        return run_with_timeout(function, *args, **kwargs)

    app = open_app(dataset)
    with monkeypatch.context() as patch:
        patch.setattr('workflow_ui.run_with_timeout', fail)
        patch.setattr('science_ui.run_with_timeout', fail)
        app.button(key='workflow_auto').click().run()
    assert not app.exception
    assert any('Timeout:' in item.value for item in app.error)
    order = ['combine_bias_frames', 'combine_flats_job', 'correct_science_frames']
    assert calls == order[:order.index(failure) + 1]
    assert 'workflow_science_corrected' not in app.session_state
    assert not app.success
    app.run()
    assert len(calls) == order.index(failure) + 1
    app.button(key='workflow_auto').click().run()
    assert not app.exception and not app.error
    assert len(app.session_state['workflow_science_corrected'][1]) == 4


def test_auto_classifies_headers_and_excludes_generated_images(dataset):
    (dataset / 'bias_0.fits').rename(dataset / 'offset.fits')
    (dataset / 'flat_Blue_0.fits').rename(dataset / 'lamp.fits')
    for name, metadata in [
            ('master_science.fits', {}), ('corrected_science.fits', {}),
            ('export.fits', {'SCICOR': True}), ('other.fits', {'IMAGETYP': 'MASTER_SCIENCE'})]:
        fits.writeto(dataset / name, np.ones((6, 8)), fits.Header(metadata))
    groups = automatic_frame_selection(sorted(dataset.glob('*.fits')))
    assert set(groups['bias']) == {'offset.fits', 'bias_1.fits'}
    assert len(groups['flat']) == 4 and 'lamp.fits' in groups['flat']
    assert len(groups['science']) == 4
    app = open_app(dataset)
    app.button(key='workflow_auto').click().run()
    assert not app.exception and not app.error
    assert len(app.session_state['workflow_science_corrected'][1]) == 4


def test_auto_reports_missing_calibrations(dataset):
    for path in dataset.glob('bias*'):
        path.unlink()
    app = open_app(dataset)
    app.button(key='workflow_auto').click().run()
    assert not app.exception
    assert any('at least two original bias' in item.value for item in app.error)
    assert 'workflow_science_corrected' not in app.session_state
