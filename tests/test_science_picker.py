from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize('kind', ['bias', 'flat', 'science'])
def test_inline_regex_selection_and_reselection(tmp_path, monkeypatch, kind):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    names = [f'target_{i}.fits' for i in range(3)]
    for name in names + ['other.fits']:
        fits.writeto(tmp_path / name, np.ones((8, 8)))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.text_input(key=f'workflow_{kind}_pattern').set_value('target.*').run()
    calls, chosen = [], [names[1]]
    def picker(**kwargs):
        if kwargs['key'].startswith(kind + '_inline_'):
            calls.append(kwargs)
            return {'action': 'change', 'selected': chosen, 'event': repr(chosen)}
    monkeypatch.setattr('science_picker.image_picker', picker)
    app.run()
    assert not app.exception
    assert app.session_state[f'workflow_{kind}_frames'] == chosen
    assert 'workflow_file_view' not in app.session_state
    assert 'workflow_stage_view' not in app.session_state
    assert [item['name'] for item in calls[-1]['items']] == names
    assert not any(button.key == f'workflow_{kind}_choose' for button in app.button)
    chosen = names
    app.run()
    assert app.session_state[f'workflow_{kind}_frames'] == names


def test_calibration_choices_control_combined_inputs(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    for kind, levels in [('bias', [10., 10., 90.]), ('flat', [110., 110., 910.])]:
        for index, level in enumerate(levels):
            fits.writeto(tmp_path / f'{kind}_{index}.fits', np.full((12, 12), level),
                         fits.Header({'IMAGETYP': kind.upper(), 'FILTER': 'Blue', 'EXPTIME': 1.}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    for kind in ('bias', 'flat'):
        chosen = [f'{kind}_{index}.fits' for index in (0, 1)]
        monkeypatch.setattr('science_picker.image_picker',
                            lambda **kwargs: ({'action': 'change', 'selected': chosen, 'event': repr(chosen)}
                                              if kwargs['key'].startswith(kind + '_inline_') else None))
        app.run()
        app.radio(key=f'workflow_{kind}_method').set_value('Mean').run()
        app.button(key=f'workflow_{kind}_combine').click().run()
        assert not app.exception and not app.error
    bias = app.session_state['workflow_bias_last'][1]
    flat = app.session_state['workflow_flat_last'][1]['Blue']
    np.testing.assert_allclose(bias.data, 10.)
    np.testing.assert_allclose(flat.data, 1.)
    assert flat.header['NCOMBINE'] == 2
    # A changed selection invalidates the previous master.
    chosen = ['flat_0.fits', 'flat_2.fits']
    app.run()
    assert app.button(key='workflow_flat_show').disabled


def test_inline_bias_regex_selection_reselection_and_combination(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    names = ['bias_1.fits', 'bias_2.fits', 'bias_3.fits']
    for name, level in zip(names + ['other.fits'], [10., 30., 90., 500.]):
        fits.writeto(tmp_path / name, np.full((8, 8), level), fits.Header({'IMAGETYP': 'BIAS'}))
    calls, event = [], [None]
    def picker(**kwargs):
        assert kwargs['inline']
        if kwargs.get('preview_only'):
            return None
        calls.append(kwargs)
        return event[0]
    monkeypatch.setattr('science_picker.image_picker', picker)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    assert [item['name'] for item in calls[-1]['items']] == names
    assert all(item['image'].startswith('data:image/png;base64,') for item in calls[-1]['items'])
    event[0] = {'action': 'change', 'selected': names[:2] + ['other.fits'], 'event': 'one'}
    app.run()
    assert app.session_state['workflow_bias_frames'] == names[:2]
    assert [item['name'] for item in calls[-1]['items']] == names  # deselected files stay visible
    app.button(key='workflow_bias_combine').click().run()
    assert not app.exception and not app.error
    np.testing.assert_allclose(app.session_state['workflow_bias_last'][1].data, 20.)
    assert app.session_state['workflow_bias_last'][1].header['NCOMBINE'] == 2
    event[0] = {'action': 'change', 'selected': names, 'event': 'two'}
    app.run()
    assert app.session_state['workflow_bias_frames'] == names
    assert app.button(key='workflow_bias_show').disabled
    event[0] = None
    app.text_input(key='workflow_bias_pattern').set_value(r'^bias_[12]\.fits$').run()
    assert [item['name'] for item in calls[-1]['items']] == names[:2]
    event[0] = {'action': 'change', 'selected': [], 'event': 'three'}
    app.run()
    assert not app.session_state['workflow_bias_frames']
    assert app.button(key='workflow_bias_combine').disabled
    app.text_input(key='workflow_bias_pattern').set_value('[').run()
    assert app.error and not app.exception
    assert app.button(key='workflow_bias_combine').disabled


@pytest.mark.parametrize('kind', ['bias', 'flat', 'science'])
def test_inline_view_preserves_selection(tmp_path, monkeypatch, kind):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    for i in range(2):
        fits.writeto(tmp_path / f'{kind}_{i}.fits', np.ones((8, 8)))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.text_input(key=f'workflow_{kind}_pattern').set_value(kind + '.*').run()
    chosen = list(app.session_state[f'workflow_{kind}_frames'])
    # The first half of a double-click may have reached Python already.
    # The view event carries the final selection after both clicks.
    app.session_state[f'workflow_{kind}_frames'] = chosen[:-1]
    monkeypatch.setattr('science_picker.image_picker', lambda **kwargs: {
        'action': 'view', 'name': f'{kind}_1.fits', 'selected': chosen, 'event': 'view-one'})
    app.run()
    assert not app.exception
    assert app.session_state['workflow_file_view'][1].name == f'{kind}_1.fits'
    assert app.session_state[f'workflow_{kind}_frames'] == chosen


def test_result_variants_select_main_view_without_reprocessing(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    for kind, base in [('bias', 10.), ('flat', 100.)]:
        for i in range(2):
            fits.writeto(tmp_path / f'{kind}_{i}.fits', base + np.arange(64.).reshape(8, 8),
                         fits.Header({'IMAGETYP': kind.upper(), 'FILTER': 'Blue', 'EXPTIME': 2.}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.button(key='workflow_bias_combine').click().run()
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    calls, event = [], [None]
    def picker(**kwargs):
        if kwargs.get('preview_only'):
            calls.append(kwargs)
            if event[0] and kwargs['key'].startswith(event[0][0]):
                return {'action': 'view', 'name': str(event[0][1]), 'event': str(event[0])}
        return None
    monkeypatch.setattr('science_picker.image_picker', picker)
    def unexpected(*args, **kwargs):
        raise AssertionError('Preview must not recompute calibration')
    monkeypatch.setattr('workflow_ui.run_with_timeout', unexpected)
    for index in (0, 1, 2, 3):
        event[0] = ('flat_result_', index)
        app.run()
        assert not app.exception
        assert app.session_state['workflow_stage_view'] == 'flat'
        assert app.session_state['workflow_flat_subtract_bias'] == (index >= 2)
        assert app.session_state['viewer_contrast'] == bool(index % 2)
    assert len(next(call for call in calls if call['key'].startswith('flat_result_'))['items']) == 4
    assert any('Blue · 2.0 s' in item.value for item in app.markdown)
    for index in (0, 1):
        event[0] = ('bias_result_', index)
        app.run()
        assert not app.exception
        assert app.session_state['workflow_stage_view'] == 'bias'
        assert app.session_state['viewer_contrast'] == bool(index)


@pytest.mark.parametrize('kind', ['bias', 'flat', 'science'])
def test_selection_reuses_previews_and_never_loads_main_image(tmp_path, monkeypatch, kind):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    for index in range(3):
        fits.writeto(tmp_path / f'{kind}_{index}.fits', np.ones((8, 8)))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.text_input(key=f'workflow_{kind}_pattern').set_value(kind + '.*').run()

    def unexpected(*args, **kwargs):
        raise AssertionError('A selection must reuse previews and must not load a full image.')

    monkeypatch.setattr('science_picker.preview_items', unexpected)
    monkeypatch.setattr('workflow_ui.load', unexpected)
    monkeypatch.setattr('science_ui.load', unexpected)
    monkeypatch.setattr('science_picker.image_picker', lambda **kwargs: (
        {'action': 'change', 'selected': [f'{kind}_1.fits'], 'event': 'selection-only'}
        if kwargs['key'].startswith(kind + '_inline_') else None))
    app.run()
    assert not app.exception
    assert app.session_state[f'workflow_{kind}_frames'] == [f'{kind}_1.fits']
    assert 'workflow_file_view' not in app.session_state
    assert 'workflow_stage_view' not in app.session_state
