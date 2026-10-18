from pathlib import Path

import numpy as np
from astropy.io import fits
from streamlit.testing.v1 import AppTest


def test_show_groups_originals_with_one_cached_master_per_group(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    for index in range(2):
        fits.writeto(tmp_path / f'bias{index}.fits', np.full((12, 12), 10. + index),
                     fits.Header({'IMAGETYP': 'BIAS'}))
        for exposure in (1., 2.):
            fits.writeto(tmp_path / f'flat_Blue_{exposure:g}_{index}.fits',
                         np.arange(144.).reshape(12, 12) + 100. * exposure,
                         fits.Header({'IMAGETYP': 'FLAT', 'FILTER': 'Blue', 'EXPTIME': exposure}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    assert app.button(key='workflow_bias_show').disabled
    assert app.button(key='workflow_flat_show').disabled
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    bias = app.session_state['workflow_bias_last'][1]
    versions = app.session_state['workflow_flat_versions'][1]

    def unexpected(*args, **kwargs):
        raise AssertionError('Showing calibration products must not recombine them.')

    monkeypatch.setattr('workflow_ui.run_with_timeout', unexpected)
    app.button(key='workflow_bias_show').click().run()
    assert not app.exception and not app.error
    assert len(app.get('imgs')) == 3
    captions = [item.value for item in app.caption]
    assert captions.count(bias.name) == 1
    assert 'bias0.fits' in captions and 'bias1.fits' in captions
    app.run()  # Close the dialog without changing the existing products.
    for corrected in (True, False):
        if not corrected:
            app.button(key='workflow_flat_bias_option').click().run()
        app.button(key='workflow_flat_show').click().run()
        assert not app.exception and not app.error
        assert len(app.get('imgs')) == 6
        assert [item.value for item in app.subheader] == ['File', 'Blue · 1 s', 'Blue · 2 s']
        captions = [item.value for item in app.caption]
        groups = [index for index, caption in enumerate(captions) if caption == '2 originals → 1 master']
        assert len(groups) == 2
        for first, (group, master) in zip(groups, versions[corrected].items()):
            names = app.session_state['workflow_flat_last'][3][group]
            assert captions[first + 1:first + 4] == names + [master.name]
        app.run()
    app.radio(key='workflow_flat_method').set_value('Median').run()
    assert app.button(key='workflow_flat_show').disabled
