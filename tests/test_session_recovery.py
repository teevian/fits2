import pytest
import numpy as np
from streamlit.testing.v1 import AppTest

from imageProcessing import frame
from session_recovery import copy_containers, recovery_store


@pytest.fixture(autouse=True)
def isolate_recovery(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    recovery_store.clear()
    yield
    recovery_store.clear()


SCRIPT = '''
import streamlit as st
from session_recovery import begin_session, checkpoint
@checkpoint
def main():
    begin_session()
    st.number_input('Radius', value=5, key='photo_radius')
    st.multiselect('Frames', ['a', 'b'], key='workflow_science_frames')
    if st.button('Generate', key='generate'):
        st.session_state['generated_count'] = st.session_state.get('generated_count', 0) + 1
        st.session_state['workflow_color_last'] = ('signature', 'pixels')
    st.write(st.session_state.get('generated_count', 0))
main()
'''


def test_reload_restores_settings_products_without_replaying_buttons():
    recovery_store.clear()
    first = AppTest.from_string(SCRIPT).run()
    first.number_input[0].set_value(11)
    first.multiselect[0].set_value(['b'])
    first.button[0].click().run()
    token = first.query_params['session']
    second = AppTest.from_string(SCRIPT)
    second.query_params['session'] = token
    second.run()
    assert not second.exception
    assert second.number_input[0].value == 11
    assert second.multiselect[0].value == ['b']
    assert second.session_state['workflow_color_last'] == ('signature', 'pixels')
    assert second.session_state['generated_count'] == 1
    second.number_input[0].set_value(13).run()
    third = AppTest.from_string(SCRIPT)
    third.query_params['session'] = token
    third.run()
    assert third.number_input[0].value == 13
    unrelated = AppTest.from_string(SCRIPT).run()
    assert unrelated.number_input[0].value == 13
    assert unrelated.multiselect[0].value == ['b']


def test_generated_frame_identity_and_arrays_survive_container_copy():
    product = frame(np.arange(25).reshape(5, 5), name='master')
    original = {'frames': ['a'], 'product': ('signature', product)}
    restored = copy_containers(original)
    restored['frames'].append('b')
    assert original['frames'] == ['a']
    assert restored['product'][1] is product
    np.testing.assert_array_equal(restored['product'][1].data, product.data)


def test_checkpoint_saves_before_rerun():
    recovery_store.clear()
    app = AppTest.from_string(SCRIPT.replace("st.write(st.session_state.get('generated_count', 0))", "if st.button('Switch'):\n        st.session_state['workflow_stage_view'] = 'color'\n        st.rerun()" )).run()
    app.button[1].click().run()
    token = app.query_params['session']
    restored = AppTest.from_string(SCRIPT)
    restored.query_params['session'] = token
    restored.run()
    assert not restored.exception
    assert restored.session_state['workflow_stage_view'] == 'color'


def test_real_app_restores_generated_master_and_view(tmp_path, monkeypatch):
    from pathlib import Path
    from astropy.io import fits
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    for index, value in enumerate((8., 12.)):
        fits.writeto(tmp_path / f'bias_{index}.fits', np.full((6, 8), value),
                     fits.Header({'IMAGETYP': 'BIAS'}))
    filename = str(Path(__file__).resolve().parents[1] / 'app.py')
    app = AppTest.from_file(filename, default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.button(key='workflow_bias_combine').click().run()
    assert not app.exception
    master = app.session_state['workflow_bias_last'][1]
    token = app.query_params['session']
    recovery_store.clear()  # Simulate losing all server memory; open the bare URL.
    restored = AppTest.from_file(filename, default_timeout=30)
    restored.run()
    assert not restored.exception
    from processing_ui import fingerprint
    recovered = restored.session_state['workflow_bias_last'][1]
    assert recovered is not master
    assert fingerprint(recovered) == fingerprint(master)
    np.testing.assert_array_equal(recovered.data, master.data)
    np.testing.assert_allclose(master.data, 10.)
    assert restored.session_state['workflow_stage_view'] == 'bias'
    assert restored.session_state['workflow_bias_frames'] == app.session_state['workflow_bias_frames']
    assert any(master.name == item.value for item in restored.caption)


def test_disk_restores_rgb_bytes_and_fingerprints_in_new_process(tmp_path):
    import subprocess, os
    from pathlib import Path
    from astropy.io import fits
    from color_composition import ColorComposite
    from processing_ui import fingerprint
    from session_disk import write_snapshot, read_snapshot
    import sys
    image = ColorComposite(np.arange(36.).reshape(3,4,3), fits.Header({'FILTER':'RGB'}), np.zeros((3,4),dtype=bool), 'final.png')
    original_id = fingerprint(image)
    write_snapshot('token', {'rgb': image, 'original_id': original_id, 'archive': b'FITS bytes'}, {})
    code = '''
from pathlib import Path
import workspace_settings
workspace_settings.SETTINGS_FILE = Path(__import__('sys').argv[1])
from session_disk import read_snapshot
from processing_ui import fingerprint
token, saved = read_snapshot()
assert token == 'token'
assert fingerprint(saved['rgb']) == saved['original_id']
assert saved['rgb'].data.shape == (3,4,3)
assert saved['rgb'].data.sum() == 630
assert saved['archive'] == b'FITS bytes'
assert saved['rgb'].png().startswith(b'\\x89PNG')
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path/'settings.json')], check=True,
                   env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])})


def test_hidden_widget_configuration_survives_close_restart_and_reopen():
    source='''
import streamlit as st
from session_recovery import begin_session, checkpoint
@checkpoint
def main():
    begin_session()
    opened=st.checkbox('Options', key='options_open')
    if opened:
        st.number_input('Reference magnitude', value=10., key='abcdefref_mag')
main()
'''
    app=AppTest.from_string(source).run()
    app.checkbox[0].check().run()
    app.number_input[0].set_value(12.34).run()
    app.checkbox[0].uncheck().run()
    app.run()  # Streamlit has now cleaned up the hidden widget.
    recovery_store.clear()
    restored=AppTest.from_string(source).run()
    restored.checkbox[0].check().run()
    assert not restored.exception
    assert restored.number_input[0].value == 12.34


def test_tampered_disk_snapshot_is_rejected():
    import sqlite3
    from session_disk import write_snapshot, read_snapshot, directory
    write_snapshot('token', {'value': 1}, {})
    with sqlite3.connect(directory()/'latest.sqlite3') as db:
        db.execute("UPDATE state SET payload=? WHERE key='value'", (b'invalid',))
    with pytest.raises(ValueError, match='danificada'):
        read_snapshot()


def test_generated_science_and_rgb_survive_worker_cleanup_and_restart(tmp_path, monkeypatch):
    from pathlib import Path
    from astropy.io import fits
    import shutil
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path/'settings.json')
    y,x=np.mgrid[:64,:80]
    stars=sum(1000*np.exp(-((x-cx)**2+(y-cy)**2)/5) for cx,cy in [(15,15),(40,18),(62,42),(24,45)])
    for i in range(2):
        fits.writeto(tmp_path/f'bias_{i}.fits',np.full((64,80),10.),fits.Header({'IMAGETYP':'BIAS'}))
        for band in ['Red','Green','Blue']:
            fits.writeto(tmp_path/f'flat_{band}_{i}.fits',np.full((64,80),1010.),
                         fits.Header({'IMAGETYP':'FLAT','FILTER':band,'EXPTIME':1.}))
            fits.writeto(tmp_path/f'deneb_{band}_{i}.fits',100+stars,
                         fits.Header({'IMAGETYP':'OBJECT','OBJECT':'deneb','FILTER':band,'EXPTIME':10.}))
    filename=str(Path(__file__).resolve().parents[1]/'app.py')
    app=AppTest.from_file(filename,default_timeout=60)
    app.session_state['folder_path']=str(tmp_path)
    app.run()
    app.button(key='workflow_auto').click().run()
    app.button(key='workflow_science_align').click().run()
    assert not app.exception and not app.error
    app.button(key='workflow_science_generate').click().run()
    assert not app.exception and not app.error
    assert '_recovery_save_error' not in app.session_state
    pixels=app.session_state['workflow_color_last'][1].data.copy()
    paths=list(app.session_state['workflow_science_corrected'][1].values())+list(app.session_state['workflow_science_aligned'][1])
    for key in ('workflow_science_correction_workspace','workflow_science_alignment_workspace'):
        app.session_state[key].cleanup()
    assert all(not path.exists() for path in paths)
    recovery_store.clear()
    restored=AppTest.from_file(filename,default_timeout=60).run()
    assert not restored.exception and not restored.error
    assert all(path.exists() for path in paths)
    assert restored.session_state['workflow_stage_view']=='color'
    assert not restored.button(key='workflow_science_generate').disabled
    np.testing.assert_array_equal(restored.session_state['workflow_color_last'][1].data,pixels)
    assert any(restored.session_state['workflow_color_last'][1].name == item.value for item in restored.caption)
