import numpy as np
from astropy.io import fits
from streamlit.testing.v1 import AppTest

from imageProcessing import frame, to_fits


def test_processing_activation_execution_and_stale_result(tmp_path):
    for name, level in [('bias_a.fits', 100), ('bias_b.fits', 102)]:
        (tmp_path / name).write_bytes(to_fits(frame(np.full((4,4), level),
                                   fits.Header({'IMAGETYP': 'BIAS'}), name=name)))
    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from processing_ui import processing_panel
folder = Path({str(tmp_path)!r})
current = read_frame(folder / 'bias_a.fits')
result = processing_panel(current, 'fixed-source', folder)
st.session_state.showing = result.name if result is not None else None
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    assert not app.exception
    assert app.selectbox(key='processing_operation').value == 'None'
    app.selectbox(key='processing_operation').select('Master bias').run()
    assert not app.exception
    app.multiselect(key='inputs_Master bias_').set_value(['bias_a.fits', 'bias_b.fits']).run()
    app.button(key='process_run').click().run()
    assert not app.exception, app.exception
    assert not app.error, app.error
    assert app.session_state['showing'] == '001_master_bias_x2.fits'
    np.testing.assert_allclose(app.session_state['reduction_products']['001_master_bias_x2.fits'].data, 101)
    app.number_input(key='sigma_Master bias').set_value(4.).run()
    assert app.session_state['showing'] is None
    app.selectbox(key='processing_operation').select('Image mask').run()
    assert not app.multiselect
    app.button(key='process_run').click().run()
    assert not app.exception
    assert app.session_state['showing'].endswith('_mask.fits')
    app.selectbox(key='processing_operation').select('None').run()
    assert app.session_state['showing'] is None
    assert len(app.session_state['reduction_products']) == 2


def test_master_reuse_and_science_batch(tmp_path):
    response = np.array([[0.5, 1.], [1., 1.5]])
    for name, data, kind in [
        ('bias_a.fits', np.full((2,2), 100), 'BIAS'),
        ('bias_b.fits', np.full((2,2), 100), 'BIAS'),
        ('flat_a.fits', response*1000+100, 'FLAT'),
        ('flat_b.fits', response*3000+100, 'FLAT'),
        ('science_a.fits', response*200+100, 'OBJECT'),
        ('science_b.fits', response*400+100, 'OBJECT'),
    ]:
        (tmp_path / name).write_bytes(to_fits(frame(data, fits.Header({'IMAGETYP': kind, 'FILTER':'Blue'}), name=name)))
    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from processing_ui import processing_panel
folder = Path({str(tmp_path)!r})
result = processing_panel(read_frame(folder / 'science_a.fits'), 'source', folder)
st.session_state.showing = result.name if result is not None else None
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    app.selectbox(key='processing_operation').select('Master bias').run()
    app.multiselect(key='inputs_Master bias_').set_value(['bias_a.fits', 'bias_b.fits']).run()
    app.button(key='process_run').click().run()
    app.selectbox(key='processing_operation').select('Flat correction').run()
    next(r for r in app.selectbox if r.label == 'Action').set_value('Create master flat').run()
    app.multiselect(key='inputs_Flat correction_Create master flat').set_value(['flat_a.fits', 'flat_b.fits']).run()
    app.checkbox(key='flat_create_bias_on').set_value(True).run()
    app.selectbox(key='flat_create_bias').select('✓ 001_master_bias_x2.fits').run()
    app.button(key='process_run').click().run()
    assert not app.exception, app.exception
    assert not app.error, app.error
    np.testing.assert_allclose(app.session_state['reduction_products']['002_master_flat_x2.fits'].data, response)
    app.selectbox(key='processing_operation').select('Reduce science images').run()
    next(r for r in app.selectbox if r.label == 'Apply to').set_value('Multiple images').run()
    app.multiselect(key='science_inputs').set_value(['science_a.fits','science_b.fits']).run()
    app.checkbox(key='science_bias_on').set_value(True).run()
    app.selectbox(key='science_bias').select('✓ 001_master_bias_x2.fits').run()
    app.checkbox(key='science_flat_on').set_value(True).run()
    app.selectbox(key='science_flat').select('✓ 002_master_flat_x2.fits').run()
    app.button(key='process_run').click().run()
    assert not app.exception, app.exception
    assert not app.error, app.error
    products = app.session_state['reduction_products']
    np.testing.assert_allclose(products['003_science_a_reduced.fits'].data, 200)
    np.testing.assert_allclose(products['004_science_b_reduced.fits'].data, 400)
