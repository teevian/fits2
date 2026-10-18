from concurrent.futures import Future

from streamlit.testing.v1 import AppTest

from imageProcessing import calibration_archive


def test_zip_dialog_renders_immediately_and_handles_completion_timeout_and_retry(monkeypatch):
    import download_ui
    pending = []

    class HeldExecutor:
        def submit(self, function, *args):
            future = Future()
            pending.append((future, function, args))
            return future

    monkeypatch.setattr(download_ui, '_zip_executor', lambda: HeldExecutor())
    app = AppTest.from_string('''
import numpy as np
import streamlit as st
from imageProcessing import frame
from download_ui import calibration_download_dialog
if 'versions' not in st.session_state:
    values = np.arange(144.).reshape(12, 12) + 4000.
    st.session_state.versions = {
        False: {'Blue': frame(values, name='master_flat_blue_x2.fits')},
        True: {'Blue': frame(values - 543., name='corrected_flat_blue_x2.fits')}}
calibration_download_dialog(st.session_state.versions, 'flats')
''').run()
    assert not app.exception
    assert len(pending) == 2
    assert {button.label for button in app.get('download_button')} == {'FITS ZIP', 'PNG ZIP'}
    assert all(button.proto.disabled for button in app.get('download_button'))
    assert any('Preparing ZIP' in caption.value for caption in app.caption)
    versions = app.session_state['versions']
    pending[0][0].set_result(calibration_archive(versions, 'fits'))
    pending[1][0].set_exception(TimeoutError('Timeout: operation stopped after 30 seconds.'))
    app.run()
    assert not app.exception
    assert len(pending) == 2
    buttons = {button.label: button for button in app.get('download_button')}
    assert not buttons['FITS ZIP'].proto.disabled
    assert buttons['PNG ZIP'].proto.disabled
    assert any('Timeout:' in error.value for error in app.error)
    app.run()
    assert len(pending) == 2  # A failed job must not retry endlessly.
    app.button(key='archive_retry_png').click().run()
    assert not app.exception and not app.error
    assert len(pending) == 3
    pending[2][0].set_result(calibration_archive(versions, 'png'))
    app.run()
    assert not app.exception and not app.error
    assert all(not button.proto.disabled for button in app.get('download_button'))
    assert not any('Preparing ZIP' in caption.value for caption in app.caption)
    app.run()
    assert len(pending) == 3  # Reopening reuses the finished exports.
    # Removing a flat invalidates only its corresponding download cache.
    app.session_state['versions'] = {False: dict(versions[False])}
    app.run()
    assert not app.exception
    assert len(pending) == 5
    for future, function, args in pending[3:]:
        future.set_result(calibration_archive(args[0], args[1]))
    app.run()
    assert all(not button.proto.disabled for button in app.get('download_button'))
