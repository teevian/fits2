from pathlib import Path
import time

import numpy as np
import pytest
from astropy.io import fits

import processing_jobs
from processing_jobs import run_with_timeout
from imageProcessing import read_frame, to_fits, frame
from imageProcessing import align_calibration_frames


def test_timeout_kills_worker_and_cleans_temporary_files(monkeypatch):
    launched = []
    popen = processing_jobs.subprocess.Popen

    def track(*args, **kwargs):
        process = popen(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(processing_jobs.subprocess, 'Popen', track)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='Timeout: operation stopped after 0.2 seconds'):
        run_with_timeout(time.sleep, 10, timeout=0.2)
    assert time.monotonic() - started < 2
    assert launched[0].poll() is not None
    assert not Path(launched[0].args[-1]).exists()
    # A cancelled job must not prevent the next one from completing.
    assert run_with_timeout(int, '42', timeout=2) == 42


def test_worker_returns_validation_errors():
    with pytest.raises(ValueError, match='invalid literal'):
        run_with_timeout(int, 'bad', timeout=2)


def test_retained_worker_can_continue_without_restarting_or_be_cancelled():
    from processing_jobs import ProcessingDecisionRequired
    with pytest.raises(ProcessingDecisionRequired) as waiting:
        run_with_timeout(time.sleep, .1, timeout=.02, allow_continue=True)
    job = waiting.value.job
    root = Path(job.workspace.name)
    pid = job.process.pid
    assert root.exists() and job.process.poll() is None
    assert job.resume(timeout=3) is None
    assert job.process.pid == pid and job.process.returncode == 0
    assert not root.exists()

    with pytest.raises(ProcessingDecisionRequired) as waiting:
        run_with_timeout(time.sleep, 10, timeout=.02, allow_continue=True)
    job = waiting.value.job
    root = Path(job.workspace.name)
    job.cancel()
    assert job.process.poll() is not None and not root.exists()
    job.cancel()  # Cancellation is safe if a session is later disposed.


def test_disk_backed_alignment_preserves_sources_and_registration(tmp_path):
    from scipy.ndimage import shift
    y, x = np.indices((512, 768))
    values = 100 + 20 * np.exp(-((x - 230)**2 + (y - 190)**2) / 150)
    values += 15 * np.exp(-((x - 550)**2 + (y - 350)**2) / 240)
    header = fits.Header({'IMAGETYP': 'FLAT', 'FILTER': 'Blue'})
    sources = []
    originals = []
    for index, data in enumerate((values, shift(values, (8, -12), mode='nearest'))):
        path = tmp_path / f'flat_Blue_{index}.fits'
        original = to_fits(frame(data, header, name=path.name))
        path.write_bytes(original)
        sources.append(path)
        originals.append(original)
    (aligned, offsets), workspace = run_with_timeout(
        align_calibration_frames, sources, 'flat', 30, keep_workspace=True)
    try:
        assert all(isinstance(item, Path) and item.exists() for item in aligned)
        assert [item.name for item in aligned] == [item.name for item in sources]
        np.testing.assert_allclose(offsets[1], (-8, 12), atol=0.5)
        assert read_frame(aligned[1]).header['ALNSTAT'] == 'SHIFTED'
        assert read_frame(aligned[0]).header['ALNCOUNT'] == 1
        assert [path.read_bytes() for path in sources] == originals
    finally:
        workspace.cleanup()
    assert not aligned[0].exists()


def test_ui_timeout_discards_partial_results_and_allows_retry(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    for name, kind, value in [('science.fits', 'OBJECT', 120),
                               ('bias1.fits', 'BIAS', 10), ('bias2.fits', 'BIAS', 10),
                               ('flat_Blue_1.fits', 'FLAT', 110),
                               ('flat_Blue_2.fits', 'FLAT', 120)]:
        header = fits.Header({'IMAGETYP': kind, 'FILTER': 'Blue'})
        (tmp_path / name).write_bytes(to_fits(frame(np.full((12, 12), value), header, name=name)))
    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
workflow_panel(read_frame(folder / 'science.fits'), 'source', folder)
'''
    app = AppTest.from_string(source, default_timeout=10).run()
    monkeypatch.setattr(processing_jobs, 'OPERATION_TIMEOUT', 0.01)
    for kind in ('bias', 'flat'):
        for action in ('combine',):
            app.button(key=f'workflow_{kind}_{action}').click().run()
            assert not app.exception
            assert any('Timeout:' in error.value for error in app.error)
            assert f'workflow_{kind}_aligned' not in app.session_state
            assert f'workflow_{kind}_last' not in app.session_state
    monkeypatch.setattr(processing_jobs, 'OPERATION_TIMEOUT', 5)
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    assert list(app.session_state['workflow_flat_last'][1]) == ['Blue']
