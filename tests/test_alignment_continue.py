from pathlib import Path
import tempfile

import numpy as np
import pytest
from astropy.io import fits
from streamlit.testing.v1 import AppTest

from processing_jobs import ProcessingDecisionRequired, run_with_timeout


@pytest.mark.parametrize('action', ['continue', 'stop'])
def test_alignment_asks_after_three_minutes_and_preserves_job(tmp_path, monkeypatch, action):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    for kind, level in [('bias', 10.), ('flat', 110.), ('science', 210.)]:
        for index in range(2):
            fits.writeto(tmp_path / f'{kind}_Blue_{index}.fits', np.full((12, 12), level),
                         fits.Header({'IMAGETYP': 'OBJECT' if kind == 'science' else kind.upper(),
                                      'FILTER': 'Blue'}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.text_input(key='workflow_science_pattern').set_value('science.*').run()
    app.button(key='workflow_science_correct').click().run()
    assert not app.exception and not app.error

    class Job:
        cancelled = False
        resumes = 0

        def cancel(self):
            self.cancelled = True

        def resume(self, timeout):
            assert timeout == 180
            self.resumes += 1
            if self.resumes == 1:
                raise ProcessingDecisionRequired(self)
            paths = list(app.session_state['workflow_science_corrected'][1].values())
            return (paths, [(0., 0.)] * len(paths)), tempfile.TemporaryDirectory()

    job = Job()
    starts = []

    def runner(function, *args, **kwargs):
        if function.__name__ == 'align_science_frames':
            assert kwargs['timeout'] == 180 and kwargs['allow_continue']
            starts.append(function)
            raise ProcessingDecisionRequired(job)
        return run_with_timeout(function, *args, **kwargs)

    monkeypatch.setattr('science_ui.run_with_timeout', runner)
    app.button(key='workflow_science_align').click().run()
    assert not app.exception and not app.error
    assert any('3 minutos' in message.value for message in app.warning)
    assert app.button(key='workflow_science_align').disabled
    assert app.button(key='workflow_science_combine').disabled
    if action == 'continue':
        app.button(key='workflow_science_alignment_continue').click().run()
        assert not app.exception and app.button(key='workflow_science_alignment_continue')
        app.button(key='workflow_science_alignment_continue').click().run()
        assert not app.exception and not app.error
        assert job.resumes == 2 and not job.cancelled
        assert 'workflow_science_aligned' in app.session_state
        assert not app.button(key='workflow_science_combine').disabled
    else:
        app.button(key='workflow_science_alignment_stop').click().run()
        assert not app.exception and job.cancelled
        assert 'workflow_science_aligned' not in app.session_state
        assert not app.button(key='workflow_science_align').disabled
    assert len(starts) == 1
    assert 'workflow_science_alignment_pending' not in app.session_state
