"""Bounded processing jobs, isolated from the Streamlit script thread."""
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
from time import monotonic

from imageProcessing import combine_flats_job, run_workflow_job


OPERATION_TIMEOUT = 30.0
ALIGNMENT_TIMEOUT = 180.0
CALIBRATION_TIMEOUT = 180.0


class ProcessingDecisionRequired(Exception):
    """The worker is retained until the user continues or cancels it."""
    def __init__(self, job):
        self.job = job
        super().__init__('Processing needs more time.')


class RetainedJob:
    def __init__(self, process, workspace, keep_workspace):
        self.process = process
        self.workspace = workspace
        self.keep_workspace = keep_workspace
        self.suspended = False

    def pause(self):
        if self.process.poll() is None and hasattr(signal, 'SIGSTOP'):
            try:
                self.process.send_signal(signal.SIGSTOP)
                self.suspended = True
            except ProcessLookupError:
                pass

    def cancel(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.workspace is not None:
            self.workspace.cleanup()
            self.workspace = None

    def resume(self, timeout=ALIGNMENT_TIMEOUT):
        if self.suspended and self.process.poll() is None:
            self.process.send_signal(signal.SIGCONT)
            self.suspended = False
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.pause()
            raise ProcessingDecisionRequired(self) from None
        try:
            root = Path(self.workspace.name)
            if self.process.returncode or not (root / 'result.pkl').exists():
                raise ValueError('Processing stopped unexpectedly. Try fewer frames.')
            with (root / 'result.pkl').open('rb') as stream:
                successful, result = pickle.load(stream)
            if not successful:
                raise ValueError(result)
            if self.keep_workspace:
                workspace, self.workspace = self.workspace, None
                return result, workspace
            return result
        finally:
            self.cancel()

    def __del__(self):
        self.cancel()


def run_with_timeout(function, *args, timeout=None, keep_workspace=False, allow_continue=False, **kwargs):
    """Bound calculations; optionally retain an overdue worker for user approval."""
    timeout = OPERATION_TIMEOUT if timeout is None else timeout
    started = monotonic()
    workspace = tempfile.TemporaryDirectory(prefix='fits-job-')
    root = Path(workspace.name)
    process = None
    try:
        if keep_workspace:
            kwargs['output_folder'] = root / 'aligned'
        with (root / 'request.pkl').open('wb') as stream:
            pickle.dump((function.__module__, function.__name__, args, kwargs), stream,
                        protocol=pickle.HIGHEST_PROTOCOL)
        environment = os.environ.copy()
        environment.update(TMPDIR=str(root), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        with (root / 'worker.log').open('wb') as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name('processing_worker.py')), str(root)],
                env=environment, stdout=log, stderr=log)
            try:
                process.wait(timeout=max(0.001, timeout - (monotonic() - started)))
            except subprocess.TimeoutExpired:
                if allow_continue:
                    job = RetainedJob(process, workspace, keep_workspace)
                    job.pause()
                    process = workspace = None
                    raise ProcessingDecisionRequired(job) from None
                raise TimeoutError(f'Timeout: operation stopped after {timeout:g} seconds.') from None
        if process.returncode or not (root / 'result.pkl').exists():
            raise ValueError('Processing stopped unexpectedly. Try fewer frames.')
        with (root / 'result.pkl').open('rb') as stream:
            successful, result = pickle.load(stream)
        if not allow_continue and monotonic() - started > timeout:
            raise TimeoutError(f'Timeout: operation stopped after {timeout:g} seconds.')
        if not successful:
            raise ValueError(result)
        if keep_workspace:
            result = (result, workspace)
            workspace = None
        return result
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if workspace is not None:
            workspace.cleanup()
