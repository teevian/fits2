"""Final LRGB composition controls below the science masters."""
import streamlit as st

from color_composition import generate_color
from processing_jobs import run_with_timeout, ALIGNMENT_TIMEOUT, ProcessingDecisionRequired
from processing_ui import fingerprint
from science_picker import result_previews


def color_panel(masters, max_shift):
    signature = repr(('lrgb-psf-v2', tuple((key, fingerprint(value)) for key, value in masters.items()), max_shift))
    pending = st.session_state.get('workflow_color_pending')
    if pending and pending[0] != signature:
        pending[1].cancel()
        st.session_state.pop('workflow_color_pending', None)
        pending = None
    clicked = st.button('Generate', key='workflow_science_generate', disabled=bool(pending))
    st.caption('LRGB · Red + Green + Blue for color; black (clear) for luminance. Matched color sharpness and stellar color balance.')
    resume = False
    if pending:
        st.warning('A geração atingiu 3 minutos. O progresso foi preservado.')
        more, stop = st.columns(2)
        resume = more.button('Continuar', key='workflow_color_continue')
        if stop.button('Parar', key='workflow_color_stop'):
            pending[1].cancel()
            st.session_state.pop('workflow_color_pending', None)
            st.rerun(scope='app')
    if clicked or resume:
        try:
            with st.spinner('Aligning filter masters and generating LRGB…'):
                if pending:
                    result = pending[1].resume(timeout=ALIGNMENT_TIMEOUT)
                else:
                    result = run_with_timeout(generate_color, masters, max_shift, use_luminance=True,
                                              timeout=ALIGNMENT_TIMEOUT, allow_continue=True)
            st.session_state.pop('workflow_color_pending', None)
            st.session_state.workflow_color_last = (signature, result)
            st.session_state.pop('workflow_color_error', None)
            st.session_state.workflow_stage_view = 'color'
            st.rerun(scope='app')
        except ProcessingDecisionRequired as exc:
            st.session_state.workflow_color_pending = (signature, exc.job)
            st.rerun(scope='app')
        except (ValueError, OSError, TypeError, KeyError, MemoryError) as exc:
            st.session_state.pop('workflow_color_pending', None)
            st.session_state.workflow_color_error = (signature, str(exc))
    error = st.session_state.get('workflow_color_error')
    if error and error[0] == signature:
        st.error(error[1])
    last = st.session_state.get('workflow_color_last')
    if not last or last[0] != signature:
        return None
    result = last[1]
    choice = result_previews([(result, 'Final LRGB · Without contrast', False),
                              (result, 'Final LRGB · With contrast', True)], key=f'color_result_{signature}')
    if choice is not None:
        st.session_state.viewer_contrast = bool(choice)
        st.session_state.workflow_stage_view = 'color'
        st.rerun(scope='app')
    for column, contrast, label in zip(st.columns(2), (False, True), ('Without contrast', 'With contrast')):
        column.download_button(label + ' · PNG', result.png(contrast),
                               result.name.replace('.png', '_contrast.png') if contrast else result.name,
                               'image/png', key=f'workflow_color_download_{contrast}')
    return result
