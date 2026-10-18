"""Automatic science masters selected by local filename regex."""
from html import escape
import re

import streamlit as st

from download_ui import archive_download_dialog, image_save_button
from frame_selection import initialize_matching_frames, select_matching_frames
from processing_jobs import run_with_timeout, ALIGNMENT_TIMEOUT, ProcessingDecisionRequired
from processing_ui import fingerprint, load
from imageProcessing import (align_science_frames, combine_science_groups, correct_science_frames,
                      source_filter, source_target, _source_header)
from workspace_settings import load_science_targets, save_science_targets
from science_picker import inline_bias_picker, science_pairs_grid
from calibration_gallery import show_science, show_science_pairs


def science_panel(current, source_id, folder, catalog, bias, flats, contrast_option,
                  prepare_calibration=None, auto_correct=False):
    initialize_matching_frames('workflow_science_pattern', 'workflow_science_frames',
                               'hd226868.*', catalog, folder)
    if 'workflow_science_targets' not in st.session_state:
        st.session_state.workflow_science_targets = load_science_targets()
    targets = st.session_state.workflow_science_targets
    selected_names = [name for name in st.session_state.workflow_science_frames if name in catalog]
    selection = (str(folder), tuple(selected_names))
    if st.session_state.get('workflow_science_target_selection') != selection:
        active_targets, checked, added = set(), set(), False
        known = {target.casefold() for target in targets}
        for name in selected_names:
            try:
                value = catalog[name]
                target = source_target(value)
                identity = target.casefold()
                if identity in checked or identity == 'science':
                    continue
                checked.add(identity)
                kind = str(_source_header(value).get('IMAGETYP', '')).upper()
                if (any(token in kind for token in ('BIAS', 'ZERO', 'FLAT', 'DARK')) or
                        name.casefold().startswith(('bias', 'flat', 'dark'))):
                    continue
                active_targets.add(identity)
                if identity not in known:
                    targets.append(target)
                    known.add(identity)
                    added = True
            except (ValueError, OSError, TypeError, KeyError):
                continue
        if added:
            save_science_targets(targets)
        st.session_state.workflow_science_target_selection = selection
        st.session_state.workflow_science_active_targets = active_targets

    def choose_target(target):
        st.session_state.workflow_science_pattern = re.escape(target) + '*'
        select_matching_frames('workflow_science_pattern', 'workflow_science_frames', catalog)

    def view_file(value):
        st.session_state.workflow_file_view = (source_id, load(value))
        st.session_state.workflow_stage_view = 'file'
        st.rerun(scope='app')

    def remove_frame(label):
        st.session_state.workflow_science_frames.remove(label)

    def filename_button(column, name, key, selected):
        column.markdown(f'<span class="image-filename {selected}" style="display:none">'
                        f'{escape(name)}</span>', unsafe_allow_html=True)
        return column.button(name, key=f'workflow_image_name_{key}', use_container_width=True)

    preview = None
    with st.expander('3. Science', expanded=True):
        st.markdown('''<style>
    .st-key-workflow_science_pattern [data-baseweb="input"] {
        border:none; border-bottom:1px solid #7f91af; background:transparent; box-shadow:none;}
    .st-key-workflow_science_pattern input {padding:2px 6px; min-height:28px;}
    .st-key-workflow_science_targets button {
        border:none; outline:none; box-shadow:none; background:transparent; border-radius:0;
        min-height:28px; padding:4px 8px; display:flex; align-items:center; justify-content:center;}
    .st-key-workflow_science_targets button p {font-size:12px; margin:0;}
    .st-key-workflow_science_list [data-testid="stVerticalBlock"] {gap:2px !important;}
    [class*="st-key-workflow_science_cell_"]:has(.selected-image),
    [class*="st-key-workflow_science_master_row_"]:has(.selected-image) {background:var(--selection-background);}
    [class*="st-key-workflow_science_cell_"] button,
    [class*="st-key-workflow_science_master_row_"] button {
        opacity:0; pointer-events:none; height:28px; min-height:28px; padding:0 4px;
        display:flex; align-items:center; justify-content:center;
        border:none; outline:none; box-shadow:none;}
    [class*="st-key-workflow_science_master_row_"] button {opacity:1; pointer-events:auto;}
    [class*="st-key-workflow_science_cell_"]:hover button,
    [class*="st-key-workflow_science_cell_"]:focus-within button,
    [class*="st-key-workflow_science_master_row_"]:hover button,
    [class*="st-key-workflow_science_master_row_"]:focus-within button {opacity:1; pointer-events:auto;}
    [class*="st-key-workflow_science_cell_"] button p {
        white-space:normal; overflow-wrap:anywhere; font-size:11px;}
    [class*="st-key-workflow_science_cell_"] {min-width:0;}
    [class*="st-key-workflow_science_cell_"] [data-testid="stHorizontalBlock"] {
        gap:2px; flex-wrap:nowrap !important;}
    [class*="st-key-workflow_science_cell_"] [data-testid="stHorizontalBlock"] > div {
        flex:0 0 22px; width:22px !important; min-width:0;}
    [class*="st-key-workflow_science_cell_"] [data-testid="stHorizontalBlock"] >
        [class*="st-key-workflow_image_name_"] {flex:1 1 0; width:auto !important;}
    [class*="st-key-workflow_science_cell_"] [data-testid="stElementContainer"]:has(.image-filename) {
        display:none;}
    .st-key-workflow_science_method [data-baseweb="radio"] > div:first-child {display:none;}
    .st-key-workflow_science_method [data-baseweb="radio"] {
        display:flex; align-items:center; justify-content:center;
        min-height:28px; padding:0 8px; margin:0;}
    .st-key-workflow_science_method [data-baseweb="radio"] > div {margin:0; padding:0;}
    .st-key-workflow_science_method [data-baseweb="radio"] p {margin:0; line-height:28px;}
    .st-key-workflow_science_shift [data-baseweb="input"] {
        border:none; border-bottom:1px solid #7f91af; background:transparent; box-shadow:none;}
    .st-key-workflow_science_combine button, .st-key-workflow_science_align button,
    .st-key-workflow_science_correct button {
        border:1px solid #7f91af; background:rgba(127,145,175,.16); padding:4px 14px; cursor:pointer;}
    .st-key-workflow_science_combine button:hover, .st-key-workflow_science_align button:hover,
    .st-key-workflow_science_correct button:hover {
        background:rgba(127,145,175,.32); border-color:#a7b8d3;}
    </style>''', unsafe_allow_html=True)
        pattern = st.session_state.workflow_science_pattern.casefold()
        active = st.session_state.get('workflow_science_active_targets', set())
        selected_keys = []
        with st.container(horizontal=True, gap='small', key='workflow_science_targets'):
            for index, target in enumerate(targets):
                key = f'workflow_science_target_{index}'
                selected = not st.session_state.get('workflow_science_pattern_error') and (
                    target.casefold() in active or pattern in (
                        re.escape(target).casefold() + '*', re.escape(target).casefold() + '.*'))
                st.button(target, key=key, on_click=choose_target, args=(target,))
                if selected:
                    selected_keys.append(f'.st-key-{key} button')
        if selected_keys:
            st.markdown('<style>' + ','.join(selected_keys) +
                        '{background:var(--selection-background);}</style>', unsafe_allow_html=True)
        label, field = st.columns([1, 3], gap='small', vertical_alignment='center')
        label.markdown('Science files:')
        field.text_input('Science files:', key='workflow_science_pattern', label_visibility='collapsed',
                         placeholder='hd226868.*', on_change=select_matching_frames,
                         args=('workflow_science_pattern', 'workflow_science_frames', catalog))
        pattern_error = st.session_state.get('workflow_science_pattern_error')
        if pattern_error:
            st.error(pattern_error)
        inline_bias_picker(catalog, view_file, kind='science')
        labels = [name for name in st.session_state.workflow_science_frames if name in catalog]
        st.session_state.workflow_science_frames = labels
        sources = [catalog[name] for name in labels]
        def correction_signature():
            return repr(('science-normalized-v2', str(folder), tuple(map(fingerprint, sources)), fingerprint(bias),
                         tuple((key, fingerprint(value)) for key, value in flats.items())))

        alignment_inputs = correction_signature()
        correct_column, correction_show_column = st.columns(2, gap='small', vertical_alignment='center')
        correct_clicked = correct_column.button('Correct', key='workflow_science_correct',
                                    disabled=bool(pattern_error) or not sources)
        if correct_clicked or auto_correct:
            try:
                if bias is None or not flats:
                    if prepare_calibration is None:
                        raise ValueError('Create a master bias and corrected flats before correcting science frames.')
                    with st.spinner('Preparing master bias and flats…'):
                        bias, flats = prepare_calibration()
                    alignment_inputs = correction_signature()
                with st.spinner('Correcting individual science frames…'):
                    corrected, workspace = run_with_timeout(
                        correct_science_frames, sources, bias, flats, keep_workspace=True)
                previous = st.session_state.get('workflow_science_correction_workspace')
                st.session_state.workflow_science_correction_workspace = workspace
                st.session_state.workflow_science_corrected = (alignment_inputs, dict(zip(labels, corrected)))
                st.session_state.pop('workflow_science_correction_error', None)
                st.session_state.pop('workflow_science_aligned', None)
                if previous:
                    previous.cleanup()
                if auto_correct:
                    st.session_state.workflow_auto_completed = (str(folder), len(corrected))
                    st.session_state.workflow_file_view = (source_id, load(corrected[0]))
                    st.session_state.workflow_stage_view = 'file'
                st.rerun(scope='app')
            except (ValueError, OSError, TypeError, KeyError, MemoryError) as exc:
                st.session_state.workflow_science_correction_error = (alignment_inputs, str(exc))
        correction_error = st.session_state.get('workflow_science_correction_error')
        if correction_error and correction_error[0] == alignment_inputs:
            st.error(correction_error[1])
        corrections = st.session_state.get('workflow_science_corrected')
        corrected = bool(corrections and corrections[0] == alignment_inputs)
        corrected_files = corrections[1] if corrected else {}
        if correction_show_column.button('show', key='workflow_science_correct_show',
                                         disabled=not corrected or bool(pattern_error)):
            show_science_pairs([(name, catalog[name], corrected_files[name]) for name in labels], 'correction')
        st.caption('Combination uses 5σ clipping around the median with MAD scatter.')
        st.caption('Dark current and overscan are not corrected in this workflow; their relevance must be checked for the camera.')
        st.caption('Correction: (original science − master bias) / normalized, bias-corrected flat of the same filter. Units remain counts.')
        if corrected:
            st.caption(f'{len(corrected_files)} science frames corrected individually.')
            if st.button('Download all corrected frames (ZIP)', key='workflow_science_corrected_download_all'):
                archive_download_dialog(corrected_files, 'science_corrected_frames', display_png=True,
                                        contrast=st.session_state.get('viewer_contrast', False))
        file_view = st.session_state.get('workflow_file_view')
        viewed_name = (file_view[1].name if file_view and file_view[0] == source_id
                       and st.session_state.get('workflow_stage_view') == 'file' else
                       current.name if st.session_state.get('workflow_stage_view') is None else None)
        def image_item(column, value, index, label, corrected_item=False):
            mode = 'corrected' if corrected_item else 'original'
            suffix = f'corrected_{index}' if corrected_item else str(index)
            with column.container(key=f'workflow_science_cell_{mode}_{index}'):
                selected = 'selected-image' if value.name == viewed_name else ''
                st.markdown(f'<span class="image-filename {selected}" style="display:none"></span>',
                            unsafe_allow_html=True)
                with st.container(horizontal=True, gap=None, vertical_alignment='center'):
                    name_clicked = st.button(value.name, key=f'workflow_image_name_science_{suffix}',
                                             help=value.name, width='stretch')
                    view_clicked = st.button('', icon=':material/visibility:', help='View image', width=22,
                                             key=f'workflow_science_file_{suffix}')
                    image_save_button(st, value, f'workflow_science_save_{mode}_{label}',
                                      display_png=True, contrast=st.session_state.get('viewer_contrast', False))
                    st.button('×', key=f'workflow_science_remove_{mode}_{label}', width=22,
                              help='Remove frame', on_click=remove_frame, args=(label,))
                    if name_clicked or view_clicked:
                        view_file(value)
        if corrected:
            left, right = st.columns(2, gap='small')
            left.caption('Original')
            right.caption('Corrected')
            science_pairs_grid([(name, catalog[name], corrected_files[name]) for name in labels], view_file)
        if corrected:
            with st.expander('File actions'), st.container(height=160, border=False, key='workflow_science_list'):
                for index, label in enumerate(labels):
                    with st.container(key=f'workflow_science_pair_{index}'):
                        left, right = st.columns(2, gap='small')
                        image_item(left, catalog[label], index, label)
                        image_item(right, corrected_files[label], index, label, True)
        label, field = st.columns([2, 3], gap='small', vertical_alignment='center')
        label.markdown('Science max shift:', help='Maximum image displacement in pixels.')
        shift_limit = max(1, max(current.data.shape) - 2)
        default_shift = max(1, min(current.data.shape) - 2)
        max_shift = field.number_input('Science max shift:', 1, shift_limit, default_shift,
                                       key='workflow_science_shift', label_visibility='collapsed')
        alignment_signature = repr(('science-alignment-v3', alignment_inputs, max_shift, corrected))
        pending = st.session_state.get('workflow_science_alignment_pending')
        if pending and pending[0] != alignment_signature:
            pending[1].cancel()
            st.session_state.pop('workflow_science_alignment_pending', None)
            pending = None
        align_column, alignment_show_column = st.columns(2, gap='small', vertical_alignment='center')
        align_clicked = align_column.button('Align', key='workflow_science_align',
                                  disabled=bool(pattern_error) or len(sources) < 2 or not corrected or bool(pending))
        continue_clicked = False
        if pending:
            st.warning('O alinhamento atingiu 3 minutos. Queres continuar por mais 3 minutos ou parar? O progresso foi preservado.')
            continue_column, stop_column = st.columns(2)
            continue_clicked = continue_column.button('Continuar', key='workflow_science_alignment_continue')
            if stop_column.button('Parar', key='workflow_science_alignment_stop'):
                pending[1].cancel()
                st.session_state.pop('workflow_science_alignment_pending', None)
                st.session_state.pop('workflow_science_alignment_error', None)
                st.rerun(scope='app')
        if align_clicked or continue_clicked:
            try:
                with st.spinner('Aligning science frames…'):
                    if pending:
                        result, workspace = pending[1].resume(timeout=ALIGNMENT_TIMEOUT)
                    else:
                        result, workspace = run_with_timeout(
                            align_science_frames, list(corrected_files.values()), bias, flats, max_shift,
                            corrected=True, keep_workspace=True, timeout=ALIGNMENT_TIMEOUT, allow_continue=True)
                st.session_state.pop('workflow_science_alignment_pending', None)
                previous = st.session_state.get('workflow_science_alignment_workspace')
                st.session_state.workflow_science_alignment_workspace = workspace
                st.session_state.workflow_science_aligned = (alignment_signature, *result)
                st.session_state.pop('workflow_science_alignment_error', None)
                if previous:
                    previous.cleanup()
                st.rerun(scope='app')
            except ProcessingDecisionRequired as exc:
                st.session_state.workflow_science_alignment_pending = (alignment_signature, exc.job)
                st.session_state.pop('workflow_science_alignment_error', None)
                st.rerun(scope='app')
            except (ValueError, OSError, TypeError, KeyError, MemoryError) as exc:
                st.session_state.pop('workflow_science_alignment_pending', None)
                st.session_state.workflow_science_alignment_error = (alignment_signature, str(exc))
        alignment_error = st.session_state.get('workflow_science_alignment_error')
        if alignment_error and alignment_error[0] == alignment_signature:
            st.error(alignment_error[1])
        alignment = st.session_state.get('workflow_science_aligned')
        aligned = bool(not pending and alignment and alignment[0] == alignment_signature)
        if alignment_show_column.button('show', key='workflow_science_align_show',
                                        disabled=not aligned or bool(pattern_error)):
            show_science_pairs([(name, corrected_files[name], after)
                                for name, after in zip(labels, alignment[1])], 'alignment')
        if aligned:
            st.caption(f'{len(sources)} science frames aligned. Combination uses these frames.')
            with st.expander('align log'):
                st.code('\n'.join(f'{name}: {dy:+.2f}, {dx:+.2f}'
                                  for name, (dy, dx) in zip(labels, alignment[2])), language=None)
        work_sources = alignment[1] if aligned else []
        label, options = st.columns([2, 3], gap='small', vertical_alignment='center')
        label.markdown('<span style="display:flex;align-items:center;min-height:28px;font-size:12px;white-space:nowrap">Combination method:</span>',
                       unsafe_allow_html=True)
        method = options.radio('Combination method:', ['Mean', 'Median'],
                               label_visibility='collapsed', horizontal=True, key='workflow_science_method')
        combine_column, show_column, contrast_column = st.columns([1, 1, 1], gap='small', vertical_alignment='center')
        ready = bool(aligned and len(sources) >= 2 and bias is not None and flats and not pattern_error)
        retry = combine_column.button('Combine', key='workflow_science_combine', disabled=not ready)
        contrast_option(contrast_column, 'workflow_science_contrast_option')
        signature = repr(('science-v4', alignment_inputs,
                          alignment_signature if aligned else None, tuple(map(fingerprint, work_sources)), method))
        last = st.session_state.get('workflow_science_last')
        valid_results = bool(ready and last and last[0] == signature and last[1])
        if show_column.button('show', key='workflow_science_show', disabled=not valid_results):
            show_science(work_sources, last[1], last[2])
        if sources and (bias is None or not flats):
            st.info('Correct prepares the master bias and flats first using the selected calibration frames. Then use Align → Combine.')
        if ready and retry:
            st.session_state.workflow_science_attempt = signature
            st.session_state.pop('workflow_science_error', None)
            try:
                with st.spinner('Combining aligned science frames…'):
                    masters, inputs, skipped = run_with_timeout(
                        combine_science_groups, work_sources, bias, flats, method=method.lower(),
                        aligned=True, corrected=True)
                st.session_state.workflow_science_last = (signature, masters, inputs, skipped)
                preferred = source_filter(current)
                st.session_state.workflow_science_selected_group = next(
                    (key for key, image in masters.items() if image.header.get('FILTER') == preferred), next(iter(masters)))
                st.session_state.workflow_stage_view = 'science'
                st.rerun(scope='app')
            except (ValueError, OSError, TypeError, KeyError, MemoryError) as exc:
                st.session_state.workflow_science_error = (signature, str(exc) or 'Science images could not be combined.')
        error = st.session_state.get('workflow_science_error')
        if error and error[0] == signature:
            st.error(error[1])
        last = st.session_state.get('workflow_science_last')
        if ready and last and last[0] == signature:
            masters, inputs, skipped = last[1:]
            if masters:
                chosen = st.session_state.get('workflow_science_selected_group')
                if chosen not in masters:
                    chosen = next(iter(masters))
                    st.session_state.workflow_science_selected_group = chosen
                preview = masters[chosen]
                st.caption(f'{len(masters)} science masters created by {method.lower()} · corrected + aligned.')
                st.markdown('**Science results by filter and exposure**')
                for key, master in masters.items():
                    st.caption(f'{key} · {master.header.get("NCOMBINE", 1)} frame(s)')
                    with st.container(key=f'workflow_science_master_row_{key}'):
                        filename, view, save, remove = st.columns([7, 1, 1, 1], gap='small', vertical_alignment='center')
                        selected = ('selected-image' if st.session_state.get('workflow_stage_view') == 'science'
                                    and key == chosen else '')
                        name_clicked = filename_button(filename, master.name, f'science_master_{key}', selected)
                        view_clicked = view.button('', icon=':material/visibility:', help='View image',
                                                    key=f'workflow_science_view_{key}')
                        image_save_button(save, master, f'workflow_science_download_{key}',
                                          display_png=True, contrast=st.session_state.get('viewer_contrast', False))
                        remove_clicked = remove.button('×', help='Remove science master', key=f'workflow_science_master_remove_{key}')
                        if name_clicked or view_clicked:
                            st.session_state.workflow_science_selected_group = key
                            st.session_state.workflow_stage_view = 'science'
                            st.rerun(scope='app')
                        if remove_clicked:
                            st.session_state.workflow_science_last = (
                                signature, {name: image for name, image in masters.items() if name != key},
                                {name: selected for name, selected in inputs.items() if name != key}, skipped)
                            st.rerun(scope='app')
                if st.button('Download all corrected science results (ZIP)', key='workflow_science_download_all'):
                    archive_download_dialog(masters, 'science_masters', display_png=True,
                                            contrast=st.session_state.get('viewer_contrast', False))
            if skipped:
                st.caption('Groups with fewer than two images are shown below and are not combined.')
                for key, files in skipped:
                    st.markdown(f'<strong>{escape(key)} · {len(files)} frame(s) · not combined</strong>',
                                unsafe_allow_html=True)
    return preview
