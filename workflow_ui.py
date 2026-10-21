"""One-run sidebar controls for the sequential CCD workflow."""
from html import escape
from pathlib import Path
import re

import streamlit as st

from processing_ui import fingerprint, list_fits, load
from frame_selection import initialize_matching_frames, select_matching_frames
from science_ui import science_panel
from science_picker import inline_bias_picker, result_previews
from calibration_gallery import show_bias, show_flats
from download_ui import calibration_download_dialog, image_save_button
from imageProcessing import flat_archive, flat_channel
from processing_jobs import run_with_timeout, CALIBRATION_TIMEOUT
from imageProcessing import combine_flats_job
from imageProcessing import (combine_bias_frames,
                      flat_exposure_groups,
                      source_filter, _source_header)


def source_name(value):
    return value[0] if isinstance(value, tuple) else value.name


def automatic_frame_selection(files):
    """Identify original calibration and science images from headers and filenames."""
    groups = {'bias': [], 'flat': [], 'science': []}
    for path in files:
        name = path.name.casefold()
        if name.startswith(('master_', 'corrected_', 'aligned_')):
            continue
        header = _source_header(path)
        kind = str(header.get('IMAGETYP', '')).upper()
        if (header.get('NAXIS') != 2 or
                str(header.get('XTENSION', '')).strip() in ('TABLE', 'BINTABLE') or
                any(header.get(key) is True for key in ('BIASCOR', 'FLATCOR', 'SCICOR')) or
                kind.startswith(('MASTER', 'CORRECTED', 'ALIGNED'))):
            continue
        if 'BIAS' in kind or 'ZERO' in kind or name.startswith(('bias', 'zero')):
            groups['bias'].append(path.name)
        elif 'FLAT' in kind or name.startswith('flat'):
            groups['flat'].append(path.name)
        elif 'DARK' not in kind and not name.startswith('dark'):
            groups['science'].append(path.name)
    if len(groups['bias']) < 2:
        raise ValueError('Auto requires at least two original bias frames in the working folder.')
    if len(groups['flat']) < 2:
        raise ValueError('Auto requires at least two original flat frames per filter in the working folder.')
    if not groups['science']:
        raise ValueError('Auto found no original science images in the working folder.')
    return groups


def workflow_panel(current, source_id, folder, on_view_change=None, auto=False):
    st.session_state.setdefault('viewer_contrast', False)
    if st.session_state.get('calibration_recipe') != 'guide-median-v2':
        for kind in ('bias', 'flat', 'science'):
            st.session_state[f'workflow_{kind}_method'] = 'Median'
        st.session_state.calibration_recipe = 'guide-median-v2'

    def toggle_contrast():
        st.session_state.viewer_contrast = not st.session_state.viewer_contrast
        if on_view_change:
            on_view_change()

    def contrast_option(column, key):
        column.button('Contrast', key=key, on_click=toggle_contrast,
                      help='Display only: 1st–99th percentile contrast. Flat pairs share the same scale.')
        highlight = 'var(--selection-background)' if st.session_state.viewer_contrast else 'transparent'
        st.markdown(f'<style>.st-key-{key} button '
                    f'{{background:{highlight};border:none;box-shadow:none;outline:none;'
                    'border-radius:0;padding:4px 10px;min-height:38px;}</style>', unsafe_allow_html=True)

    catalog = {}
    try:
        catalog.update({path.name: path for path in list_fits(folder)})
    except OSError:
        pass
    if auto:
        try:
            groups = automatic_frame_selection(catalog.values())
            for kind, labels in groups.items():
                st.session_state[f'workflow_{kind}_frames'] = labels
                st.session_state[f'workflow_{kind}_frames_folder'] = str(folder)
                st.session_state[f'workflow_{kind}_pattern'] = '|'.join(
                    re.escape(label) + '$' for label in labels)
                st.session_state.pop(f'workflow_{kind}_pattern_error', None)
            st.session_state.workflow_flat_subtract_bias = True
        except (ValueError, OSError, TypeError, KeyError) as exc:
            st.error(str(exc))
            auto = False
    catalog.update({f'✓ {name}': value for name, value in
                    st.session_state.get('reduction_products', {}).items()})

    source = lambda value: load(value) if isinstance(value, tuple) else value

    def view_file(value):
        st.session_state.workflow_file_view = (source_id, load(value))
        st.session_state.workflow_stage_view = 'file'
        st.rerun(scope='app')

    def filename_button(column, name, key, selected):
        column.markdown(f'<span class="image-filename {selected}" style="display:none">'
                        f'{escape(name)}</span>', unsafe_allow_html=True)
        return column.button(name, key=f'workflow_image_name_{key}', use_container_width=True)

    initialize_matching_frames('workflow_bias_pattern', 'workflow_bias_frames',
                               'bias*', catalog, folder)
    initialize_matching_frames('workflow_flat_pattern', 'workflow_flat_frames',
                               'flat*', catalog, folder)

    # Downloads are prepared on the save click, rather than for every listed frame.
    st.session_state.pop('workflow_source_downloads', None)

    with st.expander('1. Master bias', expanded=False):
        st.markdown("""<style>
    .st-key-workflow_bias_combine button {
        border: 1px solid #7f91af; background: rgba(127,145,175,.16);
        padding: 4px 14px; cursor: pointer;}
    .st-key-workflow_bias_combine button:hover {
        background: rgba(127,145,175,.32); border-color: #a7b8d3;}
    .st-key-workflow_bias_pattern [data-baseweb="input"],
    .st-key-workflow_bias_shift [data-baseweb="input"] {
        border: none; border-bottom: 1px solid #7f91af; background: transparent; box-shadow: none;}
    .st-key-workflow_bias_pattern input, .st-key-workflow_bias_shift input {
        padding: 2px 6px; min-height: 28px;}
    </style>""", unsafe_allow_html=True)
        bias_mode = 'Create master from frames'
        label, field = st.columns([1, 3], gap='small', vertical_alignment='center')
        label.markdown('Bias files:')
        field.text_input('Bias files:', key='workflow_bias_pattern', label_visibility='collapsed',
                         placeholder='bias*', on_change=select_matching_frames,
                         args=('workflow_bias_pattern', 'workflow_bias_frames', catalog))
        if st.session_state.get('workflow_bias_pattern_error'):
            st.error(st.session_state.workflow_bias_pattern_error)
        st.markdown("""<style>
    :root {--selection-background: rgba(127, 145, 175, .22);}
    *, *::before, *::after {border-radius: 0 !important;}
    .st-key-workflow_bias_list [data-testid="stVerticalBlock"] {gap: 2px !important;}
    [class*="st-key-workflow_bias_row_"]:has(.selected-image),
    .st-key-workflow_bias_master_row:has(.selected-image) {background: var(--selection-background);}
    [class*="st-key-workflow_bias_row_"] [data-testid="stMarkdownContainer"] p,
    .st-key-workflow_bias_master_row [data-testid="stMarkdownContainer"] p {margin: 0;}
    .image-filename {display: flex; align-items: center; min-height: 28px;
        padding: 0 6px; font-size: 11px; line-height: 16px; overflow-wrap: anywhere;}
    [data-testid="stMarkdownContainer"] p:has(.image-filename) {display:none;}
    [class*="st-key-workflow_image_name_"] button {
        opacity:1 !important; pointer-events:auto !important; height:auto !important;
        justify-content:flex-start !important; padding:0 6px !important;
        background:transparent; border:none; outline:none; box-shadow:none;}
    [class*="st-key-workflow_image_name_"] button p {
        font-size:11px; line-height:16px; text-align:left; overflow-wrap:anywhere;}
    [class*="st-key-workflow_bias_row_"] button, .st-key-workflow_bias_master_row button {
        opacity: 0; pointer-events: none; height: 28px; min-height: 28px; padding: 0 4px;
        display: flex; align-items: center; justify-content: center;
        border: none; outline: none; box-shadow: none;}
    [class*="st-key-workflow_bias_row_"]:hover button,
    [class*="st-key-workflow_bias_row_"]:focus-within button,
    .st-key-workflow_bias_master_row:hover button,
    .st-key-workflow_bias_master_row:focus-within button {opacity: 1; pointer-events: auto;}
    .st-key-workflow_bias_method [data-baseweb="radio"] > div:first-child {display: none;}
    .st-key-workflow_bias_method [data-baseweb="radio"] {
        display: flex; align-items: center; justify-content: center;
        min-height: 28px; padding: 0 8px; margin: 0;}
    .st-key-workflow_bias_method [data-baseweb="radio"] > div {margin: 0; padding: 0;}
    .st-key-workflow_bias_method [data-baseweb="radio"] p {margin: 0; line-height: 28px;}
    [data-baseweb="radio"]:has(input:checked),
    [role="tab"][aria-selected="true"], [role="option"][aria-selected="true"],
    [data-baseweb="tag"], [data-baseweb="checkbox"]:has(input:checked) {
        background: var(--selection-background);}
    </style>""", unsafe_allow_html=True)
        inline_bias_picker(catalog, view_file)
        bias_sources = [catalog[label] for label in st.session_state.workflow_bias_frames
                        if label in catalog]
        bias_master = None
        bias_inputs = [source(value) for value in bias_sources]
        label, options = st.columns([2, 3], gap='small', vertical_alignment='center')
        label.markdown('<span style="display:flex;align-items:center;min-height:28px;font-size:12px;white-space:nowrap">Combination method:</span>',
                       unsafe_allow_html=True)
        bias_method = options.radio('Combination method:', ['Mean', 'Median'],
                                    label_visibility='collapsed', horizontal=True,
                                    key='workflow_bias_method')
        bias_signature = repr(('guide-v1', str(folder), bias_mode, bias_method,
                               tuple(map(fingerprint, bias_sources))))
        def save_master_bias(master):
            number = st.session_state.get('workflow_bias_product_number', 0) + 1
            st.session_state.workflow_bias_product_number = number
            master.name = f'master_bias_{number}_x{master.header["NCOMBINE"]}.fits'
            st.session_state.workflow_bias_last = (bias_signature, master)

        bias_last = st.session_state.get('workflow_bias_last')
        if bias_mode == 'Create master from frames':
            combine_column, show_column = st.columns([1.2, 1], gap='small', vertical_alignment='center')
            bias_combine_clicked = combine_column.button('Combine', key='workflow_bias_combine',
                                                         disabled=len(bias_sources) < 2 or bool(st.session_state.get('workflow_bias_pattern_error')))
            if show_column.button('show', key='workflow_bias_show',
                                  disabled=not (bias_last and bias_last[0] == bias_signature)):
                show_bias(bias_sources, bias_last[1])
            if bias_combine_clicked or auto:
                try:
                    with st.spinner('Combining bias frames…'):
                        master = run_with_timeout(combine_bias_frames, bias_inputs, bias_method.lower())
                    save_master_bias(master)
                    bias_last = st.session_state.workflow_bias_last
                    if not auto:
                        st.session_state.workflow_stage_view = 'bias'
                        st.rerun()
                except (ValueError, OSError, TypeError, KeyError, MemoryError) as exc:
                    st.error(str(exc) or 'The bias frames could not be combined.')
                    auto = False
            if bias_last and bias_last[0] == bias_signature:
                master = bias_last[1]
                st.caption(f'{master.header["NCOMBINE"]} bias frames combined by {bias_method.lower()}.')
                choice = result_previews([(master, 'Master bias', False),
                                          (master, 'Contrast', True)],
                                         key=f'bias_result_{master.name}')
                if choice is not None:
                    st.session_state.viewer_contrast = bool(choice)
                    st.session_state.workflow_stage_view = 'bias'
                    if on_view_change:
                        on_view_change()
                    st.rerun()
                with st.container(key='workflow_bias_master_row'):
                    filename, view, save, remove = st.columns([7, 1, 1, 1], gap='small', vertical_alignment='center')
                    selected = 'selected-image' if st.session_state.get('workflow_stage_view') == 'bias' else ''
                    name_clicked = filename_button(filename, master.name, 'bias_master', selected)
                    view_clicked = view.button('', icon=':material/visibility:', help='View image',
                                                key='workflow_bias_view')
                    if name_clicked or view_clicked:
                        st.session_state.workflow_stage_view = 'bias'
                        st.rerun()
                    image_save_button(save, master, 'workflow_bias_download')
                    if remove.button('×', help='Remove master bias', key='workflow_bias_master_remove'):
                        st.session_state.pop('workflow_bias_last', None)
                        st.session_state.pop('workflow_flat_last', None)
                        st.session_state.pop('workflow_flat_versions', None)
                        if st.session_state.get('workflow_stage_view') == 'bias':
                            st.session_state.workflow_stage_view = None
                        st.rerun()
                if st.button('Download master bias (ZIP)', key='workflow_bias_download_all'):
                    calibration_download_dialog({False: {'Bias': master}}, 'master_bias')

    with st.expander('2. Flats', expanded=True):
        st.markdown("""<style>
    .st-key-workflow_flat_combine button {
        border: 1px solid #7f91af; background: rgba(127,145,175,.16);
        padding: 4px 14px; cursor: pointer;}
    .st-key-workflow_flat_combine button:hover {
        background: rgba(127,145,175,.32); border-color: #a7b8d3;}
    .st-key-workflow_flat_pattern [data-baseweb="input"],
    .st-key-workflow_flat_shift [data-baseweb="input"] {
        border: none; border-bottom: 1px solid #7f91af; background: transparent; box-shadow: none;}
    .st-key-workflow_flat_pattern input, .st-key-workflow_flat_shift input {
        padding: 2px 6px; min-height: 28px;}
    </style>""", unsafe_allow_html=True)
        flat_mode = 'Create master from frames'
        label, field = st.columns([1, 3], gap='small', vertical_alignment='center')
        label.markdown('Flat files:')
        field.text_input('Flat files:', key='workflow_flat_pattern', label_visibility='collapsed',
                         placeholder='flat*', on_change=select_matching_frames,
                         args=('workflow_flat_pattern', 'workflow_flat_frames', catalog))
        if st.session_state.get('workflow_flat_pattern_error'):
            st.error(st.session_state.workflow_flat_pattern_error)
        st.markdown("""<style>
    :root {--selection-background: rgba(127, 145, 175, .22);}
    *, *::before, *::after {border-radius: 0 !important;}
    .st-key-workflow_flat_list [data-testid="stVerticalBlock"] {gap: 2px !important;}
    [class*="st-key-workflow_flat_row_"]:has(.selected-image),
    [class*="st-key-workflow_flat_master_row_"]:has(.selected-image) {background: var(--selection-background);}
    [class*="st-key-workflow_flat_row_"] [data-testid="stMarkdownContainer"] p,
    [class*="st-key-workflow_flat_master_row_"] [data-testid="stMarkdownContainer"] p {margin: 0;}
    .image-filename {display: flex; align-items: center; min-height: 28px;
        padding: 0 6px; font-size: 11px; line-height: 16px; overflow-wrap: anywhere;}
    [class*="st-key-workflow_flat_row_"] button, [class*="st-key-workflow_flat_master_row_"] button {
        opacity: 0; pointer-events: none; height: 28px; min-height: 28px; padding: 0 4px;
        display: flex; align-items: center; justify-content: center;
        border: none; outline: none; box-shadow: none;}
    [class*="st-key-workflow_flat_row_"]:hover button,
    [class*="st-key-workflow_flat_row_"]:focus-within button,
    [class*="st-key-workflow_flat_master_row_"]:hover button,
    [class*="st-key-workflow_flat_master_row_"]:focus-within button {opacity: 1; pointer-events: auto;}
    .st-key-workflow_flat_method [data-baseweb="radio"] > div:first-child {display: none;}
    .st-key-workflow_flat_method [data-baseweb="radio"] {
        display: flex; align-items: center; justify-content: center;
        min-height: 28px; padding: 0 8px; margin: 0;}
    .st-key-workflow_flat_method [data-baseweb="radio"] > div {margin: 0; padding: 0;}
    .st-key-workflow_flat_method [data-baseweb="radio"] p {margin: 0; line-height: 28px;}
    [data-baseweb="radio"]:has(input:checked),
    [role="tab"][aria-selected="true"], [role="option"][aria-selected="true"],
    [data-baseweb="tag"], [data-baseweb="checkbox"]:has(input:checked) {
        background: var(--selection-background);}
    </style>""", unsafe_allow_html=True)
        inline_bias_picker(catalog, view_file, kind='flat')
        flat_sources = [catalog[label] for label in st.session_state.workflow_flat_frames
                        if label in catalog]
        flat_input_signature = repr((str(folder), tuple(map(fingerprint, flat_sources))))
        flat_work_inputs = [source(value) for value in flat_sources]
        label, options = st.columns([2, 3], gap='small', vertical_alignment='center')
        label.markdown('<span style="display:flex;align-items:center;min-height:28px;font-size:12px;white-space:nowrap">Combination method:</span>',
                       unsafe_allow_html=True)
        flat_method = options.radio('Combination method:', ['Mean', 'Median'],
                                    label_visibility='collapsed', horizontal=True,
                                    key='workflow_flat_method')
        def toggle_flat_bias():
            st.session_state.workflow_flat_subtract_bias = not st.session_state.workflow_flat_subtract_bias
            if st.session_state.get('workflow_flat_versions'):
                st.session_state.workflow_stage_view = 'flat'
                if on_view_change:
                    on_view_change()

        subtract_flat_bias = st.session_state.setdefault('workflow_flat_subtract_bias', True)
        combine_column, show_column, bias_option = st.columns([1.2, .8, .9], gap='small', vertical_alignment='center')
        combine_clicked = combine_column.button('Combine', key='workflow_flat_combine',
                                                 disabled=len(flat_sources) < 2 or bool(st.session_state.get('workflow_flat_pattern_error')))
        bias_option.button('-bias', key='workflow_flat_bias_option', on_click=toggle_flat_bias,
                           help='On: bias-corrected, individually normalized flats for science. Off: raw combined flats for inspection.')
        highlight = 'var(--selection-background)' if subtract_flat_bias else 'transparent'
        st.markdown(f'<style>.st-key-workflow_flat_bias_option button '
                    f'{{background:{highlight};border:none;box-shadow:none;outline:none;'
                    'padding:4px 10px;min-height:38px;}</style>', unsafe_allow_html=True)
        flat_signature = repr(('groups-guide-v1', str(folder), flat_mode, flat_method,
                               tuple(map(fingerprint, flat_sources)),
                               bias_signature, fingerprint(bias_master)))

        def save_flat_versions(versions, names, archives):
            st.session_state.workflow_flat_versions = (flat_signature, versions, names, archives)
            st.session_state.workflow_flat_last = (
                flat_signature, versions[subtract_flat_bias], archives[subtract_flat_bias], names)

        flat_versions = st.session_state.get('workflow_flat_versions')
        if flat_versions and flat_versions[0] == flat_signature:
            save_flat_versions(*flat_versions[1:])
        flat_last = st.session_state.get('workflow_flat_last')
        if show_column.button('show', key='workflow_flat_show',
                              disabled=not (flat_last and flat_last[0] == flat_signature)):
            show_flats(flat_sources, flat_last[1], flat_last[3])
        flat_preview = None
        if flat_mode == 'Create master from frames':
            try:
                cached_groups = st.session_state.get('workflow_flat_groups')
                if cached_groups and cached_groups[0] == flat_input_signature:
                    exposure_groups = cached_groups[1]
                else:
                    exposure_groups = flat_exposure_groups(flat_sources) if flat_sources else {}
                    st.session_state.workflow_flat_groups = (flat_input_signature, exposure_groups)
            except (ValueError, OSError, TypeError, KeyError) as exc:
                st.error(str(exc))
                exposure_groups = {}
            skipped = [(filter_name, exposure, selected)
                       for filter_name, exposures in exposure_groups.items()
                       for exposure, selected in exposures.items() if len(selected) < 2]
            if combine_clicked or auto:
                try:
                    existing_bias = bias_last[1] if bias_last and bias_last[0] == bias_signature else None
                    if existing_bias is None and len(bias_sources) < 2:
                        raise ValueError('Select at least two bias frames before combining flats.')
                    with st.spinner('Combining flat frames…'):
                        master_bias, versions, flat_inputs, archives = run_with_timeout(
                            combine_flats_job, flat_work_inputs, bias_inputs,
                            bias_method.lower(), flat_method.lower(), master_bias=existing_bias,
                            timeout=CALIBRATION_TIMEOUT)
                    if existing_bias is None:
                        save_master_bias(master_bias)
                    save_flat_versions(versions, flat_inputs, archives)
                    flat_last = st.session_state.workflow_flat_last
                    masters = versions[subtract_flat_bias]
                    current_filter = source_filter(current)
                    st.session_state.workflow_flat_preferred_filter = current_filter
                    st.session_state.workflow_flat_selected_filter = next(
                        (key for key, master in masters.items()
                         if master.header.get('FILTER') == current_filter), next(iter(masters)))
                    if not auto:
                        st.session_state.workflow_stage_view = 'flat'
                        st.rerun()
                except (ValueError, OSError, TypeError, KeyError, MemoryError) as exc:
                    st.error(str(exc) or 'The flat frames could not be combined.')
                    auto = False
            if flat_last and flat_last[0] == flat_signature:
                masters = flat_last[1]
                if st.session_state.get('workflow_flat_selected_filter') not in masters:
                    preferred = st.session_state.get('workflow_flat_preferred_filter')
                    st.session_state.workflow_flat_selected_filter = next(
                        (key for key, master in masters.items()
                         if master.header.get('FILTER') == preferred), next(iter(masters)))
                flat_preview = masters[st.session_state.workflow_flat_selected_filter]
                flat_kind = 'normalized corrected flats' if subtract_flat_bias else 'master flats'
                st.caption(f'{len(masters)} {flat_kind} created by {flat_method.lower()}.')
                st.caption('Color previews identify filters only. Science calibration uses normalized FITS flats; '
                           'the final RGB image uses calibrated science masters.')
                for filter_name, master in masters.items():
                    exposure = master.header.get('EXPTIME', master.header.get('EXPOSURE', 'unknown'))
                    st.markdown(f"**{master.header.get('FILTER', filter_name)} · {exposure} s**")
                    variants = st.session_state.workflow_flat_versions[1]
                    entries = []
                    for mode, label in [(False, 'Without bias subtraction'),
                                        (True, 'Bias subtracted · normalized')]:
                        variant = variants[mode][filter_name]
                        entries.extend([(variant, label, False),
                                        (variant, label + ' · Contrast', True),
                                        (variant, label + f' · Channel {flat_channel(variant)} · Contrast', True, True)])
                    choice = result_previews(entries, key=f'flat_result_{filter_name}_{flat_signature}', columns=3)
                    if choice is not None:
                        st.session_state.workflow_flat_subtract_bias = choice >= 3
                        st.session_state.viewer_contrast = bool(choice % 3)
                        st.session_state.workflow_flat_selected_filter = filter_name
                        st.session_state.workflow_stage_view = 'flat'
                        if on_view_change:
                            on_view_change()
                        st.rerun()
                    with st.container(key=f'workflow_flat_master_row_{filter_name}'):
                        filename, view, save, remove = st.columns(
                            [7, 1, 1, 1], gap='small', vertical_alignment='center')
                        selected = ('selected-image' if
                                    st.session_state.get('workflow_stage_view') == 'flat' and
                                    st.session_state.workflow_flat_selected_filter == filter_name else '')
                        name_clicked = filename_button(filename, master.name, f'flat_master_{filter_name}', selected)
                        view_clicked = view.button('', icon=':material/visibility:', help='View image',
                                                    key=f'workflow_flat_view_{filter_name}')
                        if name_clicked or view_clicked:
                            st.session_state.workflow_flat_selected_filter = filter_name
                            st.session_state.workflow_stage_view = 'flat'
                            st.rerun()
                        image_save_button(save, master, f'workflow_flat_download_{filter_name}')
                        if remove.button('×', help='Remove master flat',
                                         key=f'workflow_flat_master_remove_{filter_name}'):
                            remaining = {key: value for key, value in masters.items()
                                         if key != filter_name}
                            if remaining:
                                names = {key: flat_last[3][key] for key in remaining}
                                cached = st.session_state.workflow_flat_versions
                                versions = {mode: {key: value for key, value in images.items()
                                                   if key != filter_name}
                                            for mode, images in cached[1].items()}
                                archives = {mode: flat_archive(images) for mode, images in versions.items()}
                                save_flat_versions(versions, names, archives)
                                st.session_state.workflow_flat_selected_filter = next(iter(remaining))
                            else:
                                st.session_state.pop('workflow_flat_last', None)
                                st.session_state.pop('workflow_flat_versions', None)
                                if st.session_state.get('workflow_stage_view') == 'flat':
                                    st.session_state.workflow_stage_view = None
                            st.rerun()
                if st.button('Download all flats (ZIP)', key='workflow_flat_download_all'):
                    calibration_download_dialog(st.session_state.workflow_flat_versions[1], 'flats')
            if skipped:
                st.caption('Groups with fewer than two images are shown below and are not combined.')
            for index, (filter_name, exposure, selected) in enumerate(skipped):
                exposure_text = f'{exposure:g} s' if exposure is not None else 'unknown exposure'
                st.markdown(f'<strong>{escape(str(filter_name))} · {escape(exposure_text)} · '
                            'not combined</strong>',
                            unsafe_allow_html=True)

    science_bias = bias_last[1] if bias_last and bias_last[0] == bias_signature else None
    science_versions = st.session_state.get('workflow_flat_versions')
    science_flats = (science_versions[1][True] if science_versions and
                     science_versions[0] == flat_signature else {})
    def prepare_science_calibration():
        master = science_bias
        if not science_flats:
            if master is None and len(bias_inputs) < 2:
                raise ValueError('Select at least two bias frames to prepare the science correction.')
            if len(flat_work_inputs) < 2:
                raise ValueError('Select at least two flat frames per filter to prepare the science correction.')
            master, versions, names, archives = run_with_timeout(
                combine_flats_job, flat_work_inputs, bias_inputs,
                bias_method.lower(), flat_method.lower(), master_bias=master,
                timeout=CALIBRATION_TIMEOUT)
            if science_bias is None:
                save_master_bias(master)
            save_flat_versions(versions, names, archives)
            return master, versions[True]
        if master is None:
            master = run_with_timeout(combine_bias_frames, bias_inputs, bias_method.lower())
            save_master_bias(master)
        return master, science_flats

    science_preview = science_panel(current, source_id, folder, catalog,
                                    science_bias, science_flats, contrast_option,
                                    prepare_calibration=prepare_science_calibration, auto_correct=auto)

    stage = st.session_state.get('workflow_stage_view')
    stage_result = None
    if stage == 'bias' and bias_last and bias_last[0] == bias_signature:
        stage_result = bias_last[1]
    elif stage == 'flat' and flat_last and flat_last[0] == flat_signature:
        stage_result = flat_preview
    elif stage in ('science', 'color'):
        stage_result = science_preview
    elif stage == 'file':
        file_view = st.session_state.get('workflow_file_view')
        if file_view and file_view[0] == source_id:
            stage_result = file_view[1]

    return stage_result
