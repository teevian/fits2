"""Visual selection of bias, flat and science frames using gallery previews."""
from base64 import b64encode
from pathlib import Path
import re

import streamlit as st
import streamlit.components.v1 as components

from gallery_ui import fits_thumbnail, frame_thumbnail
from processing_ui import load


image_picker = components.declare_component('science_image_picker',
                                            path=str(Path(__file__).with_name('science_picker_component')))


def preview_items(catalog, candidates):
    items = []
    for name in candidates:
        path = catalog[name]
        try:
            if isinstance(path, Path):
                stat = path.stat()
                preview = fits_thumbnail(path, stat.st_mtime_ns, stat.st_size)
            else:
                preview = frame_thumbnail(load(path))
            items.append({'name': name, 'image': 'data:image/png;base64,' + b64encode(preview).decode('ascii')})
        except (ValueError, OSError, TypeError, KeyError) as exc:
            items.append({'name': name, 'error': str(exc)})
    return items


@st.dialog('Choose images', width='large')
def choose_science_images(catalog, candidates, generation, kind='science'):
    st.subheader({'bias': 'Bias', 'flat': 'Flats', 'science': 'Science'}[kind])
    st.caption('Click to select or deselect. Drag across images to apply the same choice. Apply to use the selection.')
    if not candidates:
        st.info('No FITS files match the current regex.')
        return
    items = preview_items(catalog, candidates)
    frames_key = f'workflow_{kind}_frames'
    selected = st.session_state[frames_key]
    result = image_picker(items=items, selected=[name for name in candidates if name in selected],
                          key=f'{kind}_picker_{generation}', default=None)
    if result is not None:
        if result.get('action') == 'apply':
            chosen = set(result.get('selected', []))
            # Preserve any uploads or selections outside the current regex.
            visible = set(candidates)
            st.session_state[frames_key] = [
                name for name in catalog if (name in visible and name in chosen)
                or (name not in visible and name in selected)]
        st.rerun(scope='app')


def choose_frames_button(kind, catalog):
    pattern_key = f'workflow_{kind}_pattern'
    if st.button('choose', key=f'workflow_{kind}_choose',
                 disabled=bool(st.session_state.get(f'{pattern_key}_error'))):
        pattern = st.session_state[pattern_key].strip()
        expression = re.compile(pattern, re.IGNORECASE)
        candidates = [name for name, value in catalog.items()
                      if isinstance(value, (Path, tuple)) and pattern and expression.match(name)]
        generation_key = f'{kind}_choose_generation'
        generation = st.session_state.get(generation_key, 0) + 1
        st.session_state[generation_key] = generation
        choose_science_images(catalog, candidates, generation, kind)


def inline_bias_picker(catalog, on_view=None, kind="bias"):
    """Select only local regex matches; commit a click or drag immediately."""
    import hashlib
    from processing_ui import fingerprint
    pattern = st.session_state[f'workflow_{kind}_pattern'].strip()
    if st.session_state.get(f'workflow_{kind}_pattern_error'):
        return
    expression = re.compile(pattern, re.IGNORECASE)
    candidates = [name for name, value in catalog.items()
                  if pattern and isinstance(value, Path) and expression.match(name)]
    selected = [name for name in candidates if name in st.session_state[f'workflow_{kind}_frames']]
    st.session_state[f'workflow_{kind}_frames'] = selected
    if not candidates:
        st.info('No FITS files match the current regex.')
        return
    context = repr((pattern, [(name, fingerprint(catalog[name])) for name in candidates]))
    key = f'{kind}_inline_' + hashlib.sha256(context.encode()).hexdigest()[:16]
    result = image_picker(items=preview_items(catalog, candidates), selected=selected,
                          inline=True, key=key, default=None)
    if result and result.get('action') in ('change', 'view'):
        event = (key, result.get('event'))
        if st.session_state.get(f'{kind}_inline_last_event') != event:
            st.session_state[f'{kind}_inline_last_event'] = event
            if result.get('action') == 'view':
                name = result.get('name')
                if name in candidates and on_view:
                    on_view(catalog[name])
                return
            chosen = set(result.get('selected', []))
            updated = [name for name in candidates if name in chosen]
            if updated != selected:
                st.session_state[f'workflow_{kind}_frames'] = updated
                st.rerun(scope='app')


def result_previews(entries, key):
    """Clickable result variants, each rendered with its own display settings."""
    from imageProcessing import display_group, default_display_limits, resolve_display_limits
    items = []
    for index, (frame, label, contrast) in enumerate(entries):
        defaults = default_display_limits(display_group('.', frame.name, frame.header))
        limits = resolve_display_limits(frame.display(), defaults, contrast=contrast)
        preview = frame_thumbnail(frame, limits=limits)
        items.append({'name': str(index), 'label': label,
                      'image': 'data:image/png;base64,' + b64encode(preview).decode('ascii')})
    result = image_picker(items=items, selected=[], inline=True, preview_only=True,
                          key=key, default=None)
    if result and result.get('action') == 'view':
        event = result.get('event')
        if st.session_state.get(key + '_event') != event:
            st.session_state[key + '_event'] = event
            name = result.get('name')
            if name in {str(i) for i in range(len(entries))}:
                return int(name)
    return None
