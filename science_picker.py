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
    """Click or sweep to select; only a double-click requests a full image load."""
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
    previews_key = f'{kind}_inline_previews'
    cached = st.session_state.get(previews_key)
    if cached is None or cached[0] != key:
        cached = (key, preview_items(catalog, candidates))
        st.session_state[previews_key] = cached
    result = image_picker(items=cached[1], selected=selected,
                          inline=True, ack=st.session_state.get(f'{kind}_inline_last_event', (None, None))[1],
                          key=key, default=None)
    if result and result.get('action') in ('change', 'view'):
        event = (key, result.get('event'))
        if st.session_state.get(f'{kind}_inline_last_event') != event:
            st.session_state[f'{kind}_inline_last_event'] = event
            if 'selected' in result:
                chosen = set(result['selected'])
                st.session_state[f'workflow_{kind}_frames'] = [name for name in candidates if name in chosen]
            if result.get('action') == 'view':
                name = result.get('name')
                if name in candidates and on_view:
                    on_view(catalog[name])


@st.cache_data(max_entries=128, show_spinner=False)
def result_thumbnail(_frame, signature, contrast):
    """Reuse variant thumbnails while selection changes in the sidebar."""
    from imageProcessing import display_group, default_display_limits, resolve_display_limits
    from color_composition import ColorComposite
    if isinstance(_frame, ColorComposite):
        return _frame.png(contrast, thumbnail=True)
    defaults = default_display_limits(display_group('.', _frame.name, _frame.header))
    limits = resolve_display_limits(_frame.display(), defaults, contrast=contrast)
    return frame_thumbnail(_frame, limits=limits)


def result_previews(entries, key):
    """Result variants open in the main viewer only on a double-click."""
    from processing_ui import fingerprint
    items = []
    for index, (frame, label, contrast) in enumerate(entries):
        preview = result_thumbnail(frame, fingerprint(frame), contrast)
        items.append({'name': str(index), 'label': label,
                      'image': 'data:image/png;base64,' + b64encode(preview).decode('ascii')})
    result = image_picker(items=items, selected=[], inline=True, preview_only=True, ack=st.session_state.get(key + '_event'),
                          key=key, default=None)
    if result and result.get('action') == 'view':
        event = result.get('event')
        if st.session_state.get(key + '_event') != event:
            st.session_state[key + '_event'] = event
            name = result.get('name')
            if name in {str(i) for i in range(len(entries))}:
                return int(name)
    return None


@st.cache_data(max_entries=256, show_spinner=False)
def science_pair_thumbnails(_before, _after, signature):
    """Use one contrast scale for the original and corrected preview of a frame."""
    import numpy as np
    from calibration_gallery import display_sample
    images = [load(_before), load(_after)]
    samples = np.concatenate([display_sample(image) for image in images])
    low, high = np.percentile(samples, [1, 99]) if samples.size else (0., 1.)
    if high <= low:
        low, high = low - max(abs(low) * .05, 1.), high + max(abs(high) * .05, 1.)
    return [(image.name, frame_thumbnail(image, (float(low), float(high)))) for image in images]


def science_pairs_grid(pairs, on_view):
    """Show original/corrected pairs in a scrollable grid with double-click viewing."""
    import hashlib
    from processing_ui import fingerprint
    signature = tuple((name, fingerprint(before), fingerprint(after)) for name, before, after in pairs)
    key = 'science_pairs_' + hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    cached = st.session_state.get('science_pairs_previews')
    if cached is None or cached[0] != key:
        items = []
        for index, (name, before, after) in enumerate(pairs):
            try:
                previews = science_pair_thumbnails(before, after, signature[index])
                for side, (filename, thumbnail) in enumerate(previews):
                    items.append({'name': f'{index}:{side}', 'label': filename, 'selectable': side == 1,
                                  'image': 'data:image/png;base64,' + b64encode(thumbnail).decode('ascii')})
            except (ValueError, OSError, TypeError, KeyError) as exc:
                for side in (0, 1):
                    items.append({'name': f'{index}:{side}', 'label': name, 'error': str(exc)})
        cached = (key, items)
        st.session_state.science_pairs_previews = cached
    selection_key = 'workflow_science_corrected_selection'
    if st.session_state.get('science_pairs_selection_generation') != key:
        st.session_state.science_pairs_selection_generation = key
        st.session_state[selection_key] = [name for name, _, _ in pairs]
    selected = st.session_state[selection_key]
    result = image_picker(items=cached[1],
                          selected=[f'{i}:1' for i, (name, _, _) in enumerate(pairs) if name in selected],
                          inline=True, scrollable=True,
                          ack=st.session_state.get(key + '_event'), key=key, default=None)
    if result and result.get('event') != st.session_state.get(key + '_event'):
        st.session_state[key + '_event'] = result.get('event')
        if 'selected' in result:
            chosen = set(result['selected'])
            st.session_state[selection_key] = [name for i, (name, _, _) in enumerate(pairs)
                                               if f'{i}:1' in chosen]
        if result.get('action') == 'view':
            sources = {f'{index}:{side}': value for index, (_, before, after) in enumerate(pairs)
                       for side, value in enumerate((before, after))}
            if result.get('name') in sources:
                on_view(sources[result['name']])
    return st.session_state[selection_key]
