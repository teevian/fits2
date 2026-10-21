"""Click selection on a display-only thumbnail, retaining original FITS coordinates."""
from base64 import b64encode
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

star_picker = components.declare_component('photometry_star_picker',
    path=str(Path(__file__).with_name('photometry_picker_component')))


@st.cache_data(max_entries=24, show_spinner=False)
def picker_preview(_image, signature):
    data = _image.display()
    valid = data[np.isfinite(data)]
    if not valid.size:
        raise ValueError('Imagem sem píxeis válidos.')
    low, high = np.percentile(valid, [10, 99.8])
    if high <= low:
        high = low + 1
    scaled = np.arcsinh(10*np.clip((data-low)/(high-low), 0, 1))/np.arcsinh(10)
    pixels = np.uint8(np.nan_to_num(scaled)*255)
    preview = Image.fromarray(pixels)
    preview.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    output = BytesIO()
    preview.save(output, format='PNG')
    return 'data:image/png;base64,' + b64encode(output.getvalue()).decode('ascii')


def selection_event(event, shape):
    """Validate browser events before allowing them to modify measurement centres."""
    if not isinstance(event, dict) or event.get('role') not in ('target', 'reference') or not isinstance(event.get('event'), str):
        return None
    try:
        x, y = float(event['x']), float(event['y'])
    except (KeyError, ValueError, TypeError):
        return None
    if not np.isfinite([x, y]).all() or not (0 <= x <= shape[1]-1 and 0 <= y <= shape[0]-1):
        return None
    return event['role'], x, y


def select_stars(image, signature, radius, inner, outer, allow_reference=True):
    prefix = 'photo_stars_' + signature
    values = image.display()
    y, x = np.unravel_index(np.nanargmax(values), values.shape)
    for key, value in [('x', float(x)), ('y', float(y)), ('ref_x', None), ('ref_y', None)]:
        if prefix+key not in st.session_state:
            st.session_state[prefix+key] = value
    event = star_picker(image=picker_preview(image, signature), width=values.shape[1], height=values.shape[0],
                        target=dict(x=st.session_state[prefix+'x'], y=st.session_state[prefix+'y']),
                        reference=dict(x=st.session_state[prefix+'ref_x'], y=st.session_state[prefix+'ref_y']),
                        radius=radius, inner=inner, outer=outer, allow_reference=allow_reference,
                        ack=st.session_state.get(prefix+'event'),
                        key=prefix+'picker', default=None)
    parsed = selection_event(event, values.shape)
    if parsed and event['event'] != st.session_state.get(prefix+'event'):
        role, x, y = parsed
        if role == 'target' or allow_reference:
            stem = '' if role == 'target' else 'ref_'
            st.session_state[prefix+stem+'x'] = x
            st.session_state[prefix+stem+'y'] = y
        st.session_state[prefix+'event'] = event['event']
    positions = {}
    columns = st.columns(4 if allow_reference else 2)
    for index, (stem, label) in enumerate([('', 'Alvo'), ('ref_', 'Referência')] if allow_reference else [('', 'Alvo')]):
        a, b = columns[index*2:index*2+2]
        positions[stem+'x'] = a.number_input(label+' · x [pixel]', min_value=0., max_value=float(values.shape[1]-1),
                                             value=None, key=prefix+stem+'x')
        positions[stem+'y'] = b.number_input(label+' · y [pixel]', min_value=0., max_value=float(values.shape[0]-1),
                                             value=None, key=prefix+stem+'y')
    return positions
