"""A separate, cached thumbnail gallery for the working folder's FITS images."""
from io import BytesIO
import re

import numpy as np
from PIL import Image
import streamlit as st

from imageProcessing import read_frame, display_group, default_display_limits, resolve_display_limits
from processing_ui import list_fits


@st.cache_data(max_entries=512, show_spinner=False)
def fits_thumbnail(path, modified, size):
    """Cache compact previews by file revision, without retaining full FITS arrays."""
    image = read_frame(path)
    return frame_thumbnail(image)


def frame_thumbnail(image, limits=None):
    """Render an in-memory image, optionally using a shared comparison scale."""
    values = image.display()
    if limits is None:
        defaults = default_display_limits(display_group('.', image.name, image.header))
        limits = resolve_display_limits(values, defaults, contrast=True)
    low, high = limits
    scaled = np.clip((values - low) / (high - low), 0., 1.)
    pixels = np.flipud(np.nan_to_num(scaled, nan=0.) * 255).astype(np.uint8)
    preview = Image.fromarray(pixels)
    preview.thumbnail((480, 320), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (480, 320))
    canvas.paste(preview, ((480 - preview.width) // 2, (320 - preview.height) // 2))
    output = BytesIO()
    canvas.save(output, format='PNG')
    return output.getvalue()


def gallery_page(folder):
    st.markdown('''<style>
    [data-testid="stSidebar"] {display:none !important;}
    .block-container {max-width:1600px; padding-top:1.5rem;}
    [data-testid="stImage"] {background:#000;}
    [data-testid="stCaptionContainer"] {overflow-wrap:anywhere;}
    </style>''', unsafe_allow_html=True)
    st.title('All FITS images')
    st.caption(str(folder))
    pattern = st.text_input('Filter files (regex)', key='gallery_pattern',
                            placeholder='e.g. deneb.*Blue or ^flat',
                            help='Filter filenames, ignoring case. Leave empty to show all images.')
    try:
        files = list_fits(folder)
    except OSError as exc:
        st.error(f'Cannot open folder: {exc}')
        return
    total = len(files)
    try:
        expression = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        st.error(f'Invalid regex: {exc}')
        return
    files = [path for path in files if expression.search(path.name)]
    st.caption(f'{len(files)} / {total} images · Use the expand icon to enlarge a preview. Previews use automatic contrast.')
    if not files:
        st.info('No FITS files match this filter.' if total else 'No FITS files in this folder.')
        return
    for offset in range(0, len(files), 4):
        for column, path in zip(st.columns(4, gap='medium'), files[offset:offset + 4]):
            with column:
                try:
                    stat = path.stat()
                    st.image(fits_thumbnail(path, stat.st_mtime_ns, stat.st_size), width='stretch')
                except (ValueError, OSError, TypeError, KeyError) as exc:
                    st.warning(f'Preview unavailable: {exc}')
                st.caption(path.name)
