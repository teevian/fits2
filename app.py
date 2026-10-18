"""Black FITS workspace with file and processing controls in the sidebar."""
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy.io import fits

from imageProcessing import display_group, default_display_limits
from image_viewer import show as show_image, show_empty
from color_composition import ColorComposite
from processing_ui import identity, list_fits
from imageProcessing import frame
from workflow_ui import workflow_panel
from workspace_settings import load_working_folder, save_working_folder
from gallery_ui import gallery_page

st.set_page_config(page_title='FITS Viewer', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {padding-top: .5rem;}
:root {color-scheme: dark; --primary-color:#7f91af; --text-color:#d3d3d3;
    --background-color:#000; --secondary-background-color:#000;}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stSidebar"], [data-testid="stSidebarContent"],
[data-testid="stDialog"] [role="dialog"] {background:#000; color:#d3d3d3;}
[data-baseweb="input"], [data-baseweb="select"] > div,
[data-baseweb="textarea"], [data-baseweb="popover"] [role="listbox"],
[data-testid="stFileUploaderDropzone"], [data-testid="stExpander"] details,
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
[data-testid="stLinkButton"] a {background:#000; color:#d3d3d3;}
[data-testid="stAlert"] {background:rgba(127,145,175,.16); color:#d3d3d3;}
[data-testid="stAlert"] svg {fill:#7f91af; color:#7f91af;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {background:#7f91af;}
button[kind="primary"] {background:#7f91af !important; color:#000 !important; border-color:#7f91af !important;}
button:hover, button:focus-visible, [role="tab"][aria-selected="true"] {color:#a7b8d3 !important;}
[data-testid="stCaptionContainer"] {color:#b8b8b8;}
.st-key-image_viewer {background:#000;}
:root {--image-height: clamp(450px, calc((100vw - 530px) * .667 + 75px), 1250px);}
[data-testid="stVerticalBlock"] {gap: .5rem !important;}
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {display: none !important;}
[data-testid="stSidebar"] {width: 380px !important; min-width: 380px; max-width: 380px; transform: translateX(0) !important;}
[data-testid="stSidebarUserContent"] {padding: .5rem .75rem;}
[data-testid="stFileUploaderDropzone"] {height: 42px; min-height: 0; padding: .25rem .5rem;}
[data-testid="stFileUploaderDropzoneInstructions"] > span,
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:last-child {display: none;}
[data-testid="stNumberInput"] button {display: none;}
[data-testid="stMain"] [data-testid="stTabs"] [role="tabpanel"] {height: 350px; overflow-y: auto;}
.st-key-image_viewer {height: var(--image-height); min-height: var(--image-height); overflow: hidden;}
.st-key-image_viewer iframe[title="st.iframe"] {height: var(--image-height) !important; border: 0 !important;}
.st-key-image_viewer [data-testid="stElementContainer"]:has(iframe[title="st.iframe"]) {
    height: var(--image-height) !important;
    flex: 0 0 var(--image-height) !important;
}
</style>''', unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
DEFAULT_FOLDER = ROOT / 'share-20261001'
if 'folder_path' not in st.session_state:
    st.session_state.folder_path = str(load_working_folder(
        DEFAULT_FOLDER if DEFAULT_FOLDER.is_dir() else ROOT))

if st.query_params.get('page') == 'all':
    gallery_page(Path(st.query_params.get('folder', st.session_state.folder_path)).expanduser())
    st.stop()


def remember_working_folder():
    save_working_folder(st.session_state.folder_path)


def header_panel(header):
    lines = []
    for card in header.cards:
        suffix = f' {card.comment}' if card.comment else ''
        lines.append(f'{card.keyword:<12}  {card.value}{suffix}')
    st.text('\n'.join(lines))


def image_details(header):
    with st.container(key="image_header"):
        header_tab, = st.tabs(['Header'])
        with header_tab:
            header_panel(header)


def refresh_image_view():
    st.session_state.refresh_image_view = True


@st.fragment
def processing_controls(original, source_id, folder, auto=False):
    result = workflow_panel(original, source_id, folder, on_view_change=refresh_image_view, auto=auto)
    if st.session_state.pop('refresh_image_view', False):
        st.rerun()
    return result


with st.sidebar:
    folder = Path(st.text_input('Working folder', key='folder_path',
                                on_change=remember_working_folder)).expanduser()
    files = []
    if folder.is_dir():
        try:
            files = list_fits(folder)
        except OSError as exc:
            st.error(f'Cannot open folder: {exc}')
    else:
        st.error('Folder not found.')
    auto_column, all_column = st.columns(2, gap='small')
    auto_clicked = auto_column.button('auto', key='workflow_auto', disabled=not files,
                            use_container_width=True,
                            help='Combine bias → combine flats with -bias → correct every science file.')
    all_column.link_button('all', '?' + urlencode({'page': 'all', 'folder': str(folder.resolve())}),
                           disabled=not files, width='stretch',
                           help='Open all FITS images in a new tab.')
    if auto_clicked:
        st.session_state.pop('workflow_auto_completed', None)
    completed = st.session_state.get('workflow_auto_completed')
    if completed and completed[0] == str(folder):
        st.success(f'Auto complete: {completed[1]} science frames corrected.')

image_area = st.container()
# The first file supplies workflow geometry; only an explicit view or drop opens it.
source = files[0] if files else None
image_selected = False
if source is None:
    with image_area.container(key='image_viewer'):
        show_empty()
    image_area.caption('Double-click an image in a grid to view it.')
    st.stop()

name = source.name
source_id = identity(source) if isinstance(source, Path) else identity(source.getvalue())
name_slot = image_area.empty()
if image_selected:
    name_slot.caption(name)

try:
    hdus = fits.open(source if isinstance(source, Path) else BytesIO(source.getvalue()), memmap=False)
except Exception as exc:
    st.error(f'Cannot read FITS file: {exc}')
    st.stop()

with hdus, image_area:
    index = st.selectbox('HDU', range(len(hdus)),
                         format_func=lambda i: f'{i} · {hdus[i].name} ({type(hdus[i]).__name__})') if len(hdus) > 1 else 0
    hdu = hdus[index]
    header = hdu.header.copy()
    try:
        data = hdu.data
    except Exception as exc:
        st.error(f'Cannot read image data: {exc}')
        st.stop()
    if data is None:
        st.info('This HDU has no data. Choose another HDU.')
        st.stop()

    if not image_selected and (data.dtype.names or data.ndim < 2):
        with st.container(key='image_viewer'):
            show_empty()
        st.caption('Double-click an image in a grid to view it.')
        st.stop()
    if data.dtype.names:
        table = pd.DataFrame({column: [value.tolist() if isinstance(value, np.ndarray) else value
                                       for value in data[column]] for column in data.dtype.names})
        st.dataframe(table, width='stretch')
        st.download_button('Download CSV', table.to_csv(index=False), 'table.csv', 'text/csv')
        image_details(header)
        with st.sidebar:
            st.caption('Processing requires a 2D image.')
    elif data.ndim == 1:
        figure = go.Figure(go.Scatter(y=data, mode='lines', line_color='#555555'))
        figure.update_layout(xaxis_title='Pixel', yaxis_title=header.get('BUNIT', 'Value'),
                             template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000',
                             height=320, margin=dict(l=45, r=10, t=10, b=40))
        st.plotly_chart(figure, use_container_width=True,
                        config={'scrollZoom': False, 'displayModeBar': False})
        st.download_button('Download CSV', pd.DataFrame({'pixel': np.arange(data.size),
                           'value': data}).to_csv(index=False), 'series.csv', 'text/csv')
        image_details(header)
        with st.sidebar:
            st.caption('Processing requires a 2D image.')
    elif data.ndim >= 2 and all(data.shape):
        active_mask = None
        if data.ndim == 2:
            source_mask = hdus['MASK'].data if 'MASK' in hdus and hdu.name != 'MASK' else None
            original = frame(data, header, source_mask, name)
            with st.sidebar:
                processed = processing_controls(original, (source_id, index), folder, auto=auto_clicked)
            if not image_selected and processed is None:
                with st.container(key='image_viewer'):
                    show_empty()
                st.caption('Double-click an image in a grid to view it.')
                st.stop()
            active = processed or original
            name = active.name
            name_slot.caption(name)
            if isinstance(active, ColorComposite):
                with st.container(key='image_viewer'):
                    show_image(active.rgb(st.session_state.get('viewer_contrast', False)), 1)
                image_details(active.header)
                st.stop()
            data, header, active_mask = active.data, active.header, active.mask
        slices = []
        if data.ndim > 2:
            with st.sidebar:
                for axis, size in enumerate(data.shape[:-2]):
                    slices.append(st.number_input(f'Plane · axis {data.ndim-axis}', 0, size-1, 0,
                                                   key=f'axis{axis}'))
        plane = np.asarray(data[tuple(slices)] if slices else data, dtype=float)
        if active_mask is not None:
            plane = np.where(active_mask, np.nan, plane)
        finite = plane[np.isfinite(plane)]
        if not finite.size:
            st.warning('This image has no valid pixels to display.')
            st.stop()
        group = display_group(folder, name, header, index)
        display_limits = default_display_limits(group)
        contrast_enabled = st.session_state.get('viewer_contrast', False)
        contrast_reference = None
        # Compare raw and bias-corrected masters on one scale covering both versions.
        # Re-fitting each separately would hide a nearly constant bias subtraction.
        flat_versions = st.session_state.get('workflow_flat_versions')
        if flat_versions:
            versions = flat_versions[1]
            for key, master in versions[False].items():
                corrected = versions[True].get(key)
                if (corrected is None or not corrected.header.get('NORMALIZ')) and (name == master.name or (corrected is not None and name == corrected.name)):
                    contrast_reference = (master.display(), corrected.display()) if corrected is not None else master.display()
                    break
        image_low, image_high = float(finite.min()), float(finite.max())
        if image_low == image_high:
            st.caption(f'This image is uniform: every valid pixel has value {image_low:g}.')
        with st.container(key='image_viewer'):
            show_image(plane, 1, display_limits=display_limits, contrast=contrast_enabled,
                       contrast_reference=contrast_reference)
        image_details(header)
    else:
        st.info('This HDU does not contain a supported image or series.')
