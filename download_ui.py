"""Offer both image formats; row downloads are prepared only when requested."""
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from imageProcessing import calibration_archive, fits_data, image_archive, image_name, png_name, preview_png, to_png
from processing_jobs import run_with_timeout


def image_downloads(image, key, fits_label='Download FITS', png_label='Download PNG', *, display_png=False, contrast=False):
    fits_column, png_column = st.columns(2, gap='small')
    fits_column.download_button(fits_label, fits_data(image), image_name(image),
                                'application/fits', key=f'{key}_fits', on_click='ignore')
    png_data = preview_png(image, contrast) if display_png else to_png(image)
    png_column.download_button(png_label, png_data, png_name(image_name(image)),
                               'image/png', key=f'{key}_png', on_click='ignore')


@st.dialog('Download image')
def image_download_dialog(image, display_png=False, contrast=False):
    st.caption(image_name(image))
    image_downloads(image, 'image_export', display_png=display_png, contrast=contrast)


@st.dialog('Download ZIP')
def archive_download_dialog(images, name, fits_archive=None, display_png=False, contrast=False):
    fits_column, png_column = st.columns(2, gap='small')
    fits_column.download_button('FITS ZIP', fits_archive if fits_archive is not None
                                else image_archive(images), f'{name}_fits.zip',
                                'application/zip', key='archive_export_fits', on_click='ignore')
    png_column.download_button('PNG ZIP', image_archive(images, 'png', display_png, contrast),
                               f'{name}_png.zip', 'application/zip',
                               key='archive_export_png', on_click='ignore')


@st.cache_resource
def _zip_executor():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix='fits-export')


def _prepare_calibration_zip(versions, file_format):
    return run_with_timeout(calibration_archive, versions, file_format)


def _calibration_export_record(versions, name):
    from processing_ui import fingerprint
    signature = repr(('display-zip-v3', tuple(
        (mode, tuple((key, fingerprint(image)) for key, image in images.items()))
        for mode, images in versions.items())))
    cache = st.session_state.setdefault('calibration_export_cache', {})
    record = cache.get(name)
    if record is None or record['signature'] != signature:
        if record:
            for job in record['jobs'].values():
                job.cancel()
        record = {'signature': signature, 'files': {}, 'errors': {}, 'jobs': {}}
        cache[name] = record
        # Snapshot the dictionaries so removing a displayed group cannot alter
        # an export already being prepared. Master pixel arrays are read-only.
        snapshot = {mode: dict(images) for mode, images in versions.items()}
        record['versions'] = snapshot
        for file_format in ('fits', 'png'):
            record['jobs'][file_format] = _zip_executor().submit(
                _prepare_calibration_zip, snapshot, file_format)
    return record


def _retry_calibration_zip(record, file_format):
    record['errors'].pop(file_format, None)
    record['jobs'][file_format] = _zip_executor().submit(
        _prepare_calibration_zip, record['versions'], file_format)


@st.fragment(run_every=1)
def _calibration_download_contents(record, name):
    for file_format, job in list(record['jobs'].items()):
        if job.done():
            try:
                record['files'][file_format] = job.result()
            except Exception as exc:
                record['errors'][file_format] = str(exc) or 'The ZIP could not be prepared.'
            del record['jobs'][file_format]
    st.caption('with_constrast: Contrast on · no_contrast: Contrast off (or original FITS).')
    fits_column, png_column = st.columns(2, gap='small')
    for file_format, column, help_text in (
            ('fits', fits_column, 'Original FITS and contrast PNG previews in two folders.'),
            ('png', png_column, 'PNGs matching the viewer with Contrast off and on, in two folders.')):
        data = record['files'].get(file_format)
        column.download_button(f'{file_format.upper()} ZIP', data if data is not None else b'',
                               f'{name}_{file_format}.zip', 'application/zip',
                               key=f'archive_export_{file_format}', on_click='ignore',
                               disabled=data is None, help=help_text)
        error = record['errors'].get(file_format)
        if error:
            column.error(error)
            column.button(f'Retry {file_format.upper()} ZIP', key=f'archive_retry_{file_format}',
                          on_click=_retry_calibration_zip, args=(record, file_format))
    if record['jobs']:
        st.caption('Preparing ZIP files… Downloads become available as each file is ready.')


@st.dialog('Download ZIP', on_dismiss='rerun')
def calibration_download_dialog(versions, name):
    _calibration_download_contents(_calibration_export_record(versions, name), name)


def image_save_button(column, image, key, display_png=False, contrast=False):
    if column.button('', icon=':material/save:', help='Save image', key=key):
        image_download_dialog(image, display_png, contrast)
