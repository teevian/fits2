"""Optional, mutually exclusive processing tools for the FITS viewer."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import streamlit as st

from imageProcessing import (Frame, calibrate, combine, mask_image, read_frame,
                       read_mask, add_history)
from download_ui import archive_download_dialog, image_downloads


OPERATIONS = ['Master bias', 'Flat correction', 'Combine images',
              'Image mask', 'Reduce science images']


def list_fits(folder):
    return sorted((p for p in Path(folder).expanduser().iterdir() if p.is_file()
                   and p.name.lower().endswith(('.fits', '.fit', '.fts', '.fits.gz', '.fit.gz', '.fz'))),
                  key=lambda p: p.name.casefold())


def identity(value):
    if isinstance(value, Path):
        stat = value.stat()
        return str(value.resolve()), stat.st_mtime_ns, stat.st_size
    if isinstance(value, Frame):
        return value.name, id(value)
    return sha256(value).hexdigest()


def load(value):
    """Uploaded inputs retain filenames via a (name, bytes) pair."""
    if isinstance(value, tuple):
        image = read_frame(value[1])
        image.name = value[0]
        return image
    return read_frame(value)


def select_input(label, key, catalog, multiple=False):
    # Adding a generated master changes the options. Preserve selections across
    # that change instead of allowing Streamlit to reset the widget.
    if key in st.session_state:
        previous = st.session_state[key]
        st.session_state[key] = ([item for item in previous if item in catalog] if multiple
                                 else (previous if previous in catalog else None))
    items = st.multiselect(label, list(catalog), key=key, placeholder='Choose images…') if multiple else [
        st.selectbox(label, list(catalog), index=None, placeholder='Choose…', key=key)]
    chosen = [catalog[item] for item in items if item is not None]
    uploads = st.file_uploader(f'Or upload {label.lower()}', type=['fits', 'fit', 'fts', 'gz', 'fz'],
                               accept_multiple_files=multiple, key=f'{key}_upload')
    if uploads:
        if not multiple:
            # A single uploaded calibration takes precedence over the selector.
            chosen = [(uploads.name, uploads.getvalue())]
        else:
            chosen += [(upload.name, upload.getvalue()) for upload in uploads]
    return chosen if multiple else (chosen[0] if chosen else None)


def fingerprint(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return value[0], identity(value[1])
    return identity(value)


def require(value, label):
    if value is None:
        raise ValueError(f'Select {label}.')
    return value


def calibration_controls(prefix, catalog, include_flat=True):
    values = {}
    values['use_bias'] = st.checkbox('Subtract master bias', key=f'{prefix}_bias_on')
    values['bias'] = select_input('Master bias', f'{prefix}_bias', catalog) if values['use_bias'] else None
    values['use_dark'] = st.checkbox('Subtract master dark', key=f'{prefix}_dark_on')
    values['dark'] = None
    values['dark_contains_bias'] = False
    values['scale_dark'] = False
    if values['use_dark']:
        values['dark'] = select_input('Master dark', f'{prefix}_dark', catalog)
        values['dark_contains_bias'] = st.selectbox('Dark state', ['Bias already removed', 'Includes bias'],
                                                key=f'{prefix}_dark_state') == 'Includes bias'
        if not values['dark_contains_bias']:
            values['scale_dark'] = st.checkbox('Scale dark by exposure', key=f'{prefix}_dark_scale')
        else:
            st.caption('Use the same exposure as the image and disable separate bias subtraction.')
    values['use_flat'] = include_flat and st.checkbox('Apply master flat', key=f'{prefix}_flat_on')
    values['flat'] = select_input('Master flat', f'{prefix}_flat', catalog) if values['use_flat'] else None
    values['flat_min'] = st.number_input('Minimum flat response', 0., 1., 0.05, 0.01,
                                         key=f'{prefix}_flat_min') if values['use_flat'] else 0.
    return values


def calibration_arguments(values):
    return dict(bias=load(require(values['bias'], 'a master bias')) if values['use_bias'] else None,
                dark=load(require(values['dark'], 'a master dark')) if values['use_dark'] else None,
                flat=load(require(values['flat'], 'a master flat')) if values['use_flat'] else None,
                flat_min=values['flat_min'], scale_dark=values['scale_dark'],
                dark_contains_bias=values['dark_contains_bias'])


def processing_panel(current, source_id, folder, on_view_change=None):
    """Return a processed preview only when its inputs/options still match."""
    operation = st.selectbox('Operation', ['None'] + OPERATIONS, key='processing_operation')
    if operation == 'None':
        return None
    products = st.session_state.setdefault('reduction_products', {})
    with st.container():
        auxiliary_folder = st.text_input('Input folder', str(folder), key=f'calibration_folder_{folder}')
        catalog = {}
        try:
            catalog.update({p.name: p for p in list_fits(auxiliary_folder)})
        except OSError as exc:
            st.warning(f'Folder unavailable: {exc}')
        catalog.update({f'✓ {name}': value for name, value in products.items()})
        inputs = []
        calibrations = None
        flat = external = None
        lower = upper = rectangle = None
        sigma = 5.
        method = 'mean'
        normalize = False
        align = False
        max_shift = 100
        subpixel = True
        flat_min = 0.05
        flat_action = ''
        aligned = True
        target = 'Open image'
        mask_enabled = False
        if operation == 'Flat correction':
            flat_action = st.selectbox('Action', ['Apply master flat', 'Create master flat'])
        combining = operation in ('Master bias', 'Combine images') or flat_action == 'Create master flat'
        if combining:
            inputs = select_input('Images to combine', f'inputs_{operation}_{flat_action}', catalog, multiple=True)
            a, b = st.columns(2)
            method = 'mean' if a.selectbox('Method', ['Mean', 'Median'], key=f'method_{operation}') == 'Mean' else 'median'
            clipping = b.checkbox('Reject outliers (sigma)', value=True, key=f'clip_{operation}')
            sigma = st.number_input('Sigma threshold', 1., 20., 5., 0.5, key=f'sigma_{operation}') if clipping else None
            if operation == 'Combine images':
                normalize = st.checkbox('Normalize each image by median', key='combine_normalize')
                align = st.checkbox('Align images before combining', key='combine_align')
                if align:
                    a, b = st.columns(2)
                    subpixel = a.selectbox('Precision', ['Subpixel', 'Whole pixel']) == 'Subpixel'
                    maximum = max(1, min(current.data.shape) // 2 - 2)
                    max_shift = b.number_input('Maximum shift', 1, maximum, min(100, maximum), 10)
                    st.caption('The first image is the reference. Corrects X/Y translation only.')
                else:
                    aligned = st.checkbox('Images are already aligned', key='combine_aligned')
            elif operation != 'Master bias':
                calibrations = calibration_controls('flat_create', catalog, include_flat=False)
        elif operation == 'Flat correction':
            flat = select_input('Master flat', 'flat_apply', catalog)
            flat_min = st.number_input('Minimum flat response', 0., 1., 0.05, 0.01)
        elif operation == 'Image mask':
            a, b = st.columns(2)
            if a.checkbox('Exclude below'):
                lower = a.number_input('Lower limit', value=0.)
            if b.checkbox('Exclude above'):
                upper = b.number_input('Upper limit', value=65535.)
            if st.checkbox('Exclude rectangle'):
                height, width = current.data.shape
                a, b = st.columns(2)
                x0 = a.number_input('X start', 0, width-1, 0, key='mask_x0')
                x1 = b.number_input('X end (exclusive)', int(x0)+1, width, width, key='mask_x1')
                c, d = st.columns(2)
                y0 = c.number_input('Y start', 0, height-1, 0, key='mask_y0')
                y1 = d.number_input('Y end (exclusive)', int(y0)+1, height, height, key='mask_y1')
                rectangle = (x0, x1, y0, y1)
            mask_enabled = st.checkbox('Apply FITS mask', key='mask_external_on')
            if mask_enabled:
                external = select_input('Mask', 'mask_external', catalog)
            st.caption('1 = excluded; 0 = valid. Non-finite values are always excluded.')
        else:
            target = st.selectbox('Apply to', ['Open image', 'Multiple images'])
            if target == 'Multiple images':
                inputs = select_input('Science images', 'science_inputs', catalog, multiple=True)
            calibrations = calibration_controls('science', catalog)
            mask_enabled = st.checkbox('Apply mask', key='science_mask_on')
            if mask_enabled:
                external = select_input('Mask', 'science_mask', catalog)

        config = dict(operation=operation, source=source_id, inputs=[fingerprint(v) for v in inputs],
                      method=method, sigma=sigma, normalize=normalize, flat_action=flat_action,
                      flat=fingerprint(flat), flat_min=flat_min, target=target, aligned=aligned,
                      align=align, max_shift=max_shift, subpixel=subpixel,
                      external=fingerprint(external), mask_enabled=mask_enabled,
                      lower=lower, upper=upper, rectangle=rectangle)
        if calibrations:
            config['calibrations'] = {key: fingerprint(value) if key in ('bias', 'dark', 'flat') else value
                                      for key, value in calibrations.items()}
        signature = repr(config)
        if st.button('Run', key='process_run'):
            try:
                result_frames = []
                if mask_enabled:
                    require(external, 'a mask')
                mask = read_mask(external[1] if isinstance(external, tuple) else external) if external is not None else None
                with st.spinner('Processing…'):
                    if combining:
                        if operation == 'Combine images' and not align and not aligned:
                            raise ValueError('Confirm that the images are aligned before combining.')
                        kind = 'bias' if operation == 'Master bias' else ('flat' if flat_action else 'combined')
                        args = calibration_arguments(calibrations) if calibrations else {}
                        args.pop('flat', None)
                        args.pop('flat_min', None)
                        progress = st.progress(0.)
                        # Paths stay lazy; uploaded data acquire their source names.
                        sources = [load(v) if isinstance(v, tuple) else v for v in inputs]
                        result_frames = [combine(sources, method=method, sigma=sigma, normalize=normalize,
                                                 kind=kind, progress=progress.progress, align=align,
                                                 max_shift=max_shift, subpixel=subpixel, **args)]
                        progress.empty()
                    elif operation == 'Flat correction':
                        result_frames = [calibrate(current, flat=load(require(flat, 'a master flat')), flat_min=flat_min)]
                    elif operation == 'Image mask':
                        result_frames = [mask_image(current, lower, upper, rectangle, mask)]
                    else:
                        args = calibration_arguments(calibrations)
                        if not any(calibrations[key] for key in ('use_bias', 'use_dark', 'use_flat')) and mask is None:
                            raise ValueError('Enable at least one reduction step.')
                        science = inputs if target == 'Multiple images' else [current]
                        if not science:
                            raise ValueError('Select science images.')
                        for value in science:
                            image = load(value)
                            reduced = calibrate(image, mask=mask, **args)
                            reduced.name = Path(image.name).stem + '_reduced.fits'
                            add_history(reduced.header, f'Source: {image.name}')
                            result_frames.append(reduced)
                    result_names = []
                    for result in result_frames:
                        if not combining and operation != 'Reduce science images':
                            suffix = 'flat' if operation == 'Flat correction' else 'mask'
                            result.name = Path(current.name).stem + f'_{suffix}.fits'
                        number = st.session_state.get('product_number', 0) + 1
                        st.session_state.product_number = number
                        result.name = f'{number:03d}_{result.name}'
                        if not combining and operation != 'Reduce science images':
                            add_history(result.header, f'Source: {current.name}')
                        products[result.name] = result
                        result_names.append(result.name)
                    st.session_state.reduction_last = dict(signature=signature, names=result_names)
                    st.rerun()
            except (ValueError, OSError, TypeError, KeyError, MemoryError) as exc:
                st.error(str(exc) or 'Insufficient memory for this operation.')
        last = st.session_state.get('reduction_last')
        if last and last['signature'] == signature:
            names = last['names']
            selected = st.selectbox('Result', names, key=f'result_{names[0]}') if len(names) > 1 else names[0]
            result = products[selected]
            st.caption(f'{len(names)} result(s) · {int(result.mask.sum())} excluded pixels in selected image')
            image_downloads(result, 'processing_download')
            if result.header.get('ALIGN') == 'TRANSLATION':
                with st.expander('Applied shifts'):
                    rows = [{'Image': i + 1,
                             'Y': result.header.get(f'ALN{i:03d}Y', 0.),
                             'X': result.header.get(f'ALN{i:03d}X', 0.)}
                            for i in range(result.header.get('NCOMBINE', 0))]
                    st.dataframe(rows, width='stretch', hide_index=True)
            if len(names) > 1:
                if st.button('Download ZIP batch', key='processing_download_batch'):
                    archive_download_dialog({name: products[name] for name in names}, 'reduced_images')
            show = st.selectbox('View', ['Result', 'Original'], key='processing_preview',
                                on_change=on_view_change)
            if show == 'Result':
                return result
        elif last:
            st.caption('Options changed. Run again to update the result.')
        if products:
            with st.expander('Session results'):
                selected_product = st.selectbox('Created file', list(products), key='saved_product')
                image_downloads(products[selected_product], 'saved_product_download',
                                fits_label='Save FITS copy', png_label='Save PNG copy')
    return None
