from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from astropy.io import fits
from PIL import Image
from streamlit.testing.v1 import AppTest

from imageProcessing import calibration_archive, fits_data, image_archive, png_name, to_png
from imageProcessing import frame, to_fits


def test_png_preserves_resolution_orientation_and_source_pixels():
    values = np.arange(12.).reshape(3, 4)
    mask = np.zeros_like(values, dtype=bool)
    mask[0, 0] = True
    image = frame(values, mask=mask, name='image.fits')
    original = to_fits(image)
    with Image.open(BytesIO(to_png(image))) as png:
        assert png.size == (4, 3)
        pixels = np.asarray(png)
        assert pixels[0, -1] == 11
        assert pixels.dtype == np.uint16
        assert pixels[-1, 0] == 0
        assert pixels[0, 0] > pixels[-1, -1]
    np.testing.assert_array_equal(image.data, values)
    np.testing.assert_array_equal(image.mask, mask)
    assert to_fits(image) == original
    for data, expected in [(np.full((2, 3), 10.), 10), (np.full((2, 3), np.nan), 0)]:
        with Image.open(BytesIO(to_png(frame(data)))) as png:
            np.testing.assert_array_equal(np.asarray(png), np.full((2, 3), expected))


def test_individual_and_zip_formats_keep_fits_bytes(tmp_path):
    image = frame(np.arange(12.).reshape(3, 4), name='master_flat_blue.fits')
    raw = to_fits(image)
    path = tmp_path / image.name
    path.write_bytes(raw)
    sources = {'local': path, 'uploaded': ('uploaded.FITS.gz', raw),
               'master': frame(np.ones((3, 4)), name='master_bias_1.fits')}
    assert fits_data(path) == raw
    assert fits_data(sources['uploaded']) == raw
    assert png_name('uploaded.FITS.gz') == 'uploaded.png'
    assert png_name('image.fit') == 'image.png'
    assert png_name('image.fits.fz') == 'image.png'
    with ZipFile(BytesIO(image_archive(sources))) as archive:
        assert set(archive.namelist()) == {
            image.name, 'uploaded.FITS.gz', 'master_bias_1.fits'}
        assert archive.read(image.name) == raw
        assert archive.read('uploaded.FITS.gz') == raw
    with ZipFile(BytesIO(image_archive(sources, 'png'))) as archive:
        assert set(archive.namelist()) == {
            'master_flat_blue.png', 'uploaded.png', 'master_bias_1.png'}
        for name in archive.namelist():
            with Image.open(BytesIO(archive.read(name))) as png:
                assert png.format == 'PNG' and png.size == (4, 3)
    assert path.read_bytes() == raw


def test_row_save_opens_both_formats_inside_processing_fragment(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    fits.writeto(tmp_path / 'bias1.fits', np.arange(144.).reshape(12, 12))
    fits.writeto(tmp_path / 'bias2.fits', np.arange(144.).reshape(12, 12))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.button(key='workflow_bias_combine').click().run()
    app.button(key='workflow_bias_download').click().run()
    assert not app.exception
    downloads = [item.label for item in app.get('download_button')]
    assert downloads.count('Download FITS') == 1  # Only the requested result dialog.
    assert downloads.count('Download PNG') == 1
    assert any(item.value.startswith('master_bias_') for item in app.caption)


def test_zip_choices_and_not_combined_list_are_available_lazily(tmp_path, monkeypatch):
    import download_ui
    conversions = []
    original = download_ui.to_png

    def record(image):
        conversions.append(image)
        return original(image)

    monkeypatch.setattr(download_ui, 'to_png', record)
    for name, level, kind, color in [
            ('science.fits', 100., 'OBJECT', 'Blue'),
            ('bias1.fits', 10., 'BIAS', None), ('bias2.fits', 10., 'BIAS', None),
            ('flat_Blue_1.fits', 110., 'FLAT', 'Blue'),
            ('flat_Blue_2.fits', 120., 'FLAT', 'Blue'),
            ('flat_Red_1.fits', 130., 'FLAT', 'Red')]:
        header = fits.Header({'IMAGETYP': kind, 'EXPTIME': 1.})
        if color:
            header['FILTER'] = color
        fits.writeto(tmp_path / name, np.full((12, 12), level), header)
    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
with st.sidebar:
    workflow_panel(read_frame(folder / 'science.fits'), 'source', folder)
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    assert not app.exception
    skipped = [item.value for item in app.markdown if 'not combined</strong>' in item.value]
    assert len(skipped) == 1
    assert '?' not in skipped[0] and 'title=' not in skipped[0]
    assert any(item.value == 'Groups with fewer than two images are shown below and are not combined.'
               for item in app.caption)
    elements = [item for item in app.sidebar if item.type in ('markdown', 'caption', 'button')]
    description_index = next(i for i, item in enumerate(elements)
                             if item.type == 'caption' and item.value.startswith('Groups with fewer'))
    assert 'not combined</strong>' in elements[description_index + 1].value
    assert not conversions
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    master = app.session_state['workflow_flat_last'][1]['Blue']
    assert master.name == 'corrected_flat_blue_1s_x2.fits'
    with ZipFile(BytesIO(app.session_state['workflow_flat_last'][2])) as archive:
        assert archive.namelist() == [master.name]
    assert not conversions
    app.button(key='workflow_flat_download_Blue').click().run()
    assert not app.exception
    assert len(conversions) == 1
    assert {'Download FITS', 'Download PNG'} <= {
        item.label for item in app.get('download_button')}
    app.run()
    app.button(key='workflow_flat_download_all').click().run()
    assert not app.exception
    assert {'FITS ZIP', 'PNG ZIP'} <= {item.label for item in app.get('download_button')}


def test_raw_view_and_png_keep_counts_without_automatic_adjustments(monkeypatch):
    from base64 import b64decode
    import json
    import image_viewer
    from imageProcessing import correct_flat_groups

    def unexpected_contrast(*args, **kwargs):
        raise AssertionError('Raw display and export must not compute adaptive contrast.')

    monkeypatch.setattr(np, 'percentile', unexpected_contrast)
    raw = frame(np.arange(100., 200.).reshape(10, 10), name='master_flat_blue_x2.fits')
    bias = frame(np.full((10, 10), 20.), name='master_bias_x2.fits')
    before = raw.data.copy()
    corrected = correct_flat_groups({'Blue': raw}, bias)['Blue']
    np.testing.assert_array_equal(raw.data, before)
    np.testing.assert_array_equal(corrected.data, before - 20.)
    for image in (raw, corrected):
        with Image.open(BytesIO(to_png(image))) as png:
            np.testing.assert_array_equal(np.asarray(png), np.flipud(image.data).astype(np.uint16))
    assert to_png(('corrected.fits', to_fits(corrected))) == to_png(corrected)
    with ZipFile(BytesIO(image_archive({'plain': raw, 'corrected': corrected}, 'png'))) as archive:
        with Image.open(BytesIO(archive.read('corrected_flat_blue_x2.png'))) as png:
            np.testing.assert_array_equal(np.asarray(png), np.flipud(corrected.data).astype(np.uint16))
    rendered = []
    monkeypatch.setattr(image_viewer.components, 'html', lambda html, **kwargs: rendered.append(html))
    for image in (raw, corrected):
        image_viewer.show(image.display(), 1)
    for image, html in zip((raw, corrected), rendered):
        encoded = json.loads(html.split('const raw = atob(', 1)[1].split(');', 1)[0])
        np.testing.assert_array_equal(np.frombuffer(b64decode(encoded), dtype='<f8').reshape(10, 10),
                                       image.data)
        assert 'const displayLow = 0.0, displayHigh = 65535.0;' in html
        assert 'imageSmoothingEnabled = false' in html
        for adjustment in ('percentiles', 'sessionStorage', 'Stretch', 'Invert', 'Math.log1p', 'Math.asinh'):
            assert adjustment not in html


def test_fits_preserves_values_png_cannot_represent():
    image = frame([[-12.5, 0.5, 4000.25, 70000.]], name='corrected_flat.fits')
    original = image.data.copy()
    with Image.open(BytesIO(to_png(image))) as png:
        np.testing.assert_array_equal(np.asarray(png), [[0, 0, 4000, 65535]])
    with fits.open(BytesIO(to_fits(image))) as hdus:
        np.testing.assert_array_equal(hdus[0].data, original)
    np.testing.assert_array_equal(image.data, original)


def test_optional_contrast_uses_previous_percentiles_and_preserves_raw_pixels(monkeypatch):
    from base64 import b64decode
    import json
    import image_viewer

    pixels = np.arange(100., 200.).reshape(10, 10)
    pixels[0, 0] = np.nan
    pixels[-1, -1] = np.inf
    original = pixels.copy()
    image = frame(pixels, name='flat.fits')
    fits_before, png_before = to_fits(image), to_png(image)
    rendered = []
    monkeypatch.setattr(image_viewer.components, 'html', lambda html, **kwargs: rendered.append(html))
    image_viewer.show(pixels, 1, display_limits=(80., 240.), contrast=True)
    low, high = np.percentile(pixels[np.isfinite(pixels)], [1, 99])
    html = rendered[-1]
    assert f'const displayLow = {low}, displayHigh = {high};' in html
    encoded = json.loads(html.split('const raw = atob(', 1)[1].split(');', 1)[0])
    np.testing.assert_array_equal(np.frombuffer(b64decode(encoded), dtype='<f8').reshape(pixels.shape), original)
    image_viewer.show(pixels, 1, display_limits=(80., 240.), contrast=False)
    assert 'const displayLow = 80.0, displayHigh = 240.0;' in rendered[-1]
    image_viewer.show(np.full((2, 2), 65535.), 1, contrast=True)
    assert 'const displayLow = 0.0, displayHigh = 65535.0;' in rendered[-1]
    np.testing.assert_array_equal(pixels, original)
    assert to_fits(image) == fits_before
    assert to_png(image) == png_before


def test_calibration_archives_include_all_flat_versions_and_share_screen_contrast(monkeypatch):
    import image_viewer
    from imageProcessing import correct_flat_groups
    from imageProcessing import resolve_display_limits

    raw = frame(np.arange(100., 200.).reshape(10, 10), name='master_flat_blue_0.2s_x12.fits')
    raw.mask[0, 0] = True
    bias = frame(np.full((10, 10), 20.), name='master_bias_x2.fits')
    red = frame(raw.data * 2., name='master_flat_red_0.3s_x3.fits')
    masters = {'Blue': raw, 'Red': red}
    versions = {False: masters, True: correct_flat_groups(masters, bias)}
    originals = {image.name: (image.data.copy(), image.mask.copy())
                 for images in versions.values() for image in images.values()}
    all_images = [image for images in versions.values() for image in images.values()]
    rendered = []
    monkeypatch.setattr(image_viewer.components, 'html', lambda html, **kwargs: rendered.append(html))
    for file_format in ('fits', 'png'):
        with ZipFile(BytesIO(calibration_archive(versions, file_format))) as archive:
            assert set(archive.namelist()) == {'with_constrast/', 'no_contrast/'} | {
                f'with_constrast/{png_name(image.name)}' for image in all_images} | {
                f'no_contrast/{image.name if file_format == "fits" else png_name(image.name)}' for image in all_images}
            for key in masters:
                reference = [versions[mode][key].display() for mode in (False, True)]
                low, high = resolve_display_limits(reference[0], contrast=True, reference=reference)
                for mode in (False, True):
                    image = versions[mode][key]
                    image_viewer.show(image.display(), 1, contrast=True, contrast_reference=reference)
                    assert f'const displayLow = {low}, displayHigh = {high};' in rendered[-1]
                    expected = np.floor(np.clip((image.data - low) / (high-low), 0., 1.) * 255. + .5).astype(np.uint8)
                    expected[image.mask] = 0
                    with Image.open(BytesIO(archive.read(f'with_constrast/{png_name(image.name)}'))) as png:
                        assert png.size == (10, 10)
                        np.testing.assert_array_equal(np.asarray(png), np.flipud(expected))
                    path = f'no_contrast/{image.name if file_format == "fits" else png_name(image.name)}'
                    if file_format == 'fits':
                        with fits.open(BytesIO(archive.read(path))) as hdus:
                            np.testing.assert_array_equal(hdus[0].data, image.data)
                            np.testing.assert_array_equal(hdus['MASK'].data, image.mask)
                    else:
                        expected_off = np.floor(np.clip(image.data / 65535., 0., 1.) * 255. + .5).astype(np.uint8)
                        expected_off[image.mask] = 0
                        with Image.open(BytesIO(archive.read(path))) as png:
                            np.testing.assert_array_equal(np.asarray(png), np.flipud(expected_off))
            with Image.open(BytesIO(archive.read(f'with_constrast/{png_name(raw.name)}'))) as png:
                plain = np.asarray(png)
            with Image.open(BytesIO(archive.read(f'with_constrast/{png_name(versions[True]["Blue"].name)}'))) as png:
                corrected = np.asarray(png)
            assert plain.mean() > corrected.mean()
    for image in all_images:
        np.testing.assert_array_equal(image.data, originals[image.name][0])
        np.testing.assert_array_equal(image.mask, originals[image.name][1])


def test_bias_archive_keeps_fits_and_exports_full_resolution_contrast(tmp_path):
    values = np.tile(np.arange(511., 624.), (12, 1))
    bias = frame(values, name='master_bias_1_x21.fits')
    bias.mask[0, 0] = True
    path = tmp_path / bias.name
    path.write_bytes(to_fits(bias))
    original = path.read_bytes()
    for source in (bias, path, (bias.name, original)):
        with ZipFile(BytesIO(calibration_archive({False: {'Bias': source}}, 'fits'))) as archive:
            assert set(archive.namelist()) == {'with_constrast/', 'no_contrast/',
                'with_constrast/master_bias_1_x21.png', 'no_contrast/master_bias_1_x21.fits'}
            if source is not bias:
                assert archive.read('no_contrast/master_bias_1_x21.fits') == original
            with Image.open(BytesIO(archive.read('with_constrast/master_bias_1_x21.png'))) as png:
                assert png.size == (113, 12)
                assert np.asarray(png).min() == 0 and np.asarray(png).max() == 255
                assert np.asarray(png)[-1, 0] == 0
    assert path.read_bytes() == original
    np.testing.assert_array_equal(bias.data, values)


def test_bias_zip_png_without_contrast_matches_visible_bias_in_viewer(monkeypatch):
    import image_viewer
    values = np.tile(np.arange(534., 552.), (12, 1))
    bias = frame(values, name='master_bias_1_x21.fits')
    bias.mask[0, 0] = True
    original = bias.data.copy()
    rendered = []
    monkeypatch.setattr(image_viewer.components, 'html', lambda html, **kwargs: rendered.append(html))
    image_viewer.show(bias.display(), 1, display_limits=(500., 630.), contrast=False)
    assert 'const displayLow = 500.0, displayHigh = 630.0;' in rendered[0]
    with ZipFile(BytesIO(calibration_archive({False: {'Bias': bias}}, 'png'))) as archive:
        expected = np.floor(np.clip((values - 500.) / 130., 0., 1.) * 255. + .5).astype(np.uint8)
        expected[bias.mask] = 0
        with Image.open(BytesIO(archive.read('no_contrast/master_bias_1_x21.png'))) as png:
            pixels = np.asarray(png)
            assert png.size == (18, 12)
            np.testing.assert_array_equal(pixels, np.flipud(expected))
            assert 60. < np.median(pixels) < 110.
        with Image.open(BytesIO(archive.read('with_constrast/master_bias_1_x21.png'))) as png:
            assert not np.array_equal(np.asarray(png), pixels)
    np.testing.assert_array_equal(bias.data, original)


def test_bias_and_flat_zip_buttons_export_both_versions_independent_of_toggles(tmp_path, monkeypatch):
    import download_ui
    def finish_exports(app, name, key):
        record = app.session_state['calibration_export_cache'][name]
        for job in list(record['jobs'].values()):
            job.result(timeout=15)
        app.button(key=key).click().run()
        assert not app.exception and not app.error
        return record
    values = np.arange(144.).reshape(12, 12)
    for name, data, kind, color in [('science.fits', values + 1000., 'OBJECT', 'Blue'),
                                  ('bias1.fits', values + 500., 'BIAS', None),
                                  ('bias2.fits', values + 500., 'BIAS', None),
                                  ('flat_Blue_1.fits', values + 4000., 'FLAT', 'Blue'),
                                  ('flat_Blue_2.fits', values + 4000., 'FLAT', 'Blue')]:
        header = fits.Header({'IMAGETYP': kind, 'EXPTIME': 1.})
        if color:
            header['FILTER'] = color
        fits.writeto(tmp_path / name, data, header)
    source = f'''
from pathlib import Path
import streamlit as st
from imageProcessing import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
workflow_panel(read_frame(folder / 'science.fits'), 'source', folder)
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    assert 'calibration_export_cache' not in app.session_state
    app.button(key='workflow_bias_download_all').click().run()
    assert not app.exception
    bias_record = finish_exports(app, 'master_bias', 'workflow_bias_download_all')
    assert set(bias_record['files']) == {'fits', 'png'}
    for file_format, data in bias_record['files'].items():
        with ZipFile(BytesIO(data)) as archive:
            assert len([name for name in archive.namelist() if name.endswith('.png') and name.startswith('with_constrast/')]) == 1
    app.run()
    app.button(key='workflow_flat_download_all').click().run()
    assert not app.exception
    flat_record = finish_exports(app, 'flats', 'workflow_flat_download_all')
    first = dict(flat_record['files'])
    for data in first.values():
        with ZipFile(BytesIO(data)) as archive:
            assert 'with_constrast/master_flat_blue_1s_x2.png' in archive.namelist()
            assert 'with_constrast/corrected_flat_blue_1s_x2.png' in archive.namelist()
    app.run()
    app.button(key='workflow_flat_bias_option').click().run()
    app.button(key='workflow_science_contrast_option').click().run()
    app.button(key='workflow_flat_download_all').click().run()
    assert not app.exception
    assert app.session_state['calibration_export_cache']['flats'] is flat_record
    assert not flat_record['jobs']
    for file_format, data in flat_record['files'].items():
        assert data is first[file_format]
        with ZipFile(BytesIO(first[file_format])) as old, ZipFile(BytesIO(data)) as new:
            assert new.namelist() == old.namelist()
            for name in new.namelist():
                if name.endswith('.png'):
                    assert new.read(name) == old.read(name)


def test_science_png_and_zip_match_viewer_contrast_and_keep_result_fits():
    from imageProcessing import resolve_display_limits
    from imageProcessing import preview_png
    values = np.linspace(.004, .008, 256).reshape(16, 16)
    values[7, 8] = .2
    mask = np.zeros_like(values, dtype=bool)
    mask[0, 0] = True
    master = frame(values, fits.Header({'IMAGETYP': 'MASTER_SCIENCE', 'SCICOR': True,
                                        'FLATNORM': False, 'ALIGN': 'TRANSLATION'}), mask,
                   'master_science_hd226868_blue_30s_x4.fits')
    scientific_bytes = to_fits(master)
    low, high = resolve_display_limits(master.display(), (0., 1.), True)
    expected = np.floor(np.clip((values - low) / (high - low), 0., 1.) * 255. + .5).astype('uint8')
    expected[mask] = 0
    png_bytes = preview_png(master, contrast=True)
    with Image.open(BytesIO(png_bytes)) as png:
        np.testing.assert_array_equal(np.asarray(png), np.flipud(expected))
        assert np.asarray(png).mean() > 100.
    with ZipFile(BytesIO(image_archive({'Blue': master}, 'png', display_png=True, contrast=True))) as archive:
        assert archive.namelist() == [png_name(master.name)]
        assert archive.read(png_name(master.name)) == png_bytes
    with ZipFile(BytesIO(image_archive({'Blue': master}, 'fits', display_png=True, contrast=True))) as archive:
        assert archive.read(master.name) == scientific_bytes
    with Image.open(BytesIO(preview_png(master, contrast=False))) as png:
        expected_fixed = np.floor(np.clip(values, 0., 1.) * 255. + .5).astype('uint8')
        expected_fixed[mask] = 0
        np.testing.assert_array_equal(np.asarray(png), np.flipud(expected_fixed))
    assert to_fits(master) == scientific_bytes
