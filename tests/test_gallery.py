from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from astropy.io import fits
from PIL import Image
from streamlit.testing.v1 import AppTest

from gallery_ui import fits_thumbnail


def test_all_link_opens_gallery_for_current_folder_and_handles_bad_file(tmp_path):
    folder = tmp_path / 'images with spaces'
    folder.mkdir()
    for index in range(4):
        fits.writeto(folder / f'image_{index}.fits', np.arange(48.).reshape(6, 8))
    app_path = str(Path(__file__).resolve().parents[1] / 'app.py')
    app = AppTest.from_file(app_path, default_timeout=20)
    app.session_state['folder_path'] = str(folder)
    app.run()
    assert not app.exception
    link = next(item for item in app.get('link_button') if item.proto.label == 'all')
    query = parse_qs(urlsplit(link.proto.url).query)
    assert query == {'page': ['all'], 'folder': [str(folder)]}
    (folder / 'broken.fits').write_text('invalid FITS')
    gallery = AppTest.from_file(app_path, default_timeout=20)
    gallery.query_params.update({key: values[0] for key, values in query.items()})
    gallery.run()
    assert not gallery.exception
    assert len(gallery.get('imgs')) == 4
    assert len(gallery.warning) == 1
    captions = [item.value for item in gallery.caption]
    assert all(path.name in captions for path in folder.iterdir())
    assert not gallery.tabs
    gallery.text_input(key='gallery_pattern').set_value(r'^IMAGE_[13]\.fits$').run()
    assert not gallery.exception and not gallery.warning
    assert len(gallery.get('imgs')) == 2
    captions = [item.value for item in gallery.caption]
    assert 'image_1.fits' in captions and 'image_3.fits' in captions
    assert 'image_0.fits' not in captions
    gallery.text_input(key='gallery_pattern').set_value('[').run()
    assert not gallery.exception and 'Invalid regex:' in gallery.error[0].value
    assert not gallery.get('imgs')
    gallery.text_input(key='gallery_pattern').set_value('nothing_matches').run()
    assert not gallery.error and not gallery.get('imgs')
    assert gallery.info[0].value == 'No FITS files match this filter.'
    gallery.text_input(key='gallery_pattern').set_value('').run()
    assert not gallery.exception and len(gallery.get('imgs')) == 4


def test_gallery_preview_is_visible_bounded_and_does_not_modify_fits(tmp_path):
    path = tmp_path / 'faint.fits'
    values = np.tile(np.linspace(500., 501., 800), (600, 1))
    fits.writeto(path, values)
    original = path.read_bytes()
    stat = path.stat()
    preview = fits_thumbnail(path, stat.st_mtime_ns, stat.st_size)
    with Image.open(BytesIO(preview)) as image:
        assert image.size == (480, 320)
        assert np.asarray(image).max() == 255
        assert np.asarray(image).min() == 0
    assert path.read_bytes() == original
    fits.writeto(path, np.flipud(values.T), overwrite=True)
    updated = path.stat()
    assert fits_thumbnail(path, updated.st_mtime_ns, updated.st_size) != preview
