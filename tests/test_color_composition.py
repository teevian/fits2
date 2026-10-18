from io import BytesIO

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

from color_composition import generate_color
from imageProcessing import frame, translate_frame


def color_masters():
    y, x = np.indices((96, 128))
    sky = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / 10.)
              for cy, cx, peak in [(22, 30, 1000), (62, 82, 600), (40, 102, 800)])
    masters = {}
    for name, offset, scale in [('Red', (0, 0), 1.), ('Green', (3, -4), .8),
                                 ('Blue', (-2, 3), .6), ('black', (1, 2), 2.)]:
        header = fits.Header({'FILTER': name, 'OBJECT': 'target', 'SCICOR': True,
                              'FLATNORM': True, 'IMAGETYP': 'MASTER_SCIENCE',
                              'EXPTIME': 30. if name == 'black' else 60.})
        masters[name] = translate_frame(frame(50. + scale*sky, header, name=f'master_{name}.fits'), offset)
    return masters


def test_lrgb_registers_filters_masks_borders_and_preserves_masters():
    masters = color_masters()
    originals = {key: (value.data.copy(), value.header.copy()) for key, value in masters.items()}
    result = generate_color(masters, 15, use_luminance=True)
    assert result.data.shape == (96, 128, 3)
    assert result.header['COLORMOD'] == 'LRGB'
    assert result.mask.any()
    assert np.isfinite(result.data).all()
    # Scaled copies of the same stellar scene become neutral after visual balance.
    bright = (~result.mask) & (result.data[..., 0] > .05)
    np.testing.assert_allclose(result.data[..., 0][bright], result.data[..., 1][bright], atol=.02)
    for key, image in masters.items():
        np.testing.assert_array_equal(image.data, originals[key][0])
        assert image.header == originals[key][1]
    plain, contrast = result.rgb(False), result.rgb(True)
    assert plain.dtype == contrast.dtype == np.uint8
    assert not np.array_equal(plain, contrast)
    assert not plain[result.mask].any() and not contrast[result.mask].any()
    for enhanced in (False, True):
        decoded = np.asarray(Image.open(BytesIO(result.png(enhanced))))
        np.testing.assert_array_equal(decoded, np.flipud(result.rgb(enhanced)))


def test_luminance_changes_brightness_and_missing_channels_fail():
    masters = color_masters()
    rgb = generate_color(masters, 15)
    # Add real spatial signal to L, keeping stellar registration stable.
    y, x = np.indices(masters['black'].data.shape)
    masters['black'].data += 120*np.exp(-((x-50)**2 + (y-50)**2)/100.)
    lrgb = generate_color(masters, 15, use_luminance=True)
    assert not np.allclose(rgb.data[~lrgb.mask], lrgb.data[~lrgb.mask])
    del masters['Blue']
    with pytest.raises(ValueError, match='Missing science masters'):
        generate_color(masters, 15, use_luminance=True)


def test_color_viewer_uses_rgb_canvas_without_flattening_channels(monkeypatch):
    import image_viewer
    captured = []
    monkeypatch.setattr(image_viewer.components, 'html', lambda html, **kwargs: captured.append(html))
    image_viewer.show(np.zeros((12, 18, 3), dtype=np.uint8), 1)
    assert 'rows = 12, columns = 18' in captured[0]
    assert 'color = true' in captured[0]
    assert 'pixels[3*source+c]' in captured[0]


def test_matching_widths_removes_false_halos_without_erasing_star_color():
    from color_composition import match_color_planes
    y,x = np.indices((256,256))
    locations = [(45,45),(45,125),(45,210),(125,45),(125,125),(125,210),(210,45),(210,125)]
    planes=[]
    for channel, sigma in enumerate((2.,5.,4.,3.)):
        plane=np.zeros_like(x,dtype=float)
        for index,(cy,cx) in enumerate(locations):
            color = 2. if index == 7 and channel == 0 else 1.
            flux=10000*color
            plane += flux/(2*np.pi*sigma**2)*np.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma**2))
        planes.append(plane)
    rgb,mask,widths,gains=match_color_planes(planes,np.zeros_like(x,dtype=bool))
    np.testing.assert_allclose(widths,(2,5,4,3),atol=.15)
    for cy,cx in locations[:7]:
        aperture=(np.hypot(y-cy,x-cx)<10)&(~mask)
        ratio=rgb[...,0][aperture]/rgb[...,1][aperture]
        np.testing.assert_allclose(ratio,1.,atol=.06)
    cy,cx=locations[-1]
    assert 1.8 < rgb[cy,cx,0]/rgb[cy,cx,1] < 2.2


def test_sky_plane_removes_gradient_without_subtracting_compact_nebula():
    from color_composition import background_plane
    y,x=np.indices((256,256))
    sky=100+.04*x-.08*y
    nebula=30*np.exp(-((x-128)**2+(y-128)**2)/(2*15**2))
    fitted,_=background_plane(sky+nebula,np.zeros_like(x,dtype=bool))
    assert np.max(np.abs(fitted-sky)) < .5
    assert (sky+nebula-fitted)[128,128] > 29.5
