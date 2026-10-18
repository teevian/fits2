from io import BytesIO
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from imageProcessing import export_crop, statistics, display_values


def test_crop_preserves_world_coordinates_and_values():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10., 20.]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crval = [180., 45.]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    data = np.arange(600).reshape(20, 30)
    raw = export_crop(data, wcs.to_header(), 3, 15, 4, 18, 10, 2)
    with fits.open(BytesIO(raw), checksum=True) as result:
        np.testing.assert_allclose(result[0].data, (data[4:18,3:15]-10)*2)
        np.testing.assert_allclose(WCS(result[0].header).all_pix2world([[0,0]],0), wcs.all_pix2world([[3,4]],0))
        assert result[0].verify_checksum() == 1


def test_unsigned_fits_not_scaled_twice():
    buffer = BytesIO()
    original = np.array([[0,32768],[50000,65535]], dtype=np.uint16)
    fits.PrimaryHDU(original).writeto(buffer)
    buffer.seek(0)
    with fits.open(buffer, memmap=False) as source:
        header = source[0].header.copy()
        raw = export_crop(source[0].data, header, 0,2,0,2)
    with fits.open(BytesIO(raw)) as result:
        np.testing.assert_array_equal(result[0].data, original)


def test_cube_and_nonfinite():
    data = np.arange(60.).reshape(3,4,5)
    with fits.open(BytesIO(export_crop(data, fits.Header(),1,4,1,3))) as result:
        np.testing.assert_array_equal(result[0].data, data[:,1:3,1:4])
    assert statistics([1, 3, np.nan, np.inf])['media'] == 2
    assert statistics([np.nan]) == {}
    for scale in ['Linear', 'Asinh', 'Logarítmica', 'Raiz quadrada']:
        np.testing.assert_allclose(display_values([0,10],0,10,scale), [0,1])
