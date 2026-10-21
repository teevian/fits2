from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pytest
from astropy.io import fits
from streamlit.testing.v1 import AppTest

from imageProcessing import frame
from photometry import measure, stellar_parameters, stellar_uncertainties
from photometry_ui import diagnostic_figures, export_report


def star(**metadata):
    y, x = np.mgrid[:81, :81]
    data = 100 + 1000 * np.exp(-((x-40)**2 + (y-40)**2) / 8)
    return frame(data, fits.Header({'EXPTIME': 10., 'BUNIT': 'adu', **metadata}), name='star.fits')


def test_flux_rate_magnitude_and_single_exposure_variance():
    image = star()
    r = measure(image, 40, 40, 7, 15, 20, gain=2., saturation=50000.)
    yy, xx = np.mgrid[:81, :81]
    aperture = (xx-40)**2 + (yy-40)**2 <= 49
    expected = np.sum(image.data[aperture] - 100)
    assert r['flux_adu'] == pytest.approx(expected, rel=1e-8)
    assert r['rate_adu_s'] == pytest.approx(expected/10)
    assert r['m_inst'] == pytest.approx(-2.5*np.log10(expected/10))
    assert r['flux_error_adu']**2 == pytest.approx(expected/2, rel=1e-6)
    assert r['sigma_m'] == pytest.approx(2.5/np.log(10)/r['snr'])
    assert r['fit']['fwhm_px'] == pytest.approx(4.71, rel=.15)


@pytest.mark.parametrize('metadata', [{'NCOMBINE': 4}, {'ALNSUBPX': True}, {'COMBINED': True}, {'ALIGN': 'TRANSLATION'}])
def test_no_fabricated_stack_errors(metadata):
    r = measure(star(**metadata), 40, 40, gain=2)
    assert r['m_inst'] is not None
    assert r['sigma_m'] is None and r['snr'] is None
    r = measure(star(**metadata), 40, 40, flux_error=40)
    assert r['flux_error_adu'] == 40
    assert r['snr'] == r['flux_adu']/40


def test_invalid_masks_edges_units_saturation_and_exposure():
    image = star()
    with pytest.raises(ValueError, match='saturada'):
        measure(image, 40, 40, saturation=500)
    with pytest.raises(ValueError, match='sai da imagem'):
        measure(image, 4, 4)
    image.mask[40, 40] = True
    with pytest.raises(ValueError, match='mascarados'):
        measure(image, 40, 40)
    image = star(BUNIT='adu/s')
    with pytest.raises(ValueError, match='ADU'):
        measure(image, 40, 40)
    image = star()
    del image.header['EXPTIME']
    assert measure(image, 40, 40)['m_inst'] is None
    image = frame(np.full((81, 81), 100.), fits.Header({'EXPTIME': 10.}))
    assert measure(image, 40, 40)['m_inst'] is None


def test_background_outlier_rejection_and_no_input_mutation():
    image = star()
    image.data[40, 54] = 1e6
    original = image.data.copy()
    r = measure(image, 40, 40)
    assert r['n_sky_rejected'] > 0
    assert r['sky_adu'] == pytest.approx(100., abs=.01)
    np.testing.assert_array_equal(image.data, original)


def test_physical_relations_and_restrictions():
    p = stellar_parameters(5.48, 4.83, 0., 10., 0., -.09, 1.)
    assert p['temperature_K'] == pytest.approx(5778, rel=.01)
    assert p['luminosity_solar'] == pytest.approx(1.)
    assert p['lifetime_years'] == pytest.approx(1e10)
    assert stellar_parameters(5.48, 4.83, 0., 10., 0., -.09)['lifetime_years'] is None
    with pytest.raises(ValueError):
        stellar_parameters(0., 5., 0., 10., 0., 0.)
    with pytest.raises(ValueError):
        stellar_parameters(5.48, 4.83, 0., 0., 0., 0.)


def test_physical_uncertainties_follow_distance_and_temperature_derivatives():
    p = stellar_parameters(5.48, 4.83, 0., 100., 0., -.09, 1.)
    e = stellar_uncertainties(p, 100., .01, .02, .03, 2., .04, .05, mass=1., sigma_mass=.1)
    assert e['sigma_color_bv0'] == pytest.approx(np.sqrt(.01**2 + .02**2 + .03**2))
    derivative = (stellar_parameters(5.48001, 4.83, 0., 100., 0., -.09)['temperature_K']
                  - stellar_parameters(5.47999, 4.83, 0., 100., 0., -.09)['temperature_K']) / .00002
    assert e['sigma_temperature_K'] == pytest.approx(abs(derivative)*e['sigma_color_bv0'])
    distance_only = stellar_uncertainties(p, 100., 0., 0., 0., 2., 0., 0.)
    assert distance_only['sigma_luminosity_solar']/p['luminosity_solar'] == pytest.approx(.04)
    assert e['sigma_lifetime_years'] > .1*p['lifetime_years']


@pytest.mark.parametrize('kwargs', [{'gain': -1}, {'flux_error': float('nan')}, {'saturation': 0},
                                   {'radius': 12, 'inner': 10}, {'x': float('nan')}])
def test_invalid_parameters_are_rejected(kwargs):
    params = dict(x=40, y=40)
    params.update(kwargs)
    with pytest.raises(ValueError):
        measure(star(), **params)


def test_export_is_self_contained_and_white():
    r = measure(star(NCOMBINE=3), 40, 40)
    figures = diagnostic_figures(r)
    assert len(figures) == 6
    assert all(f.layout.paper_bgcolor == 'white' and f.layout.plot_bgcolor == 'white' for f in figures)
    with ZipFile(BytesIO(export_report({'star': r}, {'test': True}, figures))) as archive:
        report = json.loads(archive.read('provenance.json'))
        assert report['measurements'][0]['sigma_m'] is None
        assert '0_radial.csv' in archive.namelist()
        assert 'plotly.js' in archive.read('plots.html').decode()


def test_dialog_opens_measures_and_invalidates_changed_parameters():
    source = '''
import numpy as np
from astropy.io import fits
from imageProcessing import frame
from photometry_ui import analysis_button, analysis_dialog
y, x = np.mgrid[:81, :81]
image = frame(100 + 1000*np.exp(-((x-40)**2+(y-40)**2)/8), fits.Header({'EXPTIME': 10., 'NCOMBINE': 3}), name='star.fits')
analysis_button({'star': image})
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    app.button(key='workflow_science_analysis').click().run()
    assert not app.exception
    # AppTest reruns the full script rather than the dialog fragment. Exercise
    # its body on every test rerun while testing opening separately above.
    app = AppTest.from_string(source.replace("analysis_button({'star': image})",
                              "analysis_dialog.__wrapped__({'star': image})"), default_timeout=30).run()
    next(b for b in app.button if b.label == 'Calcular análises').click().run()
    assert not app.exception, app.exception
    assert not app.error
    assert app.session_state['photo_results'][1]['star']['m_inst'] is not None
    next(n for n in app.number_input if n.label == 'Abertura [px]').set_value(8.).run()
    assert not app.exception
    assert any('Configura os parâmetros' in i.value for i in app.info)


def test_physical_ui_requires_calibration_and_propagates_when_supplied():
    app = AppTest.from_string('''
import streamlit as st
from photometry_ui import physical_panel
if 'initialized' not in st.session_state:
    st.session_state.update(initialized=True, democonfirm=True, demodistance=10.,
        demoref='Synthetic solar validation, BC=-0.09', demobc=-.09, demoms=True,
        demomass=1., demopropagate=True, demosigma_distance=.1, demosigma_mass=.05)
result, plots = physical_panel({'B': {'m_std': 5.48, 'sigma_m_std': .01},
                                'V': {'m_std': 4.83, 'sigma_m_std': .01}}, 'demo')
st.session_state.physical_output = result
st.session_state.plot_count = len(plots)
''', default_timeout=30).run()
    assert not app.exception, app.exception
    p = app.session_state['physical_output']
    assert p['luminosity_solar'] == pytest.approx(1.)
    assert p['sigma_temperature_K'] > 0
    assert p['sigma_lifetime_years'] > 0
    assert app.session_state['plot_count'] == 2
    app.checkbox(key='democonfirm').uncheck().run()
    assert app.session_state['physical_output'] is None


def field_pair(separation=75, **header):
    y, x = np.mgrid[:90, :160]
    data = 100 + 2000*np.exp(-((x-35)**2+(y-40)**2)/8) + 1000*np.exp(-((x-35-separation)**2+(y-40)**2)/8)
    return frame(data, fits.Header({'EXPTIME': 10., 'FILTER': 'V', **header}), name='pair.fits')


def test_same_field_calibration_matches_flux_ratio_and_errors():
    from photometry import calibrate_with_reference
    image = field_pair()
    target = measure(image, 35, 40, gain=2)
    r = calibrate_with_reference(image, target, 110, 40, 13., .02, gain=2,
                                 correction=.1, correction_error=.01, passband_confirmed=True)
    assert r['m_std'] == pytest.approx(13-2.5*np.log10(2)+.1, abs=1e-6)
    assert r['derived_zero_point'] == pytest.approx(13-r['reference']['m_inst'])
    assert r['sigma_m_std']**2 == pytest.approx(target['sigma_m']**2+r['reference']['sigma_m']**2+.02**2+.01**2)
    assert r['reference']['radius'] == target['radius']


def test_same_field_exposure_cancels_and_passband_is_not_assumed():
    from photometry import calibrate_with_reference
    image = field_pair()
    del image.header['EXPTIME']
    target = measure(image, 35, 40)
    r = calibrate_with_reference(image, target, 110, 40, 13.)
    assert r['m_calibrated'] == pytest.approx(13-2.5*np.log10(2), abs=1e-6)
    assert r['derived_zero_point'] is None
    assert r['m_std'] is None and r['sigma_m_calibrated'] is None


def test_shared_sky_suppresses_independent_error_and_contamination_is_rejected():
    from photometry import calibrate_with_reference
    image = field_pair(separation=28)
    target = measure(image, 35, 40, gain=2)
    r = calibrate_with_reference(image, target, 63, 40, 13., .02, gain=2)
    assert r['sigma_m_calibrated'] is None
    assert any('sobrepostos' in flag for flag in r['calibration_flags'])
    with pytest.raises(ValueError, match='próximos'):
        calibrate_with_reference(image, target, 36, 40, 13.)
    with pytest.raises(ValueError):
        calibrate_with_reference(image, target, 63, 40, float('nan'))
    with pytest.raises(ValueError):
        calibrate_with_reference(image, target, 63, 40, 13., -.1)


def test_reference_mask_saturation_and_stack_errors_are_respected():
    from photometry import calibrate_with_reference
    image = field_pair(NCOMBINE=3)
    target = measure(image, 35, 40, flux_error=100)
    with pytest.raises(ValueError, match='saturada'):
        calibrate_with_reference(image, target, 110, 40, 13., saturation=1000)
    r = calibrate_with_reference(image, target, 110, 40, 13., .02, reference_flux_error=50)
    assert r['sigma_m_calibrated'] > .02
    r = calibrate_with_reference(image, target, 110, 40, 13., reference_flux_error=50)
    assert r['sigma_m_calibrated'] is None  # Unknown catalogue error is not zero.
    image.mask[40, 110] = True
    with pytest.raises(ValueError, match='mascarados'):
        calibrate_with_reference(image, target, 110, 40, 13.)


def test_reference_measurements_and_profiles_are_exported():
    from photometry import calibrate_with_reference
    image = field_pair()
    target = measure(image, 35, 40)
    target.update(calibrate_with_reference(image, target, 110, 40, 13.))
    with ZipFile(BytesIO(export_report({'pair': target}, {}, []))) as archive:
        report = json.loads(archive.read('provenance.json'))
        assert report['measurements'][0]['reference']['x'] == 110
        assert report['measurements'][0]['reference_magnitude'] == 13.
        assert '0_reference_radial.csv' in archive.namelist()
        assert '0_reference_growth.csv' in archive.namelist()


def test_picker_validates_coordinates_and_preserves_image_orientation():
    from photometry_picker import selection_event, picker_preview
    from base64 import b64decode
    from PIL import Image
    assert selection_event({'event': 'one', 'role': 'reference', 'x': 110, 'y': 40}, (90,160)) == ('reference', 110., 40.)
    assert selection_event({'event': 'one', 'role': 'reference', 'x': 160, 'y': 40}, (90,160)) is None
    assert selection_event({'event': 'one', 'role': 'other', 'x': 1, 'y': 1}, (90,160)) is None
    assert selection_event({'event': 'one', 'role': 'target', 'x': float('nan'), 'y': 1}, (90,160)) is None
    image = field_pair()
    original = image.data.copy()
    preview = np.asarray(Image.open(BytesIO(b64decode(picker_preview(image, 'test-pair').split(',')[1]))))
    assert preview.shape == original.shape
    assert preview[40, 110] > preview[70, 110]
    np.testing.assert_array_equal(image.data, original)


def test_reference_click_calibrates_and_catalogue_changes_invalidate_results():
    source = '''
import numpy as np
import streamlit as st
from astropy.io import fits
from imageProcessing import frame
import photometry_picker
from photometry_ui import analysis_dialog
if 'image' not in st.session_state:
    y,x=np.mgrid[:90,:160]
    st.session_state.image=frame(100+2000*np.exp(-((x-35)**2+(y-40)**2)/8)+1000*np.exp(-((x-110)**2+(y-40)**2)/8),fits.Header({'EXPTIME':10.}),name='pair.fits')
photometry_picker.star_picker=lambda **kwargs: st.session_state.get('click')
analysis_dialog.__wrapped__({'pair': st.session_state.image})
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    app.session_state['click'] = dict(role='reference', x=110, y=40, event='ref1')
    app.run()
    assert not app.exception, app.exception
    assert next(n for n in app.number_input if n.label == 'Referência · x [pixel]').value == 110
    next(n for n in app.number_input if n.label == 'Magnitude da referência').set_value(13.).run()
    next(b for b in app.button if b.label == 'Calcular análises').click().run()
    assert not app.exception, app.exception
    result = app.session_state['photo_results'][1]['pair']
    assert result['m_calibrated'] == pytest.approx(13-2.5*np.log10(2), abs=1e-6)
    assert result['m_std'] is None
    next(n for n in app.number_input if n.label == 'Magnitude da referência').set_value(14.).run()
    assert any('Configura os parâmetros' in i.value for i in app.info)
    next(b for b in app.button if b.label == 'Calcular análises').click().run()
    assert app.session_state['photo_results'][1]['pair']['m_calibrated'] == pytest.approx(result['m_calibrated']+1)
    # A new reference click invalidates the old measurements too.
    app.session_state['click'] = dict(role='reference', x=111, y=40, event='ref2')
    app.run()
    assert any('Configura os parâmetros' in i.value for i in app.info)
    next(n for n in app.number_input if n.label == 'Referência · x [pixel]').set_value(110.).run()
    assert next(n for n in app.number_input if n.label == 'Referência · x [pixel]').value == 110.
    # A previously handled browser event must not overwrite manual refinement.
    next(n for n in app.number_input if n.label == 'Referência · y [pixel]').set_value(40.).run()
    next(t for t in app.text_input if t.label == 'Sistema da magnitude').set_value('Johnson V / Vega').run()
    next(c for c in app.checkbox if c.label.startswith('Banda padrão /')).check().run()
    next(b for b in app.button if b.label == 'Calcular análises').click().run()
    assert not app.exception
    assert app.session_state['photo_results'][1]['pair']['m_std'] == pytest.approx(14-2.5*np.log10(2), abs=1e-6)
