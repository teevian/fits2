from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from astropy.io import fits
from streamlit.testing.v1 import AppTest

from imageProcessing import image_archive
from imageProcessing import frame, to_fits
from imageProcessing import combine_science_groups, matching_master_flat


def make_frame(values, name, kind='OBJECT', exposure=60., filter_name=None, **metadata):
    header = fits.Header({'IMAGETYP': kind, 'EXPTIME': exposure, **metadata})
    if filter_name is not None:
        header['FILTER'] = filter_name
    return frame(values, header, name=name)


@pytest.mark.parametrize('method, expected', [('median', 200.), ('mean', 400.)])
def test_science_groups_apply_raw_bias_and_flat_without_mixing_exposures_or_targets(method, expected):
    response = np.array([[.5, 1.], [1., 1.5]])
    bias = make_frame(np.array([[10., 12.], [11., 13.]]), 'master_bias_x2.fits', 'BIAS', 0.)
    flats, sources = {}, []
    for filter_name in ('Red', 'Green', 'Blue', 'black'):
        flats[filter_name] = make_frame(response * 1000., f'corrected_flat_{filter_name}_x12.fits',
                                        'CORRECTED_FLAT', .2, filter_name, BIASCOR=True, NCOMBINE=12)
        for index, level in enumerate((100., 200., 900.)):
            # FILTER deliberately omitted: the filename must suffice.
            sources.append(make_frame(bias.data + level * response,
                                      f'hd226868_2026-10-01_21-10-56_{filter_name}_{index:04d}.fits'))
    sources += [make_frame(bias.data + level * response,
                           f'hd226868_2026-10-01_21-10-56_Red_{index+10:04d}.fits', exposure=30.)
                for index, level in enumerate((100., 300.))]
    sources += [make_frame(bias.data + level * response,
                           f'deneb_2026-10-01_21-10-56_Red_{index:04d}.fits')
                for index, level in enumerate((50., 70.))]
    sources.append(make_frame(bias.data + 400. * response,
                              'hd226868_2026-10-01_21-10-56_Green_9999.fits', exposure=10.))
    originals = [source.data.copy() for source in sources]
    flat_originals = {key: image.data.copy() for key, image in flats.items()}
    masters, inputs, skipped = combine_science_groups(sources, bias, flats, method)
    assert len(masters) == 6
    for filter_name in flats:
        key = f'hd226868 · {filter_name} · 60 s'
        image = masters[key]
        np.testing.assert_allclose(image.data, expected)
        assert image.name == f'master_science_hd226868_{filter_name.casefold()}_60s_x3.fits'
        assert image.header['FILTER'] == filter_name
        assert image.header['EXPTIME'] == 60.
        assert image.header['NCOMBINE'] == len(inputs[key]) == 3
        assert image.header['BIASCOR'] and image.header['FLATCOR']
        assert image.header['FLATFILE'] == flats[filter_name].name
    np.testing.assert_allclose(masters['hd226868 · Red · 30 s'].data, 200.)
    np.testing.assert_allclose(masters['deneb · Red · 60 s'].data, 60.)
    assert skipped == [('hd226868 · Green · 10 s', [sources[-1].name])]
    for source, original in zip(sources, originals):
        np.testing.assert_array_equal(source.data, original)
    for key, image in flats.items():
        np.testing.assert_array_equal(image.data, flat_originals[key])
    with ZipFile(BytesIO(image_archive(masters))) as archive:
        with fits.open(BytesIO(archive.read(masters['hd226868 · Blue · 60 s'].name))) as hdus:
            np.testing.assert_allclose(hdus[0].data, expected)


def test_science_flat_selection_and_invalid_inputs():
    bias = make_frame(np.ones((2, 2)), 'master_bias.fits', 'BIAS', 0.)
    reference = make_frame(np.ones((2, 2)), 'hd226868_2026-10-01_21-10-56_Blue_0099.fits')
    small = make_frame(np.ones((2, 2)), 'corrected_flat_blue_small.fits', 'FLAT', .2, 'Blue', BIASCOR=True, NCOMBINE=10)
    large = make_frame(np.ones((2, 2)), 'corrected_flat_blue_large.fits', 'FLAT', .4, 'Blue', BIASCOR=True, NCOMBINE=20)
    matching = make_frame(np.ones((2, 2)), 'corrected_flat_blue_matching.fits', 'FLAT', 60., 'Blue', BIASCOR=True, NCOMBINE=2)
    assert matching_master_flat({'small': small, 'large': large}, reference) is large
    assert matching_master_flat({'large': large, 'matching': matching}, reference) is matching
    with pytest.raises(ValueError, match='several master flats'):
        matching_master_flat({'first': large, 'second': large}, reference)
    with pytest.raises(ValueError, match='no master flat'):
        matching_master_flat({}, reference)
    with pytest.raises(ValueError, match='at least two'):
        combine_science_groups([reference], bias, {'Blue': large})
    with pytest.raises(ValueError, match='calibration frames'):
        combine_science_groups([large, large], bias, {'Blue': large})
    reference.header['BIASCOR'] = True
    with pytest.raises(ValueError, match='already been applied'):
        combine_science_groups([reference, reference], bias, {'Blue': large})


def test_science_requires_correct_align_combine_and_display_clicks_do_not_reprocess(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    response = np.tile([[.5, 1.], [1., 1.5]], (48, 64))
    y, x = np.indices(response.shape)
    stars = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / 10.)
                for cy, cx, peak in [(22, 30, 1000), (62, 82, 600), (40, 102, 800)])
    bias = np.full_like(response, 10.)
    originals = {}
    for index in range(2):
        image = make_frame(bias, f'bias{index}.fits', 'BIAS', 0.)
        (tmp_path / image.name).write_bytes(to_fits(image))
    for filter_name in ('Red', 'Green', 'Blue', 'black'):
        for index in range(2):
            image = make_frame(bias + 1000.*response, f'flat_2026-10-01_20-00-00_{filter_name}_{index:04d}.fits',
                               'FLAT', .2, filter_name)
            (tmp_path / image.name).write_bytes(to_fits(image))
            image = make_frame(bias + (100. + index * 200. + stars) * response,
                               f'hd226868_2026-10-01_21-10-56_{filter_name}_{index:04d}.fits', filter_name=filter_name)
            contents = to_fits(image)
            (tmp_path / image.name).write_bytes(contents)
            originals[image.name] = contents
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    assert not app.exception
    app.text_input(key='workflow_science_pattern').set_value('hd226868.*').run()
    assert len(app.session_state['workflow_science_frames']) == 8
    assert 'workflow_science_last' not in app.session_state
    assert app.button(key='workflow_science_correct_show').disabled
    assert app.button(key='workflow_science_align_show').disabled
    assert any('master bias and flats first' in item.value for item in app.info)
    app.button(key='workflow_flat_combine').click().run()
    assert not app.exception and not app.error
    assert app.button(key='workflow_science_combine').disabled
    assert 'workflow_science_last' not in app.session_state
    app.button(key='workflow_science_correct').click().run()
    assert app.button(key='workflow_science_combine').disabled
    app.button(key='workflow_science_align').click().run()
    assert not app.button(key='workflow_science_combine').disabled
    assert 'workflow_science_last' not in app.session_state
    app.button(key='workflow_science_combine').click().run()
    assert not app.exception and not app.error
    science = app.session_state['workflow_science_last'][1]
    assert len(science) == 4
    assert app.session_state['workflow_stage_view'] == 'science'
    for image in science.values():
        np.testing.assert_allclose(image.data[~image.mask], (200. + stars)[~image.mask], atol=1e-5)
    def unexpected_processing(*args, **kwargs):
        raise AssertionError('Viewing and contrast changes must not recompute science masters.')
    with monkeypatch.context() as patch:
        patch.setattr('science_ui.run_with_timeout', unexpected_processing)
        for key, before_label, after_label in [
                ('workflow_science_correct_show', '**Original**', '**Corrected**'),
                ('workflow_science_align_show', '**Corrected · before alignment**', '**Aligned**')]:
            app.button(key=key).click().run()
            assert not app.exception and not app.error
            assert len(app.get('imgs')) == 16
            assert [item.value for item in app.subheader if item.value != 'File'] == app.session_state['workflow_science_frames']
            text = [item.value for item in app.markdown]
            assert text.count(before_label) == text.count(after_label) == 8
            captions = [item.value for item in app.caption]
            if key == 'workflow_science_align_show':
                assert sum(item.startswith('Shift: Y ') for item in captions) == 8
            else:
                assert all(name in captions and 'corrected_science_' + name in captions for name in originals)
            app.run()
        app.button(key='workflow_science_show').click().run()
        assert not app.exception and not app.error
        assert len(app.get('imgs')) == 12
        captions = [item.value for item in app.caption]
        start = next(i for i, text in enumerate(captions) if text.startswith('One result per object'))
        captions = captions[start:]
        for master in science.values():
            assert captions.count(master.name) == 1
        for names in app.session_state['workflow_science_last'][2].values():
            assert all(name in captions for name in names)
        app.run()
        downloads = []
        patch.setattr('science_ui.archive_download_dialog',
                      lambda images, name, **kwargs: downloads.append((images, name)))
        app.button(key='workflow_science_download_all').click().run()
        assert downloads[-1] == (science, 'science_masters')
        app.button(key='workflow_science_corrected_download_all').click().run()
        assert len(downloads[-1][0]) == 8
        assert downloads[-1][1] == 'science_corrected_frames'
        app.button(key='workflow_image_name_science_master_hd226868 · Green · 60 s').click().run()
        assert not app.exception
        assert any(item.value == 'master_science_hd226868_green_60s_x2.fits' for item in app.caption)
        app.button(key='workflow_science_contrast_option').click().run()
        assert not app.exception
        app.button(key='workflow_flat_bias_option').click().run()
        assert not app.exception
        assert app.session_state['workflow_science_last'][1] is science
    app.text_input(key='workflow_science_pattern').set_value('hd226868.*_Green_.*').run()
    assert not app.exception
    assert app.button(key='workflow_science_combine').disabled
    assert not any(button.key == 'workflow_science_download_all' for button in app.button)
    assert app.button(key='workflow_science_show').disabled
    assert app.button(key='workflow_science_correct_show').disabled
    assert app.button(key='workflow_science_align_show').disabled
    app.button(key='workflow_science_correct').click().run()
    app.button(key='workflow_science_align').click().run()
    app.button(key='workflow_science_combine').click().run()
    assert len(app.session_state['workflow_science_last'][1]) == 1
    app.text_input(key='workflow_science_pattern').set_value('[').run()
    assert not app.exception
    assert any('Invalid regex' in error.value for error in app.error)
    assert len(app.session_state['workflow_science_frames']) == 2
    app.text_input(key='workflow_science_pattern').set_value('').run()
    assert not app.exception
    assert app.session_state['workflow_science_frames'] == []
    attempts = []
    def timeout(*args, **kwargs):
        attempts.append(True)
        raise TimeoutError('Timeout: operation stopped after 30 seconds.')
    app.text_input(key='workflow_science_pattern').set_value('hd226868.*_Red_.*').run()
    app.button(key='workflow_science_correct').click().run()
    app.button(key='workflow_science_align').click().run()
    with monkeypatch.context() as patch:
        patch.setattr('science_ui.run_with_timeout', timeout)
        app.button(key='workflow_science_combine').click().run()
        assert any('Timeout:' in error.value for error in app.error)
        assert not app.exception
        app.run()
        assert len(attempts) == 1
        app.button(key='workflow_science_combine').click().run()
        assert len(attempts) == 2
    app.button(key='workflow_science_combine').click().run()
    assert not app.exception and not app.error
    assert len(app.session_state['workflow_science_last'][1]) == 1
    app.button(key='workflow_science_master_remove_hd226868 · Red · 60 s').click().run()
    assert not app.exception
    assert app.session_state['workflow_science_last'][1] == {}
    app.run()
    assert app.session_state['workflow_science_last'][1] == {}
    assert {name: (tmp_path / name).read_bytes() for name in originals} == originals


def test_science_target_shortcuts_fill_regex_select_files_and_remember_other_targets(tmp_path, monkeypatch):
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    folder = tmp_path / 'images'
    folder.mkdir()
    for target, count in [('hd226868', 2), ('deneb', 3), ('M31', 2)]:
        for index in range(count):
            image = make_frame(np.ones((12, 12)),
                               f'{target}_2026-10-01_21-10-56_Blue_{index:04d}.fits', filter_name='Blue')
            (folder / image.name).write_bytes(to_fits(image))
    # A calibration frame sharing the timestamp pattern must not become a target.
    image = make_frame(np.ones((12, 12)), 'altair_2026-10-01_21-10-56_Blue_0099.fits', 'FLAT', .2, 'Blue')
    (folder / image.name).write_bytes(to_fits(image))
    path = str(Path(__file__).resolve().parents[1] / 'app.py')
    app = AppTest.from_file(path)
    app.session_state['folder_path'] = str(folder)
    app.run()
    assert app.button(key='workflow_science_target_0').label == 'hd226868'
    assert app.button(key='workflow_science_target_1').label == 'deneb'
    elements = list(app.sidebar)
    assert elements.index(app.button(key='workflow_science_target_0')) < elements.index(app.text_input(key='workflow_science_pattern'))
    app.button(key='workflow_science_target_0').click().run()
    assert not app.exception
    assert app.text_input(key='workflow_science_pattern').value == 'hd226868*'
    assert len(app.session_state['workflow_science_frames']) == 2
    app.button(key='workflow_science_target_1').click().run()
    assert not app.exception
    assert app.text_input(key='workflow_science_pattern').value == 'deneb*'
    assert len(app.session_state['workflow_science_frames']) == 3
    app.text_input(key='workflow_science_pattern').set_value('M31.*').run()
    assert not app.exception
    assert app.button(key='workflow_science_target_2').label == 'M31'
    assert workspace_settings.load_science_targets() == ['hd226868', 'deneb', 'M31']
    app.text_input(key='workflow_science_pattern').set_value('m31.*').run()
    assert app.session_state['workflow_science_targets'] == ['hd226868', 'deneb', 'M31']
    app.text_input(key='workflow_science_pattern').set_value('altair.*').run()
    assert not app.exception
    assert 'altair' not in app.session_state['workflow_science_targets']
    app.text_input(key='workflow_science_pattern').set_value('[').run()
    app.button(key='workflow_science_target_2').click().run()
    assert not app.exception and not app.error
    assert app.text_input(key='workflow_science_pattern').value == 'M31*'
    assert len(app.session_state['workflow_science_frames']) == 2
    # The custom target remains available after restarting the app.
    fresh = AppTest.from_file(path)
    fresh.session_state['folder_path'] = str(folder)
    fresh.run()
    assert not fresh.exception
    assert fresh.button(key='workflow_science_target_2').label == 'M31'


def test_science_alignment_registers_targets_across_filters_and_calibrates_before_shifting(tmp_path):
    from scipy.ndimage import shift
    from imageProcessing import read_frame
    from imageProcessing import align_science_frames
    y, x = np.indices((512, 640))
    stars = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / 18.)
                for cy, cx, peak in [(100, 140, 1000), (350, 430, 600), (230, 500, 800)])
    response = .8 + x / 2000. + y / 3000.
    response /= np.median(response)
    bias_values = 500. + (x % 13) * 30. + (y % 17) * 20.
    bias = make_frame(bias_values, 'master_bias.fits', 'BIAS', 0.)
    flats = {f: make_frame(1000. * response, f'corrected_flat_{f}.fits', 'FLAT', .2,
                           f, BIASCOR=True) for f in ('Blue', 'Green')}
    sources = []
    expected = [(0., 0.), (-3.4, 5.2), (0., 0.), (0., 0.), (-4., -3.)]
    for index, (target, filt, time, displacement) in enumerate([
            ('hd226868', 'Blue', 60., (0., 0.)),
            ('hd226868', 'Blue', 60., (3.4, -5.2)),
            ('hd226868', 'Green', 30., (0., 0.)),
            ('deneb', 'Green', 30., (0., 0.)),
            ('deneb', 'Green', 30., (4., 3.))]):
        sky = shift(stars + 50., displacement, order=1, mode='constant', cval=np.nan)
        image = make_frame(bias_values + sky * response,
                           f'{target}_2026-10-01_21-10-56_{filt}_{index:04d}.fits',
                           exposure=time, filter_name=filt)
        path = tmp_path / image.name
        path.write_bytes(to_fits(image))
        sources.append(path)
    original_bytes = [path.read_bytes() for path in sources]
    paths, offsets = align_science_frames(sources, bias, flats, 15, tmp_path / 'aligned')
    np.testing.assert_allclose(offsets, expected, atol=.25)
    for index, path in enumerate(paths):
        image = read_frame(path)
        assert image.header['BIASCOR'] and image.header['FLATCOR']
        assert image.header['ALIGN'] == 'TRANSLATION'
        assert image.header['ALNREF'] == 'corrected_science_' + sources[0 if index < 2 else 2 if index == 2 else 3].name
        valid = ~image.mask
        assert np.mean((image.data[valid] - (stars + 50.)[valid])**2) < 1.
    masters, _, _ = combine_science_groups(paths, bias, flats, aligned=True)
    assert len(masters) == 2
    for master in masters.values():
        assert master.header['ALIGN'] == 'TRANSLATION'
        assert master.header['BIASCOR'] and master.header['FLATCOR']
        assert np.mean((master.data[~master.mask] - (stars + 50.)[~master.mask])**2) < 1.
    assert [path.read_bytes() for path in sources] == original_bytes
    unrelated = make_frame(np.random.default_rng(7).normal(1000., 100., stars.shape),
                            'hd226868_2026-10-01_21-10-56_Blue_9999.fits', filter_name='Blue')
    with pytest.raises(ValueError, match='no reliable|limit|exceeds'):
        align_science_frames([sources[0], unrelated], bias, flats, 15)
    with pytest.raises(ValueError, match='structure|estrutura'):
        align_science_frames([make_frame(bias_values + response * 50., sources[0].name,
                                        filter_name='Blue')] * 2, bias, flats, 15)


def test_science_align_button_combines_registered_frames_and_invalidates_on_selection(tmp_path, monkeypatch):
    from scipy.ndimage import shift
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    y, x = np.indices((96, 128))
    stars = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / 10.)
                for cy, cx, peak in [(22, 30, 1000), (62, 82, 600), (40, 102, 800)])
    for index in range(2):
        for kind, level, exposure in [('bias', 500., 0.), ('flat', 1500., .2)]:
            image = make_frame(np.full_like(stars, level), f'{kind}_Blue_{index}.fits',
                               kind.upper(), exposure, 'Blue' if kind == 'flat' else None)
            (tmp_path / image.name).write_bytes(to_fits(image))
        image = make_frame(500. + 50. + shift(stars, (index * 3., index * -4.), order=1),
                           f'hd226868_2026-10-01_21-10-56_Blue_{index:04d}.fits', filter_name='Blue')
        (tmp_path / image.name).write_bytes(to_fits(image))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.button(key='workflow_science_target_0').click().run()
    assert app.button(key='workflow_science_align').disabled
    app.button(key='workflow_flat_combine').click().run()
    assert app.button(key='workflow_science_align').disabled
    app.button(key='workflow_science_correct').click().run()
    assert not app.button(key='workflow_science_align').disabled
    app.button(key='workflow_science_align').click().run()
    assert not app.exception and not app.error
    assert 'workflow_science_last' not in app.session_state
    from processing_jobs import run_with_timeout as actual_job
    combinations = []
    def record_combination(function, *args, **kwargs):
        combinations.append((function, args, kwargs))
        return actual_job(function, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr('science_ui.run_with_timeout', record_combination)
        app.button(key='workflow_science_combine').click().run()
    aligned = app.session_state['workflow_science_aligned']
    assert len(combinations) == 1
    assert combinations[0][0] is combine_science_groups
    assert combinations[0][1][0] == aligned[1]
    assert combinations[0][2]['aligned'] and combinations[0][2]['corrected']
    np.testing.assert_allclose(aligned[2][1], (-3., 4.), atol=.2)
    master = next(iter(app.session_state['workflow_science_last'][1].values()))
    assert master.header['ALIGN'] == 'TRANSLATION'
    valid = ~master.mask
    assert np.mean((master.data[valid] - (50. + stars)[valid])**2) < 1.
    downloads = []
    with monkeypatch.context() as patch:
        patch.setattr('download_ui.image_download_dialog', lambda image, display_png, contrast:
                      downloads.append((image, display_png, contrast)))
        patch.setattr('science_ui.archive_download_dialog', lambda images, name, **options:
                      downloads.append((images, name, options)))
        app.button(key='workflow_science_contrast_option').click().run()
        app.button(key='workflow_science_download_hd226868 · Blue · 60 s').click().run()
        assert downloads[0][0] is master
        assert downloads[0][1:] == (True, True)
        app.button(key='workflow_science_download_all').click().run()
        assert downloads[1][0] is app.session_state['workflow_science_last'][1]
        assert downloads[1][2] == {'display_png': True, 'contrast': True}
    cached = app.session_state['workflow_science_last']
    with monkeypatch.context() as patch:
        def timeout(*args, **kwargs):
            raise TimeoutError('Timeout: operation stopped after 30 seconds.')
        patch.setattr('science_ui.run_with_timeout', timeout)
        app.button(key='workflow_science_align').click().run()
        assert any('Timeout:' in item.value for item in app.error)
        app.run()
        assert app.session_state['workflow_science_last'] is cached
    app.button(key='workflow_science_align').click().run()
    assert not app.exception and not app.error
    assert not any(button.key == 'workflow_science_download_all' for button in app.button)
    app.button(key='workflow_science_combine').click().run()
    assert not app.exception and not app.error
    assert app.session_state['workflow_science_last'] is not cached
    app.text_input(key='workflow_science_pattern').set_value('deneb*').run()
    assert app.button(key='workflow_science_align').disabled
    assert not any('Combination uses these frames' in item.value for item in app.caption)


def test_science_alignment_ignores_fixed_hot_pixels_when_measuring_stars():
    from scipy.ndimage import shift
    from imageProcessing import align_science_frames
    y, x = np.indices((128, 160))
    sky = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / 10.)
              for cy, cx, peak in [(28, 34, 1000), (89, 111, 600), (49, 129, 800)])
    defects = np.zeros_like(sky)
    defects[16:112:13, 16:144:19] = 20000.
    bias = make_frame(np.full_like(sky, 500.), 'master_bias.fits', 'BIAS', 0.)
    flat = make_frame(np.full_like(sky, 1000.), 'corrected_flat_Blue.fits', 'FLAT', .2,
                      'Blue', BIASCOR=True)
    sources = [make_frame(550. + defects + shift(sky, displacement, order=1),
                          f'hd226868_2026-10-01_21-10-56_Blue_{index:04d}.fits', filter_name='Blue')
               for index, displacement in enumerate([(0., 0.), (4., -6.)])]
    aligned, offsets = align_science_frames(sources, bias, {'Blue': flat}, 15)
    np.testing.assert_allclose(offsets[1], (-4., 6.), atol=.25)
    # Registration must not erase detector pixels as a side effect.
    assert aligned[0].data.max() > 20.


def test_individual_science_correction_uses_exact_flat_values_and_keeps_originals(tmp_path):
    from imageProcessing import read_frame
    from imageProcessing import correct_science_frames
    bias = make_frame([[500., 501.], [502., 503.]], 'master_bias.fits', 'BIAS', 0.)
    flats = {
        'Green': make_frame([[100., 200.], [0., -10.]], 'corrected_flat_Green.fits',
                            'CORRECTED_FLAT', .2, 'Green', BIASCOR=True),
        'Blue': make_frame([[1000., 2000.], [3000., 4000.]], 'corrected_flat_Blue.fits',
                           'CORRECTED_FLAT', .2, 'Blue', BIASCOR=True),
    }
    sources = [make_frame([[490., 541.], [552., 563.]],
                          f'hd226868_2026-10-01_21-10-56_{color}_0001.fits', filter_name=color)
               for color in ('Green', 'Blue')]
    originals = [to_fits(image) for image in sources]
    paths = correct_science_frames(sources, bias, flats, tmp_path / 'corrected')
    corrected = [read_frame(path) for path in paths]
    np.testing.assert_allclose(corrected[0].data[:1], [[-15., 30.]])
    np.testing.assert_array_equal(corrected[0].mask, [[False, False], [True, True]])
    np.testing.assert_allclose(corrected[1].data, [[-25., 50.], [125./3., 37.5]])
    from imageProcessing import default_display_limits, display_group
    from imageProcessing import to_png
    from PIL import Image
    assert default_display_limits(display_group(tmp_path, corrected[0].name, corrected[0].header)) == (0., 65535.)
    with Image.open(BytesIO(to_png(corrected[0]))) as png:
        assert np.asarray(png)[-1, 1] == 30
    for image, source in zip(corrected, sources):
        assert image.name == 'corrected_science_' + source.name
        assert image.header['ORIGFILE'] == source.name
        assert image.header['FLATFILE'] == flats[source.header['FILTER']].name
        assert image.header['BIASCOR'] and image.header['FLATCOR'] and image.header['SCICOR']
        assert image.header['FLATNORM'] is True
    assert [to_fits(image) for image in sources] == originals
    with pytest.raises(ValueError, match='already'):
        correct_science_frames(paths, bias, flats)
    with pytest.raises(ValueError, match='master bias'):
        correct_science_frames(sources, None, flats)


def test_science_correct_lists_clickable_original_corrected_pairs_before_align(tmp_path, monkeypatch):
    from imageProcessing import read_frame
    import workspace_settings
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    response = np.tile([[800., 1000.], [1200., 1400.]], (6, 6))
    bias_values = np.full_like(response, 500.)
    originals = {}
    for index in range(2):
        bias = make_frame(bias_values, f'bias{index}.fits', 'BIAS', 0.)
        flat = make_frame(bias_values + response, f'flat_Blue_{index}.fits', 'FLAT', .2, 'Blue')
        science = make_frame(bias_values + (index + .25) * response,
                             f'hd226868_2026-10-01_21-10-56_Blue_{index:04d}.fits', filter_name='Blue')
        for image in (bias, flat, science):
            contents = to_fits(image)
            (tmp_path / image.name).write_bytes(contents)
            originals[image.name] = contents
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.button(key='workflow_science_target_0').click().run()
    assert not app.button(key='workflow_science_correct').disabled
    assert app.button(key='workflow_science_align').disabled
    app.button(key='workflow_science_correct').click().run()
    assert not app.exception and not app.error
    assert not app.button(key='workflow_science_align').disabled
    correction = app.session_state['workflow_science_corrected']
    assert len(correction[1]) == 2
    pair_key, pair_items = app.session_state['science_pairs_previews']
    assert [item['name'] for item in pair_items] == ['0:0', '0:1', '1:0', '1:1']
    assert all(item['image'].startswith('data:image/png;base64,') for item in pair_items)
    assert pair_items[0]['label'] in correction[1]
    assert pair_items[1]['label'] == correction[1][pair_items[0]['label']].name
    assert not app.button(key='workflow_science_correct_show').disabled
    for index, (original_name, path) in enumerate(correction[1].items()):
        np.testing.assert_allclose(read_frame(path).data, (index + .25) * np.median(response))
        assert app.button(key=f'workflow_image_name_science_{index}').label == original_name
        assert app.button(key=f'workflow_image_name_science_corrected_{index}').label == path.name
        assert path.name == f'corrected_science_{original_name}'
    elements = list(app)
    correct = elements.index(app.button(key='workflow_science_correct'))
    left = elements.index(app.button(key='workflow_image_name_science_0'))
    right = elements.index(app.button(key='workflow_image_name_science_corrected_0'))
    align = elements.index(app.button(key='workflow_science_align'))
    assert correct < left < right < align
    assert any(item.value == 'Original' for item in app.caption)
    assert any(item.value == 'Corrected' for item in app.caption)
    def unexpected(*args, **kwargs):
        raise AssertionError('Viewing the individual pair must reuse the corrected data.')
    with monkeypatch.context() as patch:
        patch.setattr('science_ui.run_with_timeout', unexpected)
        patch.setattr('science_picker.science_pair_thumbnails', unexpected)
        import science_picker
        picker = science_picker.image_picker
        for side, expected in [('1', .25 * np.median(response)), ('0', bias_values + .25 * response)]:
            patch.setattr('science_picker.image_picker', lambda **kwargs: (
                {'action': 'view', 'name': '0:' + side, 'event': 'pair-' + side}
                if kwargs['key'] == pair_key else picker(**kwargs)))
            app.run()
            assert not app.exception
            np.testing.assert_allclose(app.session_state['workflow_file_view'][1].data, expected)
        patch.setattr('science_picker.image_picker', picker)
        popups = []
        patch.setattr('science_ui.show_science_pairs', lambda pairs, stage: popups.append((pairs, stage)))
        app.button(key='workflow_science_correct_show').click().run()
        assert len(popups) == 1 and len(popups[0][0]) == 2 and popups[0][1] == 'correction'
        app.button(key='workflow_image_name_science_corrected_0').click().run()
        assert not app.exception
        np.testing.assert_allclose(app.session_state['workflow_file_view'][1].data, .25 * np.median(response))
        app.button(key='workflow_image_name_science_0').click().run()
        assert not app.exception
        np.testing.assert_allclose(app.session_state['workflow_file_view'][1].data, bias_values + .25 * response)
        app.button(key='workflow_science_contrast_option').click().run()
        assert not app.exception
        assert app.session_state['workflow_science_corrected'] is correction
    # A failed correction reports its timeout without replacing good outputs.
    with monkeypatch.context() as patch:
        patch.setattr('science_ui.run_with_timeout', lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError('Timeout: operation stopped after 30 seconds.')))
        app.button(key='workflow_science_correct').click().run()
        assert not app.exception
        assert any('Timeout:' in item.value for item in app.error)
        assert app.session_state['workflow_science_corrected'] is correction
    app.text_input(key='workflow_science_pattern').set_value('deneb*').run()
    assert app.button(key='workflow_science_align').disabled
    assert not any(button.key == 'workflow_image_name_science_corrected_0' for button in app.button)
    assert {name: (tmp_path / name).read_bytes() for name in originals} == originals


def test_science_alignment_handles_large_nonwrapping_pointing_offsets():
    from scipy.ndimage import shift
    from imageProcessing import align_science_frames
    y, x = np.indices((256, 320))
    rng = np.random.default_rng(31)
    sky = np.zeros_like(x, dtype=float)
    for cy in (30, 70, 110, 150, 190, 230):
        for cx in (35, 95, 155, 215, 275):
            cyj, cxj = cy + rng.uniform(-10, 10), cx + rng.uniform(-10, 10)
            sky += rng.uniform(500, 1500) * np.exp(-((x-cxj)**2 + (y-cyj)**2) / 10.)
    sources = [make_frame(550. + shift(sky, offset, order=1),
                          f'hd226868_2026-10-01_21-10-56_Blue_{index:04d}.fits', filter_name='Blue')
               for index, offset in enumerate([(0., 0.), (145., 20.)])]
    bias = make_frame(np.full_like(sky, 500.), 'master_bias.fits', 'BIAS', 0.)
    flat = make_frame(np.full_like(sky, 1000.), 'corrected_flat_Blue.fits', 'FLAT', .2,
                      'Blue', BIASCOR=True)
    aligned, offsets = align_science_frames(sources, bias, {'Blue': flat}, 200)
    np.testing.assert_allclose(offsets[1], (-145., -20.), atol=.3)
    valid = ~aligned[1].mask
    assert np.mean((aligned[1].data[valid] - (50. + sky)[valid])**2) < .5


def test_correct_prepares_four_filters_and_results_keep_separate_exposures(tmp_path, monkeypatch):
    import workspace_settings
    from imageProcessing import read_frame
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    for index in range(2):
        bias = make_frame(np.full((12, 12), 10.), f'bias{index}.fits', 'BIAS', 0.)
        (tmp_path / bias.name).write_bytes(to_fits(bias))
    for filter_name in ('Red', 'Green', 'Blue', 'black'):
        for index in range(2):
            flat = make_frame(np.full((12, 12), 110.), f'flat_{filter_name}_{index}.fits',
                              'FLAT', .2, filter_name)
            (tmp_path / flat.name).write_bytes(to_fits(flat))
            for exposure in (30., 60.):
                science = make_frame(np.full((12, 12), 10. + exposure * 100.),
                                     f'hd226868_{int(exposure)}_{filter_name}_{index}.fits',
                                     exposure=exposure, filter_name=filter_name, OBJECT='hd226868')
                (tmp_path / science.name).write_bytes(to_fits(science))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.text_input(key='workflow_science_pattern').set_value('hd226868.*').run()
    assert not app.button(key='workflow_science_correct').disabled
    app.button(key='workflow_science_correct').click().run()
    assert not app.exception and not app.error
    corrected = list(app.session_state['workflow_science_corrected'][1].values())
    assert len(corrected) == 16
    bias = app.session_state['workflow_bias_last'][1]
    flats = app.session_state['workflow_flat_versions'][1][True]
    masters, inputs, skipped = combine_science_groups(corrected, bias, flats, corrected=True)
    assert len(masters) == 8 and not skipped
    assert {(m.header['FILTER'], m.header['EXPTIME']) for m in masters.values()} == {
        (f, e) for f in ('Red', 'Green', 'Blue', 'black') for e in (30., 60.)}
    for key, master in masters.items():
        assert len(inputs[key]) == master.header['NCOMBINE'] == 2
        np.testing.assert_allclose(master.data, master.header['EXPTIME'] * 100.)
    # Viewing a different source must keep the calibration and corrected files valid.
    app.button(key='workflow_image_name_science_1').click().run()
    assert not app.exception and not app.error
    assert not app.button(key='workflow_science_align').disabled


def test_science_choice_limits_correction_alignment_and_combination(tmp_path, monkeypatch):
    import workspace_settings
    from frame_selection import science_file_number, science_file_range
    monkeypatch.setattr(workspace_settings, 'SETTINGS_FILE', tmp_path / 'settings.json')
    assert science_file_number('hd226868_2026-10-01_21-45-57_Blue_0118.fits') == 118
    assert science_file_number('target_Blue_0061.fits.gz') == 61
    assert science_file_number('target_0062.fit.fz') == 62
    assert science_file_number('target_2026-10-01.fits') is None
    assert science_file_range(['x_0118.fits', 'x_0061.fits', 'x_0060.fits'], 61, 118) == ['x_0118.fits', 'x_0061.fits']
    y, x = np.indices((64, 64))
    stars = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / 6.)
                for cy, cx, peak in [(15, 20, 1000), (40, 42, 800), (20, 48, 600)])
    for index in range(2):
        for image in (make_frame(np.full((64, 64), 10.), f'bias{index}.fits', 'BIAS', 0.),
                      make_frame(np.full((64, 64), 110.), f'flat_Blue_{index}.fits', 'FLAT', .2, 'Blue')):
            (tmp_path / image.name).write_bytes(to_fits(image))
    for number in (60, 61, 62, 118, 119):
        image = make_frame(110. + stars, f'hd226868_2026-10-01_21-45-57_Blue_{number:04d}.fits', filter_name='Blue')
        (tmp_path / image.name).write_bytes(to_fits(image))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    app.text_input(key='workflow_science_pattern').set_value('hd226868.*').run()
    names = app.session_state['workflow_science_frames']
    chosen = [name for name in names if science_file_number(name) in (61, 62, 118)]
    monkeypatch.setattr('science_picker.image_picker', lambda **kwargs: ({'action': 'change', 'selected': chosen, 'event': repr(chosen)} if kwargs['key'].startswith('science_inline_') else None))
    app.run()
    app.button(key='workflow_science_correct').click().run()
    assert not app.exception and not app.error
    selected = app.session_state['workflow_science_corrected'][1]
    assert {science_file_number(name) for name in selected} == {61, 62, 118}
    app.button(key='workflow_science_align').click().run()
    assert not app.exception and not app.error
    assert len(app.session_state['workflow_science_aligned'][1]) == 3
    app.button(key='workflow_science_combine').click().run()
    assert not app.exception and not app.error
    assert next(iter(app.session_state['workflow_science_last'][1].values())).header['NCOMBINE'] == 3
    chosen = [name for name in names if science_file_number(name) in (61, 62)]
    app.run()
    assert app.button(key='workflow_science_align').disabled
    assert app.button(key='workflow_science_combine').disabled
    chosen = []
    app.run()
    assert app.button(key='workflow_science_correct').disabled
    chosen = names
    app.run()
    assert not app.exception and not app.error
    assert app.session_state['workflow_science_frames'] == names
