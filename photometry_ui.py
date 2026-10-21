"""Scientific diagnostics in a white Streamlit dialog, with reproducible exports."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from session_recovery import checkpoint

from photometry import measure, moffat, stellar_parameters, stellar_uncertainties, calibrate_with_reference
from processing_ui import fingerprint, load
from photometry_picker import select_stars
from photometry_catalogue_ui import catalogue_panel, suggestion_controls
from photometry_style import STYLE

REFERENCES = {
    'AAVSO: differential photometry': 'https://www.aavso.org/sites/default/files/CCDPhotometryGuide_3.pdf',
    'Aperture methodology': 'https://photutils.readthedocs.io/en/stable/user_guide/aperture.html',
    'Ballesteros (2012)': 'https://arxiv.org/abs/1201.1809',
    'IAU bolometric scale': 'https://arxiv.org/abs/1510.06262',
    'Main-sequence scaling': 'https://www.astro.princeton.edu/ast203/Lectures/Lecture_11.html',
}


def results_table(values):
    table = pd.DataFrame(values).fillna('—').to_html(index=False, na_rep='—', float_format=lambda x: f'{x:.6g}',
                                         classes='photometry-results', border=0, escape=True)
    st.markdown('<div style="overflow-x:auto;background:white;color:#17202a">' + table + '</div>',
                unsafe_allow_html=True)


def white_figure(title, xlabel, ylabel):
    fig = go.Figure()
    fig.update_layout(template='plotly_white', title=title, paper_bgcolor='white',
                      plot_bgcolor='white', font=dict(color='#17202a'),
                      xaxis_title=xlabel, yaxis_title=ylabel, height=400)
    return fig


def diagnostic_figures(result):
    r = result
    profile = white_figure('Perfil radial e Moffat', 'Raio [pixel]', 'Média acima do fundo [ADU/pixel]')
    radial = r['radial']
    profile.add_scatter(x=radial[:, 0], y=radial[:, 1], mode='markers', name='Perfil',
                        error_y=dict(array=radial[:, 2], visible=True))
    residual = white_figure('Resíduos do ajuste descritivo', 'Raio [pixel]', 'Observado − modelo [ADU/pixel]')
    if r['fit']:
        p = r['fit']
        model = moffat(radial[:, 0], p['amplitude'], p['alpha'], p['beta'])
        profile.add_scatter(x=radial[:, 0], y=model, name='Moffat')
        residual.add_scatter(x=radial[:, 0], y=radial[:, 1] - model, mode='markers', name='Resíduos')
    residual.add_hline(y=0)
    growth = white_figure('Curva de crescimento', 'Raio [pixel]', 'Fluxo acumulado acima do fundo [ADU]')
    growth.add_scatter(x=r['growth'][:, 0], y=r['growth'][:, 1], name='Fluxo')
    growth.add_vline(x=r['radius'], line_dash='dash')
    hist = white_figure('Fundo local após rejeição 3σ', 'Contagem [ADU/pixel]', 'Número de píxeis')
    hist.add_histogram(x=r['sky_sample'], nbinsx=50, name='Anel')
    pixels = white_figure('Distribuição de contagens no recorte', 'Contagem [ADU/pixel]', 'Número de píxeis')
    pixels.add_histogram(x=r['cutout'][~r['cutout_mask']], nbinsx=80, name='Recorte')
    aperture = white_figure('Mapa de abertura e anel de fundo', 'x [pixel; origem 0]', 'y [pixel; origem 0]')
    x0, y0 = r['origin']
    cut = np.where(r['cutout_mask'], np.nan, r['cutout'])
    limits = np.nanpercentile(cut, [5, 99])
    aperture.add_heatmap(z=cut, x=np.arange(cut.shape[1]) + x0, y=np.arange(cut.shape[0]) + y0,
                         colorscale='Greys', zmin=limits[0], zmax=limits[1], colorbar_title='ADU')
    for radius in (r['radius'], r['inner'], r['outer']):
        aperture.add_shape(type='circle', x0=r['x']-radius, x1=r['x']+radius,
                           y0=r['y']-radius, y1=r['y']+radius, line_color='#c0392b')
    aperture.update_yaxes(scaleanchor='x', scaleratio=1)
    return [aperture, profile, residual, growth, hist, pixels]


def export_report(results, configs, figures, physical=None):
    """Self-contained plots (bundled Plotly), raw numeric profiles and provenance."""
    output = BytesIO()
    records = []
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for i, (name, result) in enumerate(results.items()):
            def scalar_record(measurement):
                return {key: scalar_record(value) if key == 'reference' else value
                        for key, value in measurement.items()
                        if key not in ('radial', 'growth', 'sky_sample', 'cutout', 'cutout_mask', 'origin')}
            scalar = scalar_record(result)
            records.append(dict(image=name, **scalar))
            archive.writestr(f'{i}_radial.csv', pd.DataFrame(result['radial'], columns=['radius_px', 'mean_adu', 'scatter_sem_adu']).to_csv(index=False))
            archive.writestr(f'{i}_growth.csv', pd.DataFrame(result['growth'], columns=['radius_px', 'flux_adu']).to_csv(index=False))
            if 'reference' in result:
                archive.writestr(f'{i}_reference_radial.csv', pd.DataFrame(result['reference']['radial'], columns=['radius_px', 'mean_adu', 'scatter_sem_adu']).to_csv(index=False))
                archive.writestr(f'{i}_reference_growth.csv', pd.DataFrame(result['reference']['growth'], columns=['radius_px', 'flux_adu']).to_csv(index=False))
        archive.writestr('measurements.csv', pd.DataFrame(records).to_csv(index=False))
        archive.writestr('provenance.json', json.dumps(dict(version=2, configuration=configs,
                          measurements=records, physical=physical, references=REFERENCES), indent=2, allow_nan=False))
        html = '<html><meta charset="utf-8"><body style="background:white;color:black"><h1>Diagnósticos fotométricos</h1>'
        for i, fig in enumerate(figures):
            html += fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
        archive.writestr('plots.html', html + '</body></html>')
    return output.getvalue()


def physical_panel(results, key):
    st.markdown('**Temperatura, diagrama H–R e escala de vida estelar**')
    st.caption('Seleciona duas medições da mesma estrela em Johnson B e V, calibradas no sistema Vega. '
               'Red/Green/Blue de uma câmara não equivalem automaticamente a Johnson B/V. '
               'Uma estrela produz um ponto H–R; uma população exige um catálogo de fontes identificadas.')
    calibrated = [name for name, r in results.items() if r.get('m_std') is not None]
    if len(calibrated) < 2:
        st.info('São necessárias pelo menos duas magnitudes calibradas para estimar a cor e a temperatura.')
        return None, []
    bkey = st.selectbox('Medição Johnson B', calibrated, key=key+'b')
    vkey = st.selectbox('Medição Johnson V', calibrated, index=1, key=key+'v')
    confirm = st.checkbox('Confirmo a mesma estrela isolada, Johnson B/V Vega e aplicabilidade da aproximação de cor', key=key+'confirm')
    st.caption('Para supergigantes, binárias, estrelas peculiares ou emissão não estelar, usar modelos atmosféricos/SED e modelos evolutivos próprios. '
               'A relação de cor é apenas uma estimativa; estes resultados não incluem sistemáticos do modelo.')
    distance = st.number_input('Distância independente [pc; 0 = desconhecida]', min_value=0., value=0., key=key+'distance')
    ebv = st.number_input('E(B−V) [mag]', min_value=0., value=0., key=key+'ebv')
    av = st.number_input('A_V [mag]', min_value=0., value=0., key=key+'av')
    bc = st.number_input('Correção bolométrica BC_V [mag]', value=0., key=key+'bc')
    provenance = st.text_input('Referência e justificação de distância, extinção e BC_V (incluindo valores nulos)', key=key+'ref')
    main_sequence = st.checkbox('Classificação independente confirma sequência principal', key=key+'ms')
    mass = st.number_input('Massa independente [M☉; 0 = desconhecida]', min_value=0., value=0., key=key+'mass')
    with st.expander('Propagação de incertezas físicas'):
        propagate = st.checkbox('Fornecer erros 1σ e assumir entradas independentes', key=key+'propagate')
        st.caption('A_V e E(B−V), pontos zero e BC podem estar correlacionados. '
                   'Se estiverem, esta aproximação não substitui um modelo conjunto. '
                   'As coordenadas H–R partilham V; as barras são marginais. '
                   '0 significa parâmetro tratado como exato, não erro desconhecido.')
        uncertainties = {}
        for field, label in [('sigma_distance', 'σ distância [pc]'), ('sigma_ebv', 'σ E(B−V) [mag]'),
                             ('sigma_av', 'σ A_V [mag]'), ('sigma_bc', 'σ BC_V [mag]'),
                             ('sigma_mass', 'σ massa [M☉]')]:
            uncertainties[field] = st.number_input(label, min_value=0., value=0., key=key+field)
    if not (confirm and distance > 0 and provenance.strip() and bkey != vkey):
        st.info('Preenche e confirma os dados físicos para desbloquear o ponto H–R. Não se infere idade a partir de uma imagem.')
        return None, []
    try:
        physical = stellar_parameters(results[bkey]['m_std'], results[vkey]['m_std'], ebv, distance, av, bc,
                                      mass if main_sequence and mass > 0 else None)
    except ValueError as exc:
        st.error(str(exc))
        return None, []
    physical['assumptions'] = dict(B=bkey, V=vkey, distance_pc=distance, ebv=ebv, av=av, bc_v=bc,
                                   mass_solar=mass, main_sequence=main_sequence, reference=provenance)
    if propagate:
        b_error, v_error = results[bkey]['sigma_m_std'], results[vkey]['sigma_m_std']
        if b_error is None or v_error is None:
            st.warning('Faltam erros fotométricos B/V: propagação física indisponível.')
        else:
            physical.update(stellar_uncertainties(physical, distance, b_error, v_error,
                                                  mass=mass if main_sequence and mass > 0 else None,
                                                  **uncertainties))
            physical['assumptions']['uncertainties'] = uncertainties
    results_table([{k: v for k, v in physical.items() if k != 'assumptions'}])
    st.warning('Estimativas condicionais: mesmo com propagação, a dispersão da relação de cor e os sistemáticos do modelo não estão incluídos. '
               'Erros grandes ou correlacionados exigem modelação conjunta antes de publicação. '
               'τ ≈ 10¹⁰ (M/M☉)/(L/L☉) anos é uma escala de vida na sequência principal, não a idade nem o tempo restante.')
    hr = white_figure('Hertzsprung–Russell · estimativa condicional', 'Temperatura de cor [K]', 'Luminosidade bolométrica [L☉]')
    hr.add_scatter(x=[physical['temperature_K']], y=[physical['luminosity_solar']], mode='markers', name='Estrela selecionada',
                   error_x=dict(array=[physical.get('sigma_temperature_K', 0)]),
                   error_y=dict(array=[physical.get('sigma_luminosity_solar', 0)]))
    hr.update_xaxes(type='log', autorange='reversed')
    hr.update_yaxes(type='log')
    cmd = white_figure('Diagrama cor–magnitude', '(B−V)₀ [mag]', 'M_V [mag]')
    cmd.add_scatter(x=[physical['color_bv0']], y=[physical['M_V']], mode='markers', name='Estrela selecionada',
                    error_x=dict(array=[physical.get('sigma_color_bv0', 0)]),
                    error_y=dict(array=[physical.get('sigma_M_V', 0)]))
    cmd.update_yaxes(autorange='reversed')
    for fig in (hr, cmd):
        st.plotly_chart(fig, use_container_width=True, theme=None)
    return physical, [hr, cmd]


@st.dialog('Análise astrofísica observacional', width='large')
@checkpoint
def analysis_dialog(masters, corrected=None):
    st.markdown(STYLE, unsafe_allow_html=True)
    st.caption('Escolhe a referência, confirma a banda e seleciona as estrelas na imagem. Todos os valores podem ser editados.')
    a, b, c = st.columns([1.1, 1.8, 1.3])
    mode = a.selectbox('Dados', ['Masters finais por filtro', 'Exposições corrigidas individuais'], key='photo_data_mode')
    sources = masters if mode == 'Masters finais por filtro' else (corrected or {})
    if not sources:
        st.info('Não há exposições corrigidas disponíveis.')
        return
    selected = b.multiselect('Imagens', list(sources), default=list(sources), key='photo_selected_images')
    calibration_mode = c.selectbox('Calibração', ['Referência no mesmo FOV', 'Ponto zero manual', 'Apenas instrumental'], key='photo_calibration_mode')
    if not selected:
        return
    images = {name: load(sources[name]) for name in selected}
    identity = sha256(repr([(n, fingerprint(sources[n])) for n in selected]).encode()).hexdigest()[:16]
    reference_name, catalogue_report = '', None
    if calibration_mode == 'Referência no mesmo FOV':
        reference_name, catalogue_report = catalogue_panel(identity)
    with st.expander('Abertura e fundo'):
        a, b, c = st.columns(3)
        radius = a.number_input('Abertura [px]', min_value=1., max_value=200., value=6., key='photo_radius')
        inner = b.number_input('Fundo interior [px]', min_value=2., max_value=300., value=10., key='photo_inner')
        outer = c.number_input('Fundo exterior [px]', min_value=3., max_value=400., value=16., key='photo_outer')
        st.caption('Usa regiões sem vizinhos. Os mesmos raios são aplicados ao alvo e à referência.')
    rows = []
    for i, (name, image) in enumerate(images.items()):
        if not np.isfinite(image.display()).any():
            st.error(f'{name}: sem píxeis válidos.')
            return
        image_key = sha256(repr((name, fingerprint(sources[name]))).encode()).hexdigest()[:16]
        is_reference = calibration_mode == 'Referência no mesmo FOV'
        row = dict(image=name)
        with st.container(border=True):
            st.markdown(f'**{name}**')
            suggestion = None
            if is_reference:
                suggestion = suggestion_controls(image_key, image.header.get('FILTER', ''), reference_name, catalogue_report)
                a, b, _ = st.columns([1, 1, 2])
                row['ref_mag'] = a.number_input('Magnitude da referência', value=None, key=image_key+'ref_mag', format='%.4f', help='Magnitude tabelada na banda selecionada; editável.')
                row['ref_mag_error'] = b.number_input('Erro ± [mag]', min_value=0., value=None, key=image_key+'ref_mag_error', format='%.4f', help='Vazio = desconhecido. Zero = tratado como exato.')
                row['reference_id'] = reference_name
                row['catalogue_suggestion'] = suggestion
                if suggestion:
                    edited = (row['ref_mag'] != suggestion['magnitude'] or row['ref_mag_error'] != suggestion['error'])
                    st.caption('Valor editado por ti; a fonte original fica registada.' if edited else 'Preenchido pelo catálogo. Podes alterar os valores acima.')
            with st.expander('Selecionar alvo e referência' if is_reference else 'Selecionar alvo', expanded=i == 0):
                row.update(select_stars(image, image_key, radius, inner, outer, allow_reference=is_reference))
            with st.expander('Opções avançadas · ' + name):
                a, b, c = st.columns(3)
                row['gain'] = a.number_input('Ganho [e⁻/ADU]', min_value=0., value=0., key=image_key+'gain')
                row['saturation'] = b.number_input('Limite linear [ADU]', min_value=0., value=0., key=image_key+'saturation')
                row['sigma_flux'] = c.number_input('Erro fluxo alvo [ADU]', min_value=0., value=0., key=image_key+'sigma_flux')
                st.caption('0 = desconhecido. Erros externos devem incluir fonte, fundo e covariância do processamento.')
                if is_reference:
                    a, b = st.columns(2)
                    row['catalogue_band'] = a.text_input('Sistema da magnitude', value=None, placeholder='Ex.: Johnson V / Vega', key=image_key+'band') or ''
                    row['reference_flux_error'] = b.number_input('Erro fluxo referência [ADU]', min_value=0., value=0., key=image_key+'ref_flux_error')
                    row['passband_confirmed'] = st.checkbox('Banda padrão / transformação validada', key=image_key+'passband', help='Necessário para usar a magnitude no painel H–R. Selecionar um catálogo não valida o filtro instrumental.')
                    a, b = st.columns(2)
                    row['correction'] = a.number_input('Correção diferencial [mag]', value=0., key=image_key+'diff_correction', help='−k ΔX + c Δcor + diferença de correção de abertura.')
                    row['correction_error'] = b.number_input('Erro correção [mag]', min_value=0., value=0., key=image_key+'diff_error')
                elif calibration_mode == 'Ponto zero manual':
                    row['calibrated'] = st.checkbox('Aplicar ponto zero', key=image_key+'calibrated')
                    a, b = st.columns(2)
                    row['Z_eff'] = a.number_input('Ponto zero efetivo', value=0., key=image_key+'Z_eff')
                    row['sigma_Z'] = b.number_input('Erro ponto zero', min_value=0., value=0., key=image_key+'sigma_Z')
            if suggestion:
                row['catalogue_edited'] = (row['ref_mag'] != suggestion['magnitude'] or row['ref_mag_error'] != suggestion['error'] or row['catalogue_band'] != suggestion['system'])
        rows.append(row)
    configs = pd.DataFrame(rows)
    with st.expander('Notas da calibração'):
        cal_ref = st.text_input('Referência bibliográfica / justificação das correções', key='photo_calibration_notes')
        st.caption('Fotometria de píxeis inteiros e fundo com clipping 3σ. Sem correção automática de abertura ou de vizinhos.')
    run_signature = repr((identity, configs.to_json(), calibration_mode, radius, inner, outer, cal_ref, reference_name))
    if st.button('Calcular análises', type='primary'):
        results, errors = {}, []
        with st.spinner('A medir os FITS lineares…'):
            for config in configs.to_dict('records'):
                name = config['image']
                try:
                    for field in ('gain', 'saturation', 'sigma_flux') + (('sigma_Z',) if calibration_mode == 'Ponto zero manual' else ()):
                        if not np.isfinite(config[field]) or config[field] < 0:
                            raise ValueError(f'{field} deve ser finito e não negativo.')
                    r = measure(images[name], config['x'], config['y'], radius, inner, outer,
                                gain=config['gain'] or None, saturation=config['saturation'] or None,
                                flux_error=config['sigma_flux'] or None)
                    r['m_std'], r['sigma_m_std'] = None, None
                    if calibration_mode == 'Ponto zero manual' and config['calibrated']:
                        if not cal_ref.strip() or not np.isfinite(config['Z_eff']):
                            raise ValueError('Calibração exige referência/sistema e Z_eff finito.')
                        if r['m_inst'] is not None:
                            r['m_std'] = r['m_inst'] + config['Z_eff']
                            if r['sigma_m'] is not None:
                                r['sigma_m_std'] = float(np.hypot(r['sigma_m'], config['sigma_Z']))
                    if calibration_mode == 'Referência no mesmo FOV':
                        needed = [config.get(k) for k in ('ref_x', 'ref_y', 'ref_mag')]
                        if any(value is None or not np.isfinite(value) for value in needed):
                            r['flags'].append('Seleciona a referência e indica a magnitude tabelada para obter a calibração diferencial.')
                        else:
                            try:
                                sigma_catalogue = config.get('ref_mag_error')
                                if sigma_catalogue is not None and pd.isna(sigma_catalogue):
                                    sigma_catalogue = None
                                if config['passband_confirmed'] and not config['catalogue_band'].strip():
                                    raise ValueError('Indica a banda e o sistema da magnitude tabelada para confirmar a calibração padrão.')
                                differential = calibrate_with_reference(images[name], r, *needed,
                                    magnitude_error=sigma_catalogue, gain=config['gain'] or None,
                                    saturation=config['saturation'] or None,
                                    reference_flux_error=config['reference_flux_error'] or None,
                                    correction=config['correction'], correction_error=config['correction_error'],
                                    passband_confirmed=config['passband_confirmed'])
                                r.update(differential)
                                r['flags'].extend(differential['calibration_flags'])
                                if not config['reference_id'].strip():
                                    r['flags'].append('Identificação da referência não registada; preencher para rastreabilidade.')
                            except (ValueError, TypeError) as exc:
                                r['flags'].append(f'Calibração diferencial rejeitada: {exc}')
                    results[name] = r
                except (ValueError, TypeError) as exc:
                    errors.append(f'{name}: {exc}')
        st.session_state['photo_results'] = (run_signature, results, errors)
    cached = st.session_state.get('photo_results')
    if not cached or cached[0] != run_signature:
        st.info('Configura os parâmetros e calcula. Os resultados anteriores não se aplicam a parâmetros alterados.')
        return
    _, results, errors = cached
    for error in errors:
        st.error(error)
    if not results:
        return
    figures = []
    for name, r in results.items():
        st.markdown(f'**{name}**')
        if r.get('m_calibrated') is not None:
            error_text = (f" ± {r['sigma_m_calibrated']:.4f}" if r['sigma_m_calibrated'] is not None
                          else ' (incerteza total indisponível)')
            st.markdown(f"**Magnitude do alvo pela referência: {r['m_calibrated']:.4f}{error_text} mag**")
        results_table([{k: v for k, v in r.items() if k in (
            'flux_adu', 'flux_error_adu', 'rate_adu_s', 'm_inst', 'sigma_m', 'm_std', 'sigma_m_std',
            'snr', 'sky_adu', 'sky_std_adu', 'n_aperture', 'n_sky', 'n_sky_rejected',
            'm_calibrated', 'sigma_m_calibrated', 'derived_zero_point', 'differential_mag')}])
        for flag in r['flags']:
            st.warning(flag)
        if r['fit']:
            st.caption(f"FWHM descritiva: {r['fit']['fwhm_px']:.3f} pixel; β = {r['fit']['beta']:.3f}. Não equivale diretamente ao seeing.")
        st.caption('Barras do perfil: dispersão/√N dentro de cada anel; não incluem covariância nem erro comum do fundo. '
                   'Moffat: ajuste descritivo não ponderado aos anéis, sem incerteza de FWHM validada.')
        plots = diagnostic_figures(r)
        for i, fig in enumerate(plots):
            fig.update_layout(title=f'{name} · {fig.layout.title.text}')
            st.plotly_chart(fig, use_container_width=True, theme=None, key=f'{identity}_{name}_{i}',
                            config={'toImageButtonOptions': {'format': 'svg'}})
        figures.extend(plots)
        if 'reference' in r:
            reference = r['reference']
            with st.expander(f'Medição e gráficos da estrela de referência · {name}'):
                results_table([{k: reference[k] for k in ('x', 'y', 'flux_adu', 'flux_error_adu', 'm_inst', 'sigma_m', 'snr', 'sky_adu')}])
                st.caption(f"Magnitude tabelada: {r['reference_magnitude']:.4f}; erro tabelado: {r['reference_magnitude_error']}. Mesma abertura do alvo.")
                for flag in reference['flags']:
                    st.warning(flag)
                for i, fig in enumerate(diagnostic_figures(reference)):
                    fig.update_layout(title=f'{name} · Referência · {fig.layout.title.text}')
                    st.plotly_chart(fig, use_container_width=True, theme=None, key=f'{identity}_{name}_ref_{i}',
                                    config={'toImageButtonOptions': {'format': 'svg'}})
                    figures.append(fig)
    physical, physical_figures = physical_panel(results, identity)
    figures.extend(physical_figures)
    provenance = dict(rows=json.loads(configs.to_json(orient='records')), calibration_mode=calibration_mode, aperture_px=radius, inner_px=inner, outer_px=outer,
                      calibration_reference=cal_ref, catalogue_lookup=catalogue_report, mode=mode,
                      headers={name: str(image.header) for name, image in images.items()},
                      fingerprints={name: repr(fingerprint(sources[name])) for name in selected})
    st.download_button('Descarregar relatório científico (ZIP: CSV, JSON e gráficos HTML)',
                       export_report(results, provenance, figures, physical), 'photometry_report.zip', 'application/zip')
    with st.expander('Método e referências'):
        st.latex(r'm_T=m_R-2.5\log_{10}(F_T/F_R)+\Delta_{corr},\quad Z_R=m_R-m_{inst,R}')
        st.latex(r'\sigma_{m_T}^2\simeq\sigma_{m_{inst,T}}^2+\sigma_{m_{inst,R}}^2+\sigma_{m_R}^2+\sigma_{corr}^2')
        st.latex(r'F=\sum_A C_p-n\hat b,\quad q=F/t,\quad m_{inst}=-2.5\log_{10}(q/[1\,ADU\,s^{-1}])')
        st.latex(r'\sigma_F^2\simeq F/g+n(1+n/m)s_b^2+\sigma_{cal}^2,\quad \sigma_m\simeq(2.5/\ln10)/(S/N)')
        st.caption('O ruído de leitura presente no fundo não é somado novamente. Masters não recebem um ganho efetivo inventado. '
                   'Sem calibração de fluxo e resposta espectral, ADU/s não são fluxo energético. '
                   'O relatório é uma análise exploratória rastreável; a validação de fontes, calibração e sistemáticos é necessária para publicação.')
        for title, url in REFERENCES.items():
            st.markdown(f'[{title}]({url})')


def analysis_button(masters, corrected=None):
    if st.button('Análise astrofísica', key='workflow_science_analysis', icon=':material/analytics:'):
        analysis_dialog(masters, corrected)
