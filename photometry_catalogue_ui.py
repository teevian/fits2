"""Compact, editable catalogue assistance. Network requests only after a name changes or Search."""
from hashlib import sha256
from urllib.parse import quote

import streamlit as st

from photometry_catalogues import lookup_reference, filter_band, seed_suggestion, BANDS


def catalogue_panel(context):
    prefix = 'photo_catalogue_' + context
    # A form submits the name and button together. Blurring the textbox must not
    # start a second fragment rerun while a catalogue request is still running.
    with st.form(prefix+'form', border=False):
        name_column, search_column = st.columns([5, 1], vertical_alignment='bottom')
        name = name_column.text_input('Nome da estrela de referência', value='V1674 Cyg',
                                      placeholder='Ex.: V1674 Cyg, HD 172167', key=prefix+'name').strip()
        search = search_column.form_submit_button('Pesquisar')
    report = st.session_state.get(prefix+'report')
    if search:
        with st.spinner('A consultar SIMBAD e VizieR…'):
            try:
                report = lookup_reference(name)
            except Exception as exc:
                # Keep unexpected network/parser failures in the dialog as well.
                # A report for a different star must never populate this name.
                report = None
                st.session_state[prefix+'error'] = f'Não foi possível concluir a pesquisa: {exc}'
            else:
                st.session_state.pop(prefix+'error', None)
        st.session_state[prefix+'report'] = report
        st.session_state[prefix+'queried'] = name
    if st.session_state.get(prefix+'error'):
        st.error(st.session_state[prefix+'error'])
    if not name or name != st.session_state.get(prefix+'queried'):
        report = None
    if report:
        st.caption(' · '.join(f"{item['service']}: {item['count']} valores ({item['status']})" for item in report['services']))
        objects = report.get('objects', [])
        if objects:
            obj = objects[0]
            st.caption(f"Identificação: {obj['name']} · tipo SIMBAD: {obj['object_type'] or 'não indicado'}. Confirma que é a estrela clicada e uma referência estável.")
        if not report['records']:
            st.info('Sem magnitudes utilizáveis nesta consulta. Os campos continuam disponíveis para entrada manual.')
        available = [band for band in BANDS if any(r['band'] == band for r in report['records'])]
        st.caption('Bandas disponíveis: ' + ', '.join(available) + '. Os filtros identificados são preenchidos automaticamente.')
        if objects and objects[0].get('object_type') in ('SB*', 'EB*', 'V*'):
            st.warning('O catálogo classifica este objeto como binário ou variável. As magnitudes tabeladas podem depender da época; confirma a sua adequação como referência.')
        with st.expander('Fontes e valores encontrados'):
            st.caption('SIMBAD e VizieR são serviços do CDS; podem citar as mesmas medições. Não são duas confirmações independentes. '
                       'A pesquisa VizieR usa um raio de 3″ e pode não encontrar estrelas com posições antigas ou movimento próprio elevado.')
            for item in report['services']:
                st.markdown(f"[{item['service']} · {item['status']}]({item['url']})")
                if item.get('error'):
                    st.caption(item['error'])
            if report['records']:
                # Plain HTML inherits our white table style; no dark-themed canvas.
                import pandas as pd
                values = [{'Fonte': r['service'], 'Catálogo': r['catalogue'], 'Objeto': r['object_id'],
                           'Banda': r['band'], 'Mag.': r['magnitude'], 'Erro': r['error'],
                           'Publicação': r.get('bibcode') or 'ver catálogo'} for r in report['records']]
                st.markdown('<div class="photometry-table-wrap">'+pd.DataFrame(values).fillna('—').to_html(index=False, escape=True, classes='photometry-results', border=0)+'</div>', unsafe_allow_html=True)
            st.caption('Consulta UTC: ' + report['retrieved_utc'])
    return name, report


def suggestion_controls(prefix, filter_name, reference_name, report):
    """Runs before numeric widgets so new suggestions can populate their session values."""
    if not report:
        if st.session_state.get(prefix+'catalogue_original'):
            seed_suggestion(st.session_state, prefix, ('cleared', reference_name), None, reference_name)
        return None
    version = sha256((report['query']+report['retrieved_utc']).encode()).hexdigest()[:12]
    bands = [band for band in BANDS if any(r['band'] == band for r in report['records'])]
    default = filter_band(filter_name)
    a, b = st.columns([1, 3])
    options = ['Escolher banda'] + bands
    band = a.selectbox('Banda do catálogo', options, index=options.index(default) if default in options else 0,
                       key=prefix+'catalogue_band_'+version)
    candidates = [r for r in report['records'] if r['band'] == band]
    by_id = {r['id']: r for r in candidates}
    safe = [r for r in candidates if not r.get('ambiguous') and r.get('quality') not in ('D', 'E')]
    ids = [''] + [r['id'] for r in candidates]
    def label(value):
        if not value:
            return 'Sem sugestão selecionada'
        row = by_id[value]
        return f"{row['service']} · {row['catalogue']} · {row['magnitude']:.4f} mag · {row['object_id']}" + (' [associação a confirmar]' if row.get('ambiguous') else '')
    selected = b.selectbox('Valor sugerido', ids, index=ids.index(safe[0]['id']) if safe else 0,
                           format_func=label, key=prefix+'suggestion_'+version+band)
    suggestion = by_id.get(selected)
    seed_suggestion(st.session_state, prefix, (report['query'], version, band, selected), suggestion, reference_name)
    if default and default not in bands:
        st.info(f'Sem magnitude tabelada na banda {default} nesta pesquisa. O campo permanece vazio; não foi substituído por outra banda.')
    if band == 'Escolher banda' and report['records']:
        st.caption(f'Filtro «{filter_name}»: escolhe a banda de catálogo. RGB não identifica automaticamente Johnson B/V/R.')
    if suggestion:
        source = f"[{suggestion['service']} · {suggestion['catalogue']}]({suggestion['url']})"
        bibcode = suggestion.get('bibcode')
        if bibcode:
            source += f" · [{bibcode}](https://ui.adsabs.harvard.edu/abs/{quote(bibcode, safe='')}/abstract)"
        if suggestion.get('reference_code'):
            source += ' · referência interna: ' + suggestion['reference_code']
        st.caption('Fonte: ' + source)
        st.caption(' · '.join(suggestion['notes'])) if suggestion['notes'] else None
        if suggestion.get('ambiguous'):
            st.warning('Associação ambígua: verifica objeto e coordenadas no catálogo antes de usar este valor.')
        if len(candidates) > 1 and max(r['magnitude'] for r in candidates)-min(r['magnitude'] for r in candidates) > .1:
            st.caption('As fontes diferem mais de 0,1 mag. Verifica época, banda e variabilidade; não foi calculada uma média.')
    return suggestion
