from pathlib import Path
import json
from copy import deepcopy
from urllib.parse import parse_qs, urlparse

import pytest
from streamlit.testing.v1 import AppTest

from photometry_catalogues import (parse_simbad, parse_vizier, lookup_reference,
                                    filter_band, seed_suggestion, simbad_url)

FIXTURES = Path(__file__).with_name('fixtures')


def test_real_responses_preserve_sources_and_do_not_invent_errors():
    simbad, objects = parse_simbad((FIXTURES/'vega_simbad.json').read_bytes(), 'simbad-query')
    vizier = parse_vizier((FIXTURES/'vega_vizier.xml').read_bytes(), 'vizier-query')
    assert objects[0]['name'] == '* alf Lyr'
    v = next(r for r in simbad if r['band']=='V')
    assert v['magnitude'] == .03 and v['error'] is None
    assert v['bibcode'] == '2002yCat.2237....0D'
    second = next(r for r in vizier if r['catalogue']=='II/7A/catalog' and r['band']=='V')
    assert second['magnitude'] == pytest.approx(.03)
    assert second['reference_code']=='8029' and second['bibcode']=='1978A&AS...34..477M'
    assert not second['ambiguous']  # Repeated measurements of one star, not five stars.
    derived = next(r for r in vizier if r['catalogue']=='II/168/ubvmeans' and r['band']=='B')
    assert derived['error'] is None and any('Derivada' in note for note in derived['notes'])
    assert all(r['query_url'] for r in simbad+vizier)


def test_both_services_are_queried_and_partial_failure_remains_usable():
    requested=[]
    def getter(url):
        requested.append(url)
        if 'sim-tap' in url:
            raise TimeoutError('unavailable')
        return (FIXTURES/'vega_vizier.xml').read_bytes()
    report=lookup_reference('Vega', getter)
    assert len(requested)==2
    assert report['services'][0]['status']=='indisponível'
    assert report['services'][1]['status']=='ok'
    assert report['records'] and all(r['service']=='VizieR' for r in report['records'])
    json.dumps(report,allow_nan=False)


def test_apass_direct_errors_and_ambiguous_associations():
    xml=b'''<VOTABLE version="1.3"><RESOURCE name="II/336"><TABLE name="II/336/apass9">
<FIELD name="RAJ2000" datatype="double"/><FIELD name="DEJ2000" datatype="double"/>
<FIELD name="Vmag" datatype="double"/><FIELD name="e_Vmag" datatype="double"/>
<DATA><TABLEDATA><TR><TD>10.0</TD><TD>20.0</TD><TD>12.3</TD><TD>0.02</TD></TR>
<TR><TD>10.0001</TD><TD>20.0</TD><TD>13.0</TD><TD>0.03</TD></TR></TABLEDATA></DATA>
</TABLE></RESOURCE></VOTABLE>'''
    records=parse_vizier(xml,'query')
    assert len(records)==2 and all(r['ambiguous'] for r in records)
    assert records[0]['error']==.02
    assert records[0]['system']=='Johnson V / Vega'


def test_query_escaping_and_explicit_band_mapping():
    query=parse_qs(urlparse(simbad_url("O'Brien")).query)['query'][0]
    assert "i.id='O''Brien'" in query
    assert filter_band('V')=='V'
    assert filter_band('Johnson B')=='B'
    assert filter_band('r')=="r'" and filter_band('R')=='R'
    assert all(filter_band(name) is None for name in ['Red','Green','Blue','black','Clear','Rc'])
    with pytest.raises(ValueError):lookup_reference('')


def test_autofill_is_once_and_preserves_manual_edits():
    state={}
    suggestion={'magnitude':12.3,'error':.02,'system':'Johnson V / Vega'}
    assert seed_suggestion(state,'star',('Vega','V'),suggestion,'Vega')
    state['starref_mag']=12.4
    assert not seed_suggestion(state,'star',('Vega','V'),suggestion,'Vega')
    assert state['starref_mag']==12.4
    assert state['starcatalogue_original']['magnitude']==12.3
    assert seed_suggestion(state,'star',('Other','V'),None,'Other')
    assert state['starref_mag'] is None and state['starref_mag_error'] is None


UI_SOURCE='''
import streamlit as st
import numpy as np
from astropy.io import fits
from imageProcessing import frame
from photometry_ui import analysis_dialog
if 'image' not in st.session_state:
    y,x=np.mgrid[:90,:160]
    st.session_state.image=frame(100+2000*np.exp(-((x-35)**2+(y-40)**2)/8)+1000*np.exp(-((x-110)**2+(y-40)**2)/8),fits.Header({'EXPTIME':10.,'FILTER':FILTER_NAME}),name='pair.fits')
analysis_dialog.__wrapped__({'pair':st.session_state.image})
'''


def fake_report(name):
    return dict(query=name,retrieved_utc='2026-10-20T00:00:00+00:00',objects=[],
                services=[dict(service=s,count=1,status='ok',url='https://example.org') for s in ('SIMBAD','VizieR')],
                records=[dict(id=str(i),service=s,catalogue='test',band='V',magnitude=.03 if name=='Vega' else .77,
                              error=None,system='Johnson V / Vega',url='https://example.org',object_id=name,
                              bibcode='',notes=[],ambiguous=False)
                         for i,s in enumerate(('SIMBAD','VizieR'))])


def test_ui_autofills_retains_edits_changes_star_and_tracks_original(monkeypatch):
    calls=[]
    def lookup(name):
        calls.append(name)
        return fake_report(name)
    monkeypatch.setattr('photometry_catalogue_ui.lookup_reference',lookup)
    app=AppTest.from_string(UI_SOURCE.replace('FILTER_NAME',repr('V')),default_timeout=30).run()
    next(t for t in app.text_input if t.label=='Nome da estrela de referência').set_value('Vega')
    next(b for b in app.button if b.label=='Pesquisar').click().run()
    assert not app.exception,app.exception
    assert next(n for n in app.number_input if n.label=='Magnitude da referência').value==.03
    next(n for n in app.number_input if n.label=='Magnitude da referência').set_value(.11).run()
    next(n for n in app.number_input if n.label=='Abertura [px]').set_value(7.).run()
    assert next(n for n in app.number_input if n.label=='Magnitude da referência').value==.11
    assert calls==['Vega']
    assert any('Valor editado' in c.value for c in app.caption)
    next(t for t in app.text_input if t.label=='Nome da estrela de referência').set_value('Altair')
    next(b for b in app.button if b.label=='Pesquisar').click().run()
    assert not app.exception,app.exception
    assert next(n for n in app.number_input if n.label=='Magnitude da referência').value==.77
    assert calls==['Vega','Altair']


def test_rgb_requires_explicit_catalogue_band(monkeypatch):
    monkeypatch.setattr('photometry_catalogue_ui.lookup_reference',fake_report)
    app=AppTest.from_string(UI_SOURCE.replace('FILTER_NAME',repr('Green')),default_timeout=30).run()
    next(t for t in app.text_input if t.label=='Nome da estrela de referência').set_value('Vega')
    next(b for b in app.button if b.label=='Pesquisar').click().run()
    assert not app.exception
    assert next(n for n in app.number_input if n.label=='Magnitude da referência').value is None
    next(s for s in app.selectbox if s.label=='Banda do catálogo').select('V').run()
    assert next(n for n in app.number_input if n.label=='Magnitude da referência').value==.03
    assert not next(c for c in app.checkbox if c.label=='Banda padrão / transformação validada').value


def test_v1674_default_search_fills_every_available_filter_and_survives_refresh(monkeypatch):
    report=json.loads((FIXTURES/'v1674cyg_catalogues.json').read_text())
    calls=[]
    def lookup(name):
        calls.append(name)
        return deepcopy(report)
    monkeypatch.setattr('photometry_catalogue_ui.lookup_reference',lookup)
    source='''
import streamlit as st
import numpy as np
from astropy.io import fits
from imageProcessing import frame
from photometry_ui import analysis_dialog
if 'images' not in st.session_state:
    st.session_state.images={band:frame(np.ones((60,60)),fits.Header({'FILTER':band}),name=band+'.fits')
                             for band in ['U','B','V',"g'","r'","i'",'J','H','K','R']}
analysis_dialog.__wrapped__(st.session_state.images)
'''
    app=AppTest.from_string(source,default_timeout=30).run()
    assert next(t for t in app.text_input if t.label=='Nome da estrela de referência').value=='V1674 Cyg'
    assert calls==[]  # No network request merely from opening the dialog.
    next(b for b in app.button if b.label=='Pesquisar').click().run()
    assert not app.exception,app.exception
    values=[n.value for n in app.number_input if n.label=='Magnitude da referência']
    assert len(values)==10  # Previously only the first three images were selected.
    assert values[:9]==pytest.approx([10.677999846637249,10.517000198364258,9.958000183105469,
                                      10.140000343322754,9.765000343322754,9.592000007629395,
                                      8.976,8.769,8.682])
    assert values[9] is None  # No invented Johnson R from a Sloan r' measurement.
    assert calls==['V1674 Cyg']
    next(b for b in app.button if b.label=='Pesquisar').click().run()
    assert not app.exception
    assert calls==['V1674 Cyg','V1674 Cyg']


def test_unexpected_lookup_error_stays_visible_in_dialog(monkeypatch):
    def fail(name):
        raise RuntimeError('service unavailable')
    monkeypatch.setattr('photometry_catalogue_ui.lookup_reference',fail)
    app=AppTest.from_string(UI_SOURCE.replace('FILTER_NAME',repr('V')),default_timeout=30).run()
    next(b for b in app.button if b.label=='Pesquisar').click().run()
    assert not app.exception
    assert any('Não foi possível concluir' in item.value for item in app.error)
    assert next(n for n in app.number_input if n.label=='Magnitude da referência').value is None
    assert any(b.label=='Calcular análises' for b in app.button)
