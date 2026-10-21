"""Read-only catalogue suggestions, preserving passbands, provenance and missing errors."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import re
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import numpy as np
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
from astropy import units as u

SIMBAD = 'https://simbad.cds.unistra.fr/simbad/sim-tap/sync'
VIZIER = 'https://vizier.cds.unistra.fr/viz-bin/votable'
CATALOGUES = 'II/336/apass9,II/7A/catalog,II/168/ubvmeans'
BANDS = ['U', 'B', 'V', 'R', 'I', "g'", "r'", "i'", 'J', 'H', 'K']


def finite(value):
    try:
        if np.ma.is_masked(value):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def text_value(value):
    if value is None or np.ma.is_masked(value):
        return ''
    return str(value).strip()


def fetch(url):
    request = Request(url, headers={'User-Agent': 'FITS2-observational-photometry/1.0', 'Accept': '*/*'})
    with urlopen(request, timeout=15) as response:
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError('Resposta do catálogo excede o limite de leitura.')
    return raw


def simbad_url(name):
    # ADQL string literals escape apostrophes by doubling, not shell escaping.
    identifier = name.replace("'", "''")
    query = ("SELECT TOP 100 b.main_id,b.ra,b.dec,b.otype,f.filter,f.flux,f.flux_err,f.bibcode,f.qual "
             "FROM basic AS b JOIN ident AS i ON b.oid=i.oidref LEFT JOIN flux AS f ON b.oid=f.oidref "
             f"WHERE i.id='{identifier}'")
    return SIMBAD + '?' + urlencode(dict(request='doQuery', lang='adql', format='json', query=query))


def vizier_url(name):
    return VIZIER + '?' + urlencode({'-source': CATALOGUES, '-c': name, '-c.rs': '3',
                                    '-out.all': '1', '-out.max': '30'})


def record(service, catalogue, band, magnitude, error, object_id, url, **extra):
    magnitude, error = finite(magnitude), finite(error)
    if magnitude is None or not -30 < magnitude < 40:
        return None
    if error is not None and error < 0:
        error = None
    return dict(service=service, catalogue=catalogue, band=band, magnitude=magnitude,
                error=error, object_id=object_id, url=url, **extra)


def parse_simbad(raw, query_url):
    data = json.loads(raw)
    columns = [item['name'] for item in data['metadata']]
    rows = [dict(zip(columns, row)) for row in data['data']]
    records, objects = [], {}
    for row in rows:
        identity = text_value(row.get('main_id'))
        objects[identity] = dict(name=identity, ra=finite(row.get('ra')), dec=finite(row.get('dec')),
                                 object_type=text_value(row.get('otype')))
        band = row.get('filter')
        if band not in BANDS:
            continue
        notes = []
        if band not in ('U', 'B', 'V'):
            notes.append('Sistema da banda a verificar na publicação; R/I não distinguem Johnson de Cousins aqui.')
        if row.get('qual') in ('D', 'E'):
            notes.append('Qualidade fotométrica baixa no SIMBAD: ' + row['qual'])
        value = record('SIMBAD', 'flux', band, row.get('flux'), row.get('flux_err'), identity,
                       'https://simbad.cds.unistra.fr/simbad/sim-id?' + urlencode({'Ident': identity}),
                       system=f'Johnson {band} / Vega' if band in ('U', 'B', 'V') else f'{band} / verificar sistema na fonte',
                       bibcode=text_value(row.get('bibcode')), quality=text_value(row.get('qual')),
                       notes=notes, query_url=query_url, ambiguous=len({r['main_id'] for r in rows}) > 1,
                       **{k: objects[identity][k] for k in ('ra', 'dec', 'object_type')})
        if value:
            records.append(value)
    return records, list(objects.values())


def parse_vizier(raw, query_url):
    votable = parse(BytesIO(raw), verify='ignore')
    for info in votable.infos:
        if info.name in ('Error', 'QUERY_STATUS') and (info.name == 'Error' or info.value == 'ERROR'):
            raise ValueError('O VizieR não conseguiu resolver a consulta.')
    records = []
    for resource in votable.resources:
        metadata = {info.name: info.value for info in resource.infos}
        bibcode = metadata.get('cites', '').removeprefix('bibcode:')
        for table in resource.tables:
            catalogue = table.name or ''
            data = table.to_table(use_names_over_ids=True)
            rows = [{col: row[col] for col in data.colnames} for row in data]
            # Multiple epochs for the same identifier are not different stars.
            def identity(row):
                return text_value(row.get('LID')) or text_value(row.get('SimbadName')) or (
                    f"{text_value(row.get('RAJ2000'))},{text_value(row.get('DEJ2000'))}")
            ambiguous = len({identity(row) for row in rows}) > 1
            for row in rows:
                ra, dec = finite(row.get('RAJ2000', row.get('_RA'))), finite(row.get('DEJ2000', row.get('_DE')))
                object_id = text_value(row.get('SimbadName')) or identity(row)
                common = dict(bibcode=bibcode, ra=ra, dec=dec, query_url=query_url,
                              ambiguous=ambiguous, row_id=text_value(row.get('recno')),
                              reference_code=text_value(row.get('ref')), object_type=text_value(row.get('var')))
                url = 'https://vizier.cds.unistra.fr/viz-bin/VizieR?' + urlencode({'-source': catalogue, '-c': f'{ra} {dec}', '-c.rs': '3'})
                values = []
                if catalogue == 'II/336/apass9':
                    for band, column in [('B', 'Bmag'), ('V', 'Vmag'), ("g'", "g'mag"), ("r'", "r'mag"), ("i'", "i'mag")]:
                        uncertain = text_value(row.get('u_e_' + column)) not in ('', '0')
                        values.append((band, row.get(column), None if uncertain else row.get('e_' + column),
                                       f'Johnson {band} / Vega' if band in ('B', 'V') else f'Sloan {band} / AB (APASS)',
                                       ['Erro assinalado como incerto no APASS.'] if uncertain else []))
                elif catalogue == 'II/7A/catalog':
                    for band in ('U', 'B', 'V', 'R', 'I', 'J', 'H', 'K'):
                        values.append((band, row.get(band), None, f'Johnson {band} / Vega (II/7A)',
                                       ['Medição individual; erro não fornecido. Ver código de referência do catálogo.']))
                elif catalogue == 'II/168/ubvmeans':
                    v, bv, ub = (finite(row.get(col)) for col in ('Vmag', 'B-V', 'U-B'))
                    notes = ['O catálogo fornece dispersão, não um erro formal da média; o erro fica vazio.']
                    values.append(('V', v, None, 'Johnson V / Vega', notes))
                    if v is not None and bv is not None:
                        values.append(('B', v + bv, None, 'Johnson B / Vega', notes + ['Derivada: B = V + (B−V); covariância não fornecida.']))
                        if ub is not None:
                            values.append(('U', v + bv + ub, None, 'Johnson U / Vega', notes + ['Derivada: U = V + (B−V) + (U−B); covariância não fornecida.']))
                for band, mag, error, system, notes in values:
                    value = record('VizieR', catalogue, band, mag, error, object_id, url,
                                   system=system, notes=notes, **common)
                    if value:
                        records.append(value)
    return records


def lookup_reference(name, getter=fetch):
    name = ' '.join(name.split())
    if not name or len(name) > 150 or any(ord(c) < 32 for c in name):
        raise ValueError('Indica um nome estelar válido (até 150 caracteres).')
    urls = {'SIMBAD': simbad_url(name), 'VizieR': vizier_url(name)}
    report = dict(query=name, retrieved_utc=datetime.now(timezone.utc).isoformat(), records=[], objects=[], services=[])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {service: pool.submit(getter, url) for service, url in urls.items()}
        for service, future in futures.items():
            try:
                raw = future.result()
                if service == 'SIMBAD':
                    records, objects = parse_simbad(raw, urls[service])
                    report['objects'] = objects
                else:
                    records = parse_vizier(raw, urls[service])
                report['records'].extend(records)
                report['services'].append(dict(service=service, status='ok' if records else 'sem magnitudes',
                                                count=len(records), url=urls[service]))
            except Exception as exc:
                # Keep the other service usable; never substitute invented values.
                report['services'].append(dict(service=service, status='indisponível', count=0, url=urls[service],
                                                error=f'{type(exc).__name__}: {str(exc)[:180]}'))
    if len(report['objects']) == 1:
        obj = report['objects'][0]
        if obj['ra'] is not None and obj['dec'] is not None:
            center = SkyCoord(obj['ra']*u.deg, obj['dec']*u.deg)
            for value in report['records']:
                if value['service'] == 'VizieR' and value['ra'] is not None and value['dec'] is not None:
                    separation = float(center.separation(SkyCoord(value['ra']*u.deg, value['dec']*u.deg)).arcsec)
                    value['separation_arcsec'] = separation
                    if separation > 3:
                        value['ambiguous'] = True
                        value['notes'].append('Posição difere mais de 3″ do SIMBAD; associação não automática.')
    # Bibliographic entries can overlap between the two services; no averaging or claim of independence.
    for i, value in enumerate(report['records']):
        value['id'] = str(i)
    return report


def filter_band(filter_name):
    """Only explicit standard-band labels; never equate RGB colour names to passbands."""
    value = str(filter_name).strip()
    aliases = {'Johnson U': 'U', 'Johnson B': 'B', 'Johnson V': 'V', 'Johnson R': 'R', 'Johnson I': 'I',
               'SDSS g': "g'", 'SDSS r': "r'", 'SDSS i': "i'", 'g': "g'", 'r': "r'", 'i': "i'"}
    return value if value in BANDS else aliases.get(value)


def seed_suggestion(state, prefix, token, suggestion, reference_name):
    """Set editable widgets once per new suggestion, never on ordinary reruns."""
    if state.get(prefix+'suggestion_token') == token:
        return False
    state[prefix+'suggestion_token'] = token
    for suffix, value in [('ref_mag', suggestion['magnitude'] if suggestion else None),
                          ('ref_mag_error', suggestion['error'] if suggestion else None),
                          ('band', suggestion['system'] if suggestion else '')]:
        state[prefix+suffix] = value
    state[prefix+'passband'] = False
    state[prefix+'catalogue_original'] = suggestion
    state[prefix+'catalogue_reference_name'] = reference_name
    return True
