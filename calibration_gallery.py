"""Calibration provenance: each group contains its inputs and one master."""
import numpy as np
import streamlit as st

from gallery_ui import frame_thumbnail
from processing_ui import load


def display_sample(image):
    values = image.display().ravel()
    sample = values[::max(1, values.size // 16384)]
    return sample[np.isfinite(sample)]


def combination_group(title, sources, master, key, science=False):
    # Keep only small samples in memory while finding one scale for the group.
    samples = [display_sample(master)]
    for source in sources:
        try:
            samples.append(display_sample(load(source)))
        except (ValueError, OSError, TypeError, KeyError):
            pass
    sample = np.concatenate(samples)
    low, high = np.percentile(sample, [1, 99]) if sample.size else (0., 1.)
    if high <= low:
        low, high = low - max(abs(low) * .05, 1.), high + max(abs(high) * .05, 1.)
    limits = (float(low), float(high))

    with st.container(border=True, key=key):
        st.subheader(title)
        st.caption(f'{len(sources)} corrected, aligned frames → 1 result' if science
                   else f'{len(sources)} originals → 1 master')
        originals, result = st.columns([3, 1], gap='large')
        with originals:
            st.markdown('**Corrected, aligned inputs**' if science else '**Combined originals**')
            for offset in range(0, len(sources), 3):
                for column, source in zip(st.columns(3), sources[offset:offset + 3]):
                    with column:
                        name = source[0] if isinstance(source, tuple) else source.name
                        try:
                            st.image(frame_thumbnail(load(source), limits), width='stretch')
                        except (ValueError, OSError, TypeError, KeyError) as exc:
                            st.warning(f'Preview unavailable: {exc}')
                        st.caption(name)
        with result:
            st.markdown('**→ Resulting master**')
            st.image(frame_thumbnail(master, None if master.header.get('NORMALIZ') else limits), width='stretch')
            st.caption(master.name)


@st.dialog('Bias · combination groups', width='large')
def show_bias(sources, master):
    st.caption('Each group shows the originals used and their resulting master once. Normalized flat results use their own scale; raw inputs retain their count scale.')
    combination_group('Master bias', sources, master, 'bias_combination_group')


@st.dialog('Flats · combination groups', width='large')
def show_flats(sources, masters, inputs):
    st.caption('Each group shows the originals used and their resulting master once. Normalized flat results use their own scale; raw inputs retain their count scale.')
    catalog = {source[0] if isinstance(source, tuple) else source.name: source for source in sources}
    for index, (group, master) in enumerate(masters.items()):
        combination_group(group, [catalog[name] for name in inputs[group]], master,
                          f'flat_combination_group_{index}')


@st.dialog('Science · combination groups', width='large')
def show_science(sources, masters, inputs):
    st.caption('One result per object and filter. Each group shows the corrected, aligned images actually combined and its final result once.')
    catalog = {source.name: source for source in sources}
    for index, (group, master) in enumerate(masters.items()):
        combination_group(group, [catalog[name] for name in inputs[group]], master,
                          f'science_combination_group_{index}', science=True)


@st.dialog('Science · before / after', width='large')
def show_science_pairs(pairs, stage):
    before_label, after_label = ('Original', 'Corrected') if stage == 'correction' else ('Corrected · before alignment', 'Aligned')
    st.caption('One row per image. Each pair shares the same contrast scale; filenames appear below the previews.')
    for index, (original_name, before, after) in enumerate(pairs):
        with st.container(border=True, key=f'science_{stage}_pair_{index}'):
            st.subheader(original_name)
            try:
                images = [load(before), load(after)]
                samples = np.concatenate([display_sample(image) for image in images])
                low, high = np.percentile(samples, [1, 99]) if samples.size else (0., 1.)
                if high <= low:
                    low, high = low - max(abs(low) * .05, 1.), high + max(abs(high) * .05, 1.)
                for column, label, image in zip(st.columns(2), (before_label, after_label), images):
                    with column:
                        st.markdown(f'**{label}**')
                        st.image(frame_thumbnail(image, (float(low), float(high))), width='stretch')
                        st.caption(image.name)
                if stage == 'alignment':
                    header = images[1].header
                    st.caption(f'Shift: Y {header.get("ALNSHFTY", 0.):+.2f} px · X {header.get("ALNSHFTX", 0.):+.2f} px')
            except (ValueError, OSError, TypeError, KeyError) as exc:
                st.warning(f'Preview unavailable: {exc}')
