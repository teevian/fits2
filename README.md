# FITS Viewer

A local FITS viewer for astronomical images. The interface is in English and runs in a browser; files and processing stay on your computer.

## Start

On macOS, double-click `iniciar.command`. Or run:

```sh
source venv/bin/activate
python -m streamlit run app.py --server.address 127.0.0.1
```

Open http://localhost:8501. Press Ctrl+C in the terminal to stop it.

## Use

The included `share-20261001` folder opens automatically on first use. The last working folder is remembered; change it at the top of the sidebar when needed. The viewer starts with a black canvas. Click grid thumbnails to select or deselect frames. Drag across thumbnails to select or deselect several at once. Double-click a thumbnail to open the full image; its FITS header appears underneath. Selection updates only the sidebar and leaves the canvas in place. Exports remain alongside the processing results. The interface uses a black background, light grey text and muted blue-grey highlights. The sidebar contains collapsible processing sections.

**all**, next to **auto**, opens a new browser tab with every FITS file in the working folder in a four-column gallery. The **Filter files (regex)** field filters filenames without distinguishing case; an empty filter shows all files. Medium-sized previews show filenames underneath and can be enlarged. Automatic preview contrast makes faint images visible; original FITS values remain unchanged. Previews are cached and refreshed when their files change.

Click **auto**, directly below **Working folder**, to combine the folder's original bias frames, combine its flats and subtract the new master bias (**-bias**), then **Correct** every original science image using the corresponding corrected master flat. Each step waits for the previous one to succeed. Auto uses the selected Median/Mean calibration methods and includes all science targets and file numbers, replacing previous frame selections. It identifies calibrations by FITS headers or filenames, excludes dark frames and generated products, and leaves original files unchanged. The masters and original/corrected science pairs appear in their usual sections; the first corrected image opens in the viewer. Missing inputs or processing errors stop the sequence and show an error. Alignment and science combination remain available as subsequent manual steps.

Click a filename or its view icon to open the image, including generated masters. The viewer uses the full pixel array without smoothing or preview subsampling. **Contrast** remains a toggle beside **Combine** in Science; bias and flat results offer normal and contrast thumbnails that open on double-click. It applies the previous linear screen contrast between the 1st and 99th percentiles of valid pixels. Raw combined flats are in counts; corrected flats are dimensionless, normalized responses. They therefore use separate display scales. With contrast off, biases use a fixed 500–630 display range, normalized flats use 0–2, and science images use 0–65535 counts (legacy unnormalized science ratios retain 0–1). The controls above the image have been removed. Contrast affects only the screen: alignment, combination and bias subtraction always use the original pixel values, as do downloads. Uniform images are identified with their constant pixel value. Axes, a pixel-value readout, sit beside the image. FITS exports preserve floating-point and negative values. PNG exports write 16-bit integer counts without contrast scaling (normalized flat responses map 0–2 onto 16-bit grayscale); values outside 0–65535 are clipped and fractional counts are rounded because PNG cannot represent FITS floating-point values.

The File section includes downloads, statistics and the header. Bias, flat and science combination default to Median with per-pixel 5σ clipping around the median using MAD scatter (up to five iterations). Mean remains available with the same rejection. Flats are bias-subtracted individually, masked where their response is nonpositive, scaled by their valid median, then combined with the same clipping. The resulting flat is normalized again to median 1. These two sections have no alignment controls. Alignment happens only when **Align** is clicked. **Combine** calculates both master flats and corrected flats; **-bias** is enabled by default and selects the already computed version. With **-bias** on, the result is the calibrated, normalized master flat used for science. With it off, raw combined flats are diagnostic previews only. Normalization must happen before combination; subtracting bias from an already combined raw flat is insufficient for variable illumination.

In **Processing**, the `bias*` and `flat*` patterns select matching files from the working folder. Regex matching ignores case, and combination starts at **Median** with **Mean** also available. Bias and flat frames appear in selectable thumbnail grids; clicking toggles selection; dragging across thumbnails applies the same selection choice; double-clicking opens an image without changing the selection. Generated images use the same controls and can be downloaded individually or together as FITS or PNG.

**Science**, the third section, accepts a filename regex such as `hd226868.*` and remembers target shortcuts. Select original frames in the first grid, then use **Correct**. If necessary, this prepares the master bias and normalized, bias-corrected flats first. Each science image is corrected as `(science - master_bias) / normalized_master_flat` for its filter, preserving count units. The comparison grid displays originals on the left and corrected images on the right. All newly corrected images are selected automatically; originals in this comparison grid cannot be selected for combination. Click or drag over corrected images to change the selection, or double-click either version to view it.

**Align + Combine** registers the selected corrected images within each object/filter group and immediately combines them. **Median** is the default; **Mean** is available beside the button. Both use 5σ rejection. Red, Green, Blue and black inputs for one target produce four masters, even across different exposures. Corrected frames are scaled to the first frame’s exposure before rejection and combination. The master records that reference exposure and EXPNORM, so mixed exposure times do not receive inconsistent brightness weights. A filter with one selected image retains that corrected image as its result. Correction happens before registration; full-resolution pixels are resampled once and uncovered borders are masked. The maximum shift is in pixels, and the align log records measured shifts. Uncertain registration reports an error. At three minutes, **Continuar** resumes the same worker or **Parar** cancels it.

The **show** button beside **Align + Combine** displays each group's corrected, aligned inputs and its resulting master. Below the controls, a grid shows each master without and with contrast; these are two display variants of the same scientific image. Double-click a variant to open it. Results also have individual viewing and download controls, plus a ZIP download. Changing inputs or settings invalidates stale results; viewing and contrast do not repeat processing.

**Download master bias (ZIP)** and **Download all flats (ZIP)** package the produced images in `with_constrast/` and `no_contrast/`. The flat ZIP includes every available master flat and its bias-corrected version, independent of the display toggles. **FITS ZIP** stores unchanged scientific FITS in `no_contrast/` and contrast PNG previews in `with_constrast/`; **PNG ZIP** stores 8-bit previews matching the viewer with Contrast off in `no_contrast/` and with Contrast on in `with_constrast/`. The no-contrast bias PNG uses the same fixed 500–630 display range as the viewer, so it remains visible; scientific counts remain unchanged in the FITS. Normalized and raw flats use separate percentile scales, matching the viewer. Opening the download dialog immediately shows both format choices while ZIPs prepare in background workers. Each format becomes available independently, with a 30-second limit and a retry button on failure. Reopening or toggling **-bias**/**Contrast** reuses the completed ZIPs; changing the produced masters invalidates the export cache. Export never changes master data or calibration results.

Original files are never overwritten; viewing, analysis and exports use full resolution. Browser uploads use Streamlit's default 200 MB limit. Large FITS files and batches can require substantial memory.

**show**, beside **Combine** in Master bias and Flats, opens a gallery of combination groups. Each bordered group shows all contributing originals in a grid on the left and the resulting master once on the right, with filenames underneath. Flats are grouped by filter across all exposure times. Each input is scaled to the first input’s exposure before rejection and combination; corrected flats have bias subtracted first, then are individually median-normalized. The corrected master has median 1. Count-valued inputs share a contrast scale; normalized flat results use their own scale. The flat preview follows **-bias**. Show becomes available after combination and uses the existing results.

## Install and test


All image calculations, FITS operations, display-scale helpers and exports are implemented in `imageProcessing.py`. Each function has a one-line Portuguese comment describing its operation. The UI imports this module directly; the former processing modules only re-export the same functions for compatibility. `processing_jobs.py` retains process isolation and timeout handling.

Within `imageProcessing.py`, section comments group FITS reading and validation, alignment, calibration, combination, masks, statistics, science correction, display and exports. `calibrate` calls individual helpers for bias subtraction (`_subtract_bias`), dark subtraction (`_subtract_dark`), normalized flat division (`_divide_by_normalized_flat`) and external masks (`_apply_bad_pixel_mask`). These helpers update only the working copy created by `calibrate`. `correct_science_frame` applies normalized flat correction and records FLATNORM=True and DARKCOR=False. No dark or overscan correction is silently assumed.

Python 3.12 (the NumPy 1.x / Astropy 6 environment):

```sh
python3.12 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip install pytest
venv/bin/python -m pytest -q
```

`requirements-lock.txt` records the versions used to build the included environment.

**Master bias** and **Flats** show a selection grid directly below its regex. Only matching files appear; click or drag to select/deselect them immediately. Deselected images remain visible, and **Combine** uses only the selected images (at least two).

**Science** also provides an inline grid for regex matches. Click or drag to select/deselect immediately. **Correct** uses only the selected images; the adjacent **show** opens original/corrected pairs.

**show** beside Science **Align + Combine** opens a grouped gallery with every corrected, aligned input and one final result per object/filter. The previews share a scale within each group. **Download all corrected frames (ZIP)** exports individual corrections after Correct; **Download all corrected science results (ZIP)** exports final grouped results after Align + Combine, with FITS and PNG choices. Changing inputs or settings disables stale previews and downloads.

See [CALIBRATION_AUDIT.md](CALIBRATION_AUDIT.md) for the Astropy guide comparison, exact formula, input inventory, verification and remaining camera-dependent limitations. Old in-memory calibration signatures are invalidated by the new recipe; recompute using auto or Correct.

Science also has **show** beside **Correct** (original → corrected), with a shared contrast scale for each pair. The **show** beside **Align + Combine** displays the contributing corrected, aligned images and each final master. Both use cached results.

The master bias result shows its normal preview beside a contrast preview, above its filename. These previews do not alter the scientific pixels.

Double-click either master-bias preview to open that exact contrast version in the main viewer. Each master-flat group identifies its filter and exposure and offers four previews that open on double-click: raw combined flat and bias-subtracted normalized flat, each with and without contrast. Bias and flat sections no longer have a separate Contrast button. Preview selection reuses the existing results.

Science uses the same inline regex selection grid as Bias and Flats, with click or drag to select and double-click to view. There is no separate choose dialog. The show beside Correct compares each original with its corrected result side by side.

Click anywhere on a tile to select or deselect it. A stronger blue outline indicates selection; there are no checkboxes. Drag across tiles to apply the first tile’s selection choice to every image crossed, including while scrolling. Double-click an original or result thumbnail to view it in the main canvas while preserving the selection.

Grid gestures disable native text/image dragging and text highlighting. Hold the mouse button and drag to apply the initial select/deselect operation across images; mouse-wheel scrolling and automatic scrolling at the top/bottom edges keep painting new rows until release.


### Final color image

Below the science master results, **Generate** defaults to an RGB display image from one target's calibrated Red, Green and Blue science masters. Enable **Use black as luminance** only when the instrument's black position is confirmed as clear/luminance; this explicitly selects LRGB. The name black alone does not identify a passband. Raw images and calibration frames are rejected. The selected channels are registered to the Red master with the existing stellar subpixel translation algorithm. Different dimensions, binning or known pixel scales are rejected; rotation and optical distortion are not corrected by this workflow. Borders outside the common valid footprint are masked.

For display, a robust sky plane is fitted to clipped grid medians. Common isolated stars provide Gaussian PSF widths and aperture fluxes. RGB channels are convolved to the broadest measured RGB width using the quadrature-difference Gaussian kernel; invalid margins are expanded accordingly. At least three usable common stars are required. Median stellar flux ratios provide one global gain per channel, so the ensemble has neutral median color while individual star colors remain distinct. This is an automatic visual balance, not catalog zero-point calibration. The clear channel provides luminance, with regularized chromaticity in low-signal pixels. One common final display scale replaces the former independent channel percentiles.

Source masters remain unchanged. **Without contrast** uses linear mapping with hue-preserving over-range compression; **With contrast** uses Astropy’s Lupton arcsinh mapping. Both variants appear below Generate and can be downloaded as PNG. Generate opens the result on the main color canvas; double-clicking a variant switches the canvas without recomputing registration. The new exposure and color recipes invalidate earlier cached masters/results: run **Align + Combine → Generate** again. Gaussian matching addresses approximately Gaussian, spatially consistent blur; severely defocused, distorted or saturated stars can still show residual artifacts.

References: [Astropy CCD reduction guide](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-00-Understanding-an-astronomical-CCD-image.html), [Astropy RGB visualization](https://docs.astropy.org/en/stable/visualization/rgb.html), [Lupton et al. (2004)](https://arxiv.org/abs/astro-ph/0312483), and [Siril's registration and level-equalization guidance](https://siril.org/tutorials/rgb_composition/).

PSF-matching reference: [Photutils Gaussian matching](https://photutils.readthedocs.io/en/3.1.0/user_guide/psf_matching.html).

Flat channel PNGs are display aids only: Red, Green and Blue populate their corresponding RGB channel; black and unassigned filters use neutral grayscale (Gray), without assigning a physical luminance band. Both ZIP folders include these previews. Calibration and final color composition use scientific FITS arrays, never these PNGs.

### Análise astrofísica observacional

Depois de criar os masters, **Análise astrofísica** abre um popup com fotometria dos
FITS lineares, perfis radiais/Moffat, curvas de crescimento, histogramas e exportação
reprodutível. Temperatura, ponto H–R e escala de vida na sequência principal exigem
calibração e dados físicos independentes. Ver [PHOTOMETRY.md](PHOTOMETRY.md) para
métodos, referências e limites científicos.

A calibração predefinida usa uma referência do mesmo FOV: clicar no alvo e na
estrela de referência, indicar a magnitude tabelada por filtro e calcular. O popup
mede ambos os fluxos e determina a magnitude diferencial e o ponto zero, guardando
também os dados e diagnósticos da referência no relatório.

Introduzir o nome da referência consulta **SIMBAD e VizieR** e sugere magnitudes
por banda, com fontes e publicações visíveis. Os campos continuam editáveis e o
relatório preserva a origem dos valores. Filtros RGB genéricos exigem seleção
explícita da banda. O popup usa fundo branco, campos compactos e opções avançadas
recolhidas para facilitar a leitura.

### Recuperação automática da última sessão

O programa guarda configurações, seleções, parâmetros de análise, masters e RGB
final em `.streamlit/session-recovery/`, fora do controlo de versões. Inclui cópias
dos FITS corrigidos e alinhados que os processos de cálculo criam em pastas
temporárias. Reabrir o endereço inicial, fazer reload ou reiniciar o servidor
recupera automaticamente a última sessão local, sem repetir o processamento.
Os parâmetros de popups fechados também são conservados.

A gravação é transacional e atualiza apenas entradas alteradas. As imagens mantêm
as identidades usadas pelo workflow, evitando invalidar resultados apenas porque
o processo reiniciou. Os inputs originais devem continuar acessíveis; alterações
nesses ficheiros continuam a invalidar os cálculos dependentes. Botões e trabalhos
em execução não são retomados automaticamente. Erros de gravação ou recuperação
são apresentados na interface. Esta recuperação destina-se à aplicação local de
um utilizador e não substitui as exportações científicas permanentes.
