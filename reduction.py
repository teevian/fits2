"""Compatibilidade: as operações de reduction estão em imageProcessing.py."""
from imageProcessing import (
    Frame,
    _registration_array,
    add_history,
    calibrate,
    clean_header,
    combination_exposure,
    combine,
    compatible,
    estimate_translation,
    exposure,
    frame,
    mask_image,
    read_frame,
    read_mask,
    to_fits,
    translate_frame,
)
