"""Input-file validation for the spot-detection pipeline: genes.yaml,
channel_parameters.yaml, and the chromatic z-shift dict."""
from __future__ import annotations

import yaml


def validate_genes_yaml(genes_path):
    with open(genes_path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f'Genes YAML must be a mapping at the top level, got {type(data).__name__}')

    for gene, channels in data.items():
        if not isinstance(channels, list):
            raise ValueError(
                f'Gene "{gene}": expected a list of channel wavelengths, '
                f'got {type(channels).__name__}'
            )
        if len(channels) == 0:
            raise ValueError(f'Gene "{gene}": channel list must not be empty')
        for i, ch in enumerate(channels):
            if not isinstance(ch, int):
                raise ValueError(
                    f'Gene "{gene}", entry {i}: expected an integer channel wavelength, '
                    f'got {type(ch).__name__} ({ch!r})'
                )
    return data


def validate_chromatic_z_shift(chromatic_z_shift_px):
    """
    chromatic_z_shift_px must be None, or a dict mapping channel wavelength
    (int) to an integer pixel z-shift.
    """
    if chromatic_z_shift_px is None:
        return
    if not isinstance(chromatic_z_shift_px, dict):
        raise ValueError(
            f'chromatic_z_shift_px must be None or a dict keyed by channel '
            f'wavelength (int), got {type(chromatic_z_shift_px).__name__}'
        )
    for ch, shift in chromatic_z_shift_px.items():
        if not isinstance(ch, int):
            raise ValueError(f'chromatic_z_shift_px key {ch!r} must be an int wavelength')


def validate_channel_parameters(channel_parameters):
    """
    channel_parameters must be a dict {wavelength: {'threshold': int, 'tolerance': int}},
    as read from channel_parameters.yaml.
    """
    for wl, params in channel_parameters.items():
        if 'threshold' not in params:
            raise ValueError(f'Missing entry for threshold in channel {wl}')
        if type(params['threshold']) is not int:
            raise ValueError(f'The threshold value for channel {wl} is not of type int')
        if 'tolerance' not in params:
            raise ValueError(f'Missing entry for tolerance in channel {wl}')
        if type(params['tolerance']) is not int:
            raise ValueError(f'The tolerance value for channel {wl} is not of type int')
    return channel_parameters
