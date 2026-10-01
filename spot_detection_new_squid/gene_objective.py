"""Calibrate per-channel thresholds against the *gene* outcome.

Why not spots-per-megavoxel
---------------------------
`target_density` asks every channel to yield the same spot density. That is
only right when every channel carries the same number of transcripts, which is
false as soon as genes use different colour combinations: a channel appearing
in three barcodes should carry roughly three times the spots of one appearing
in a single barcode. Worse, the channels are *coupled* -- whether a 488 spot
ends up in a gene depends on whether its partner 561 and 638 spots were also
detected -- so tuning each channel in isolation cannot optimise the thing you
actually care about.

This module optimises the end result directly: choose per-channel thresholds
that maximise the number of spot groups matching a barcode in genes.yaml while
holding down the ones that match nothing.

The degenerate solution, and the fix
------------------------------------
Maximising "assigned" alone is degenerate. Lowering every threshold produces
more spots, more spots produce more accidental co-localizations, and some of
those accidents happen to span exactly a barcode's channel set -- especially
for two-colour genes. Left unchecked the optimiser drives every threshold to
the noise floor and reports a huge, entirely spurious gene count.

So the score is chance-corrected using *decoy* barcodes: channel combinations
of the same size that are NOT in genes.yaml. Real genes are enriched above
chance; accidental co-localizations are spread evenly across all combinations
of a given size. The mean count per decoy combination therefore estimates how
many "hits" a genuine barcode would collect by chance alone, and

    score = assigned - chance_weight * expected_assigned_by_chance

has an interior optimum: too permissive and the decoy rate explodes, too
strict and real genes are lost. This is the same logic as blank/decoy barcodes
in MERFISH-style pipelines, but computed from the gene set already on hand.
"""
from __future__ import annotations

from collections import Counter
from math import comb

import numpy as np

from .gene_filter import group_points


def gene_score(grouped_keys, gene_keys, all_channels, chance_weight=1.0,
               unassigned_weight=1.0):
    """Score one set of grouped keys::

        score = assigned - chance_weight * chance - unassigned_weight * unassigned

    `chance` is the expected number of assigned groups arising from random
    co-localization, estimated from decoy combinations of matching size.

    Both penalties are needed and they cover different failures. The decoy
    term stops the optimiser manufacturing barcodes out of accidental
    co-localizations, but it only ever looks at combinations the *size* of a
    barcode -- so a flood of single-channel junk spots is invisible to it.
    The `unassigned` term covers that, and is the half of the objective that
    keeps thresholds above the junk. It also self-limits over-strictness:
    losing one channel of a three-colour gene costs an assigned group *and*
    creates an unassigned one, so the penalty is two-sided.
    """
    counts = Counter(grouped_keys)
    total = sum(counts.values())
    assigned = sum(n for k, n in counts.items() if k in gene_keys)
    unassigned = total - assigned

    n_ch = len(all_channels)
    sizes = {len(k) for k in gene_keys}
    chance = 0.0
    for size in sizes:
        n_gene_keys = sum(1 for k in gene_keys if len(k) == size)
        n_possible = comb(n_ch, size) if size <= n_ch else 0
        n_decoy = n_possible - n_gene_keys
        if n_decoy <= 0:
            continue  # every combination of this size is a gene; no decoys exist
        decoy_obs = sum(n for k, n in counts.items()
                        if len(k) == size and k not in gene_keys)
        chance += n_gene_keys * (decoy_obs / n_decoy)

    score = assigned - chance_weight * chance - unassigned_weight * unassigned
    return score, assigned, unassigned, chance


def evaluate(chunk_candidates, thresholds, tolerances, gene_keys, all_channels,
             radius, depth, resolution, chance_weight=1.0, unassigned_weight=1.0):
    """Score one full parameter set across every calibration chunk.

    `chunk_candidates` is a list of {channel: (Candidates, coords)} -- one per
    chunk, all channels sampled at the same (z, y, x) so they can be grouped
    together. Post-filtering plus grouping is all that happens here, so an
    evaluation is cheap enough to sit inside an optimiser loop.
    """
    tot_score = tot_assigned = tot_unassigned = tot_chance = 0.0
    for per_channel in chunk_candidates:
        coords, chans = [], []
        for ch, (cand, xyz) in per_channel.items():
            keep = ((cand.raw_intensity >= thresholds[ch]) &
                    (cand.raw_prominence >= tolerances[ch]))
            if not np.any(keep):
                continue
            sel = xyz[keep]
            # (z, y, x) voxels -> (x, y, z) nm, matching gene_filter's frame.
            coords.append(np.column_stack([sel[:, 2] * resolution[0],
                                           sel[:, 1] * resolution[1],
                                           sel[:, 0] * resolution[2]]))
            chans.append(np.full(len(sel), ch, dtype=int))
        if not coords:
            continue
        grouped = group_points(np.vstack(coords), np.concatenate(chans), radius, depth)
        s, a, u, c = gene_score(grouped['key'], gene_keys, all_channels,
                                chance_weight, unassigned_weight)
        tot_score += s
        tot_assigned += a
        tot_unassigned += u
        tot_chance += c
    return tot_score, int(tot_assigned), int(tot_unassigned), tot_chance


def threshold_grid(chunk_candidates, channel, n=30):
    """Candidate thresholds for one channel, spanning its own intensity
    distribution from the noise median out into the bright tail.

    Deliberately *absolute* rather than a multiple of the seed estimate. Spots
    are often well under 1% of candidates, and at that signal fraction Otsu can
    land inside the noise mode -- on a fixture where real spots sat at 1500 over
    a background of 120, the seed came back at 133. A relative search around
    such a seed cannot reach the right answer, whereas percentiles of the
    channel's own candidates always bracket it. Spacing is geometric in the
    upper tail, where all the interesting thresholds live.
    """
    pooled = np.concatenate([
        per_channel[channel][0].raw_intensity
        for per_channel in chunk_candidates if channel in per_channel])
    if pooled.size == 0:
        return np.array([0.0])
    pcts = 100.0 - np.geomspace(50.0, 0.005, n)
    return np.unique(np.percentile(pooled, pcts))


def optimize(chunk_candidates, base_thresholds, tolerances, gene_keys, all_channels,
             radius, depth, resolution, chance_weight=1.0, unassigned_weight=1.0,
             grid_points=30, passes=3, verbose=True):
    """Coordinate ascent on the per-channel threshold.

    Only the thresholds are optimised -- `tolerance` keeps its
    false-positive-budget meaning, and halving the search space keeps this
    tractable. Each channel is scanned over `threshold_grid` with the others
    held fixed, repeated for `passes` sweeps or until nothing moves.
    `base_thresholds` supplies only the starting point.

    Returns (thresholds, scales, history), where `scales` reports the final
    value as a multiple of the seed for reference.
    """
    channels = sorted(base_thresholds)
    grids = {ch: threshold_grid(chunk_candidates, ch, grid_points) for ch in channels}
    current = dict(base_thresholds)

    best_score, a, u, c = evaluate(chunk_candidates, current, tolerances,
                                   gene_keys, all_channels, radius, depth, resolution,
                                   chance_weight, unassigned_weight)
    if verbose:
        print(f'    start: score={best_score:.1f} assigned={a} unassigned={u} chance={c:.1f}')

    history = [{'pass': 0, 'channel': None, 'threshold': np.nan, 'score': best_score,
                'assigned': a, 'unassigned': u, 'chance': c}]

    for p in range(1, passes + 1):
        moved = False
        for ch in channels:
            trial = dict(current)
            best_for_ch, best_stats = current[ch], None
            for t in grids[ch]:
                trial[ch] = float(t)
                score, aa, uu, cc = evaluate(chunk_candidates, trial, tolerances,
                                             gene_keys, all_channels, radius, depth,
                                             resolution, chance_weight, unassigned_weight)
                if score > best_score + 1e-9:
                    best_score, best_for_ch, best_stats = score, float(t), (aa, uu, cc)
            if best_for_ch != current[ch]:
                current[ch] = best_for_ch
                moved = True
                if verbose and best_stats:
                    aa, uu, cc = best_stats
                    print(f'    pass {p} ch {ch}: threshold={best_for_ch:.0f} '
                          f'score={best_score:.1f} assigned={aa} unassigned={uu} chance={cc:.1f}')
                history.append({'pass': p, 'channel': ch, 'threshold': best_for_ch,
                                'score': best_score,
                                'assigned': best_stats[0] if best_stats else np.nan,
                                'unassigned': best_stats[1] if best_stats else np.nan,
                                'chance': best_stats[2] if best_stats else np.nan})
        if not moved:
            break

    scales = {ch: current[ch] / base_thresholds[ch] if base_thresholds[ch] else np.nan
              for ch in channels}
    return current, scales, history
