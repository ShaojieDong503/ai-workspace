# MERRA-2 day window: evidence that the archive is UTC-6

The files carry no metadata recording the averaging window, so the window has to
be established from behaviour. Two independent pieces of evidence, one weak and
one strong.

## Strong: a lag-mix fit

Each UTC-6 daily value is fitted as a weighted mix of the UTC-day means for
D-1, D and D+1. The UTC-day means come from the older export
(`SourceRegion_MERRA_2010_2024-*.zip`), which is on the same lattice offset by
exactly 2 columns and 2 rows. The fit uses all pixels from 2015-07-02 to
2015-07-29.

A true [D 06:00, D+1 06:00) window should put weights of roughly 0, 0.75 and 0.25
on D-1, D and D+1.

| variable | D-1 | D | D+1 | residual / SD |
|---|---:|---:|---:|---:|
| PBLH | -0.06 | 0.72 | 0.33 | 0.38 |
| TLML | -0.11 | 0.88 | 0.24 | 0.09 |
| ULML | -0.11 | 0.88 | 0.25 | 0.19 |
| U10M | -0.11 | 0.90 | 0.21 | 0.18 |
| V2M | -0.11 | 0.91 | 0.21 | 0.22 |

The D+1 weight is positive for every variable and D-1 is about zero. A UTC-day
window would put almost no weight on D+1; a shift in the wrong direction would
weight D-1 instead. The weights are not exactly 0.75 / 0.25 because consecutive
days are correlated, which also explains the small negative D-1 coefficients.

## Weak (superseded)

An earlier comparison against the UTC-day export gave a median ratio of 0.973 and
a correlation of 0.951 for the same cells and month. That shows the two archives
differ, but it cannot distinguish a six-hour shift from an unshifted day on a
slightly different grid - it was over-read as evidence and is recorded here only
so the earlier claim is not mistaken for support.

## Limits

Absolute proof would need one hourly granule: compute its mean over
[D 06:00, D+1 06:00) and compare with the daily file. Until then, the window is
established to the level above and the build proceeds on that basis.
