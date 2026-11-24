# Feature reduction

dataset A: 62 candidate columns, 145,315 development rows

constant columns dropped: none

## Correlation groups

|Spearman| >= 0.95, computed inside each training fold, but a pair is only merged when neither column is present where the other is missing on more than 10% of rows. Groups are listed with the folds they appear in.

Pairs declared in `feature_reduction.nested_families` are merged whatever their overlap, because the wider member carries the narrower one. Nesting is declared, not inferred: a gap-filled copy looks statistically identical to a nested pair but must stay separate. Each declaration is verified before use.

| folds | columns |
|---:|---|
| 10/10 | `aod055`, `aod047` |
| 10/10 | `aod_nbr_25km_d0`, `aod_nbr_50km_d0` |
| 10/10 | `aod_nbr_25km_d0_n`, `aod_nbr_50km_d0_n` |
| 10/10 | `fire_count_100km`, `fire_frp_sum_100km` |
| 10/10 | `fire_count_250km`, `fire_frp_sum_250km` |
| 10/10 | `fire_count_25km`, `fire_frp_sum_25km` |
| 10/10 | `fire_count_500km`, `fire_frp_sum_500km` |
| 10/10 | `fire_count_50km`, `fire_frp_sum_50km` |
| 10/10 | `hms_any`, `hms_n_polygons` |
| 10/10 | `merra_bccmass`, `merra_bcexttau` |
| 10/10 | `merra_ducmass25`, `merra_duexttau` |
| 10/10 | `merra_qlml`, `merra_tlml` |
| 10/10 | `merra_speed`, `merra_speedmax` |
| 10/10 | `merra_ulml`, `merra_u10m`, `merra_u50m` |
| 10/10 | `merra_vlml`, `merra_v10m`, `merra_v50m` |

## Correlated pairs NOT merged (poor co-observation)

| pair | corr | both present | A only | B only |
|---|---:|---:|---:|---:|
| `aod055` / `aod_last_obs` | 1.00 | 0.302 | 0.000 | 0.436 |
| `aod055` / `smoke_aod055` | 0.97 | 0.011 | 0.290 | 0.000 |
| `aod_last_obs` / `smoke_aod055` | 0.97 | 0.011 | 0.727 | 0.000 |

## Per-column completeness and gain importance

Gain is split among correlated columns: when two columns carry the same information, the tree picks one and the other looks unimportant. A low gain on a member of a correlation group is not evidence that the variable is useless.

| column | completeness | gain mean | gain sd |
|---|---:|---:|---:|
| `merra_bcsmass` | 1.000 | 8,967,045.5 | 728,130.3 |
| `merra_ocsmass` | 1.000 | 3,580,130.9 | 648,512.1 |
| `merra_so4smass` | 1.000 | 1,253,074.2 | 121,931.5 |
| `merra_tlml` | 1.000 | 1,007,714.3 | 42,054.3 |
| `fire_count_500km` | 1.000 | 784,344.1 | 176,093.5 |
| `merra_v850` | 1.000 | 777,511.6 | 213,657.4 |
| `merra_v250` | 1.000 | 666,872.4 | 17,474.4 |
| `merra_dusmass25` | 1.000 | 642,826.1 | 79,221.9 |
| `merra_v10m` | 1.000 | 523,762.7 | 102,680.3 |
| `merra_suexttau` | 1.000 | 504,597.1 | 197,765.5 |
| `fire_frp_sum_500km` | 1.000 | 469,350.1 | 225,842.8 |
| `merra_so4cmass` | 1.000 | 448,781.3 | 34,851.1 |
| `merra_ducmass25` | 1.000 | 426,723.6 | 64,587.9 |
| `merra_qlml` | 1.000 | 318,199.2 | 45,022.0 |
| `merra_vlml` | 1.000 | 311,014.2 | 33,965.4 |
| `merra_pblh` | 1.000 | 300,939.4 | 41,316.0 |
| `merra_u850` | 1.000 | 263,749.5 | 61,134.1 |
| `merra_sscmass25` | 1.000 | 253,136.9 | 32,279.6 |
| `merra_sssmass25` | 1.000 | 199,756.5 | 37,072.7 |
| `merra_duexttau` | 1.000 | 188,921.4 | 58,068.0 |
| `merra_ulml` | 1.000 | 185,648.8 | 44,816.3 |
| `aod_nbr_50km_d1_3_n` | 1.000 | 183,562.0 | 29,464.9 |
| `merra_v50m` | 1.000 | 182,028.5 | 33,792.8 |
| `merra_u250` | 1.000 | 176,556.6 | 15,681.2 |
| `aod_last_obs` | 0.738 | 175,064.7 | 26,084.4 |
| `aod_nbr_100km_d0` | 0.735 | 173,782.8 | 42,077.8 |
| `hms_n_polygons` | 0.999 | 157,844.5 | 31,100.7 |
| `merra_u500` | 1.000 | 156,262.8 | 27,850.8 |
| `aod_nbr_50km_d1_3` | 0.863 | 154,136.9 | 38,192.3 |
| `aod_nbr_100km_d0_n` | 1.000 | 153,104.2 | 13,130.3 |
| `merra_u10m` | 1.000 | 137,335.2 | 38,106.6 |
| `merra_occmass` | 1.000 | 134,568.0 | 15,692.1 |
| `aod_last_obs_age_d` | 0.738 | 127,766.4 | 14,151.4 |
| `merra_v500` | 1.000 | 124,994.6 | 22,656.4 |
| `fire_frp_sum_250km` | 1.000 | 121,854.5 | 57,404.4 |
| `merra_bcexttau` | 1.000 | 114,565.7 | 21,603.4 |
| `merra_speed` | 1.000 | 106,626.8 | 27,695.4 |
| `aod_nbr_50km_d0` | 0.611 | 102,807.9 | 17,643.6 |
| `merra_speedmax` | 1.000 | 92,705.5 | 23,841.0 |
| `merra_bccmass` | 1.000 | 86,139.4 | 8,166.2 |
| `merra_totexttau` | 1.000 | 83,401.8 | 12,581.2 |
| `fire_dist_nearest_km` | 1.000 | 64,578.3 | 8,052.4 |
| `merra_u50m` | 1.000 | 63,059.9 | 34,882.9 |
| `merra_ocexttau` | 1.000 | 56,663.2 | 9,372.5 |
| `aod_nbr_25km_d0` | 0.512 | 35,196.9 | 15,742.2 |
| `smoke_aod055` | 0.011 | 31,010.7 | 14,822.6 |
| `hms_density_max` | 0.138 | 25,654.4 | 16,078.5 |
| `aod_nbr_50km_d0_n` | 1.000 | 21,176.0 | 15,114.6 |
| `aod055` | 0.299 | 14,145.8 | 7,828.8 |
| `aod_nbr_25km_d0_n` | 1.000 | 12,507.9 | 8,057.1 |
| `smoke_mask_frac` | 1.000 | 11,620.0 | 2,881.7 |
| `fire_count_250km` | 1.000 | 6,839.0 | 4,057.0 |
| `injection_height` | 0.001 | 4,416.1 | 2,818.8 |
| `aod047` | 0.299 | 4,111.9 | 2,000.1 |
| `fire_frp_sum_100km` | 1.000 | 3,476.3 | 1,280.9 |
| `hms_any` | 0.999 | 1,150.2 | 1,675.6 |
| `aod_missing` | 1.000 | 0.0 | 0.0 |
| `fire_count_25km` | 1.000 | 0.0 | 0.0 |
| `fire_count_50km` | 1.000 | 0.0 | 0.0 |
| `fire_count_100km` | 1.000 | 0.0 | 0.0 |
| `fire_frp_sum_25km` | 1.000 | 0.0 | 0.0 |
| `fire_frp_sum_50km` | 1.000 | 0.0 | 0.0 |

Monthly completeness is in `feature_reduction_monthly.json`.

## Ranking within correlation groups (suggestion only)

- aod055, aod047: `aod055`, `aod047`
- aod_nbr_25km_d0, aod_nbr_50km_d0: `aod_nbr_50km_d0`, `aod_nbr_25km_d0`
- aod_nbr_25km_d0_n, aod_nbr_50km_d0_n: `aod_nbr_50km_d0_n`, `aod_nbr_25km_d0_n`
- merra_bccmass, merra_bcexttau: `merra_bcexttau`, `merra_bccmass`
- merra_ducmass25, merra_duexttau: `merra_ducmass25`, `merra_duexttau`
- merra_qlml, merra_tlml: `merra_tlml`, `merra_qlml`
- merra_ulml, merra_u10m, merra_u50m: `merra_ulml`, `merra_u10m`, `merra_u50m`
- merra_vlml, merra_v10m, merra_v50m: `merra_v10m`, `merra_vlml`, `merra_v50m`
- merra_speed, merra_speedmax: `merra_speed`, `merra_speedmax`
- fire_count_25km, fire_frp_sum_25km: `fire_count_25km`, `fire_frp_sum_25km`
- fire_count_50km, fire_frp_sum_50km: `fire_count_50km`, `fire_frp_sum_50km`
- fire_count_100km, fire_frp_sum_100km: `fire_frp_sum_100km`, `fire_count_100km`
- fire_count_250km, fire_frp_sum_250km: `fire_frp_sum_250km`, `fire_count_250km`
- fire_count_500km, fire_frp_sum_500km: `fire_count_500km`, `fire_frp_sum_500km`
- hms_any, hms_n_polygons: `hms_n_polygons`, `hms_any`
