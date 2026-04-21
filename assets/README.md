# Assets

This directory stores downloaded runtime assets for CLUTCH.

Keep this file in the repo. The bundle installer replaces only the payload entries
inside `assets/`, so local docs and placeholder files can stay tracked.

Release-local assets currently include:

- `mano_v1_2/` and MANO mesh helpers for hand reconstruction/rendering
- `egovid5m_release/` with the minimum dataset-derived file currently kept for
  released prompt inference:
  - `_aux_exp/text_processing_valid_index_nset_72_tr0_013_rot0_2/mean_std_w_s20.npy`

The full EgoVid5M motion arrays and annotations are still external dataset content and
are not bundled here.

Install the uploaded asset bundle with:

```bash
./download_assets.sh --url <uploaded-assets-zip-url>
```
