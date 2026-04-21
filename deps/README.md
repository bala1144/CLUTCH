# Deps

This directory stores external runtime dependencies that are not Python packages.

Current release bootstrap:
- `./download_deps.sh` downloads:
  - `deps/t2m/` for the TM2T evaluator used by quantitative T2M and M2T metrics
  - `deps/glove/` for the word vectorizer used by EgoVid5M dataloading
- `./download_deps.sh` also creates placeholders for:
  - `deps/mGPT_instructions/`
  - `deps/transforms/`
  - `deps/smpl/`

Notes:
- T2M and M2T inference smoke tests can run without `deps/t2m/`, but quantitative metrics will be skipped.
- `deps/glove/` is required by the current EgoVid5M datamodule.
- The placeholder directories remain for backward-compatible config structure and can be removed later if the code path is simplified further.
