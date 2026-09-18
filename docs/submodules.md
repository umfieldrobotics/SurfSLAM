# Submodules and patches

SurfSLAM depends on third-party model code that we pull in as git submodules. Each
submodule points at its **original upstream repository**, pinned to an exact commit.
Our modifications are not vendored into forks; they live as one patch file per
submodule in [`patches/`](../patches/) and are applied to the submodule working tree
by [`scripts/setup_submodules.sh`](../scripts/setup_submodules.sh).

```bash
./scripts/setup_submodules.sh          # init submodules and apply patches (idempotent)
./scripts/setup_submodules.sh --check  # report state without changing anything
```

The patch file is matched to its submodule by name: `patches/<name>.patch` is applied
inside `submodules/<name>`. Both submodules are configured with `ignore = dirty` in
`.gitmodules`, since a patched working tree is the intended steady state.

## Current submodules

| Submodule | Upstream | Patch |
| --- | --- | --- |
| `submodules/DEFOM-Stereo` | [Insta360-Research-Team/DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo) | Renames the `core` package to `defom_core` so it can be imported alongside other model code without name collisions, and adds a `depth_anything_checkpoint` option to `DefomEncoder` for loading a custom (e.g. underwater-finetuned) Depth Anything V2 checkpoint. |
| `submodules/SuperGluePretrainedNetwork` | [magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) | Extends the match visualization utilities with a `no_lines` mode (distinct per-match colors/markers instead of connecting lines), used by `examples/place_recognition/video_match.py`. |

## Updating a submodule pin

```bash
cd submodules/<name>
git stash                    # set aside the applied patch
git fetch origin && git checkout <new-commit>
git stash pop                # re-apply our changes; resolve conflicts if any
cd ../..
git add submodules/<name>    # record the new pin
```

If the patch no longer applies cleanly on the new commit, fix up the working tree and
regenerate the patch as below.

## Regenerating a patch

After editing files inside a submodule, capture the working-tree diff:

```bash
git -C submodules/<name> diff > patches/<name>.patch
```

Keep patches minimal: source changes only — no `__pycache__`, assets, or unrelated
formatting churn.
