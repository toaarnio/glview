# Refactor Handoff

Last updated: 2026-05-04

## Current status

The branch contains the recent `FileList` cleanup plus the loader responsiveness fix.

Verification at this point:
- `make test`
- Result: `81 passed`

## Recent commits

Newest first:

- `c1c4cba` `fix: avoid blocking ui during image decode`
- `025abb2` `refactor: drop filelist field view compatibility api`
- `70dae3e` `fix: restore colorful placeholder textures`
- `7064a1a` `fix: drop images that fail during loading`
- `1a8c841` `refactor: store filelist state per entry`
- `414bbd9` `refactor: remove dead helpers and share border crop`
- `17e0d66` `refactor: remove standalone import fallbacks`

## Current worktree notes

The old uncommitted `FileList` API cleanup described in earlier notes is now committed.

Local-only state currently worth remembering:
- `glview/version.py` is still bumped to `1.22.0` in the working tree for testing and is not meant to be committed yet.
- `REFACTOR_HANDOFF.md` itself is an untracked local handoff note.

## Important completed changes

- Added headless tests for:
  - texture lifecycle
  - file catalog lifecycle
  - image loader behavior
  - UI state transitions
  - UI smoke wiring
  - renderer helper contracts
- Split image slot state from image payload storage.
- Replaced direct loader/UI payload mutation with queued handoff plus slot token checks.
- Added snapshot-based `FileList` reads for UI/render code.
- Moved texture cache ownership into the renderer.
- Converted `FileList` storage from parallel arrays to per-entry records.
- Removed the remaining `FileList` field-view compatibility API.
- Restored colorful placeholder textures.
- Drop images that fail during loading instead of keeping bad state around.
- Moved image decode work out from under the `FileList` mutex so UI state reads are not blocked by loader I/O/decode.

## Current architecture notes

- `PygletUI` owns:
  - `state: ViewerState`
  - `config: ViewConfigState`
  - `ops: UIOperations`
- `GLRenderer` owns:
  - render orchestration
  - renderer texture manager
  - tile render target
- `FileList` now stores mutable `FileEntry` records.
- `FileList.snapshot()` returns immutable `FileEntrySnapshot` records.
- `ImageProvider` now snapshots enough request state to decode outside the catalog lock and republishes results only if the slot token is still current.

## Remaining likely work

These are the remaining items that still look relevant now:

1. Evaluate whether `glview/glrenderer.py` still needs `self.ctx.finish()` on every redraw.
2. Decide whether the local `1.22.0` version bump should be reverted or committed later as a release-only change.
3. Refresh this note again once the renderer-side work is settled.

## Things deliberately not done

- Did not introduce `EasyDict`.
- Did not replace all dicts with object-style wrappers.
- Did not remove `ctx.finish()` yet.
- Did not touch `argv.py` beyond existing exclusions.
- Did not collapse all renderer uniform dict usage into a custom object.
- Did not replace `config = types.SimpleNamespace()` with a typed dataclass.
- Did not replace lambdas with functools.partial or direct expressions.
- Did not split postprocess uniforms into core/tone/debug sections.

## If resuming immediately

Start with:

1. `git status --short`
2. `make test`
3. Inspect `glview/glrenderer.py` around `ctx.finish()` before choosing the next performance-oriented change.
