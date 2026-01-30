# Release Process

This document describes the steps to create a release tag for Spectra.

## Prerequisites

- Working tree is clean (no uncommitted changes)
- All tests pass
- Version number is correct in `src/metrics_lie/__init__.py`

## Release Steps

1. **Verify working tree is clean:**
   ```bash
   git status
   ```
   Ensure there are no uncommitted changes.

2. **Run the full test suite:**
   ```bash
   .\.venv\Scripts\python.exe -m pytest -q
   ```
   All tests must pass.

3. **Verify version:**
   ```bash
   .\.venv\Scripts\spectra.exe --version
   ```
   Confirm the output matches the intended release version (e.g., "Spectra 0.2.0").

4. **Create the release tag:**
   ```bash
   git tag v0.2.0
   ```
   Replace `v0.2.0` with the actual version number.

5. **Push the tag to remote:**
   ```bash
   git push origin v0.2.0
   ```

## Bumping Version for Next Release

To prepare for the next release:

1. Edit `src/metrics_lie/__init__.py`:
   ```python
   __version__ = "0.3.0"  # Update to next version
   ```

2. Update `CHANGELOG.md` with a new section for the next version.

3. Commit the version bump and changelog:
   ```bash
   git add src/metrics_lie/__init__.py CHANGELOG.md
   git commit -m "Bump version to 0.3.0"
   ```

4. Follow the release steps above to create the new tag.

## Notes

- Version is the single source of truth in `src/metrics_lie/__init__.py`
- Do not duplicate version strings elsewhere
- Tags follow semantic versioning: `v<major>.<minor>.<patch>`

