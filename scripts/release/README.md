# Python release automation

Run `npm ci --ignore-scripts` and `npm test` in this directory to check release
detection without GitHub credentials or writes. The workflow runs these tests
before allowing release automation to run on `main`.

`release.mjs` uses the pinned release-please library with the repository's existing
manifest and Python strategies. Before release-please splits commits by package
path, it also attributes changes to the Python wheel's local Rust dependencies
and shared build inputs to `tket/tket-py`. The extra path exists only in memory;
commit SHAs, messages, and PR links remain intact for changelog generation.
Release-please still chooses versions and stops at each package's existing release
boundary. This includes Rust fixes already merged since the last Python release.

The dependency directories and shared build files are listed at the top of
`release.mjs`. Update these when adding local dependencies to the Python extension.
The `pg-libs` directory is treated conservatively as one dependency group. Changes
confined to the other Python packages or the QIS compiler do not trigger a `tket`
release. Changes to shared workspace manifests and lockfiles do apply to `tket`.

The regression fixture records the file lists and subjects of #2085 and #2081.
Tests pass these through the real release-please manifest and Python strategy and
check the resulting release PR, changelog, and version updates.

The workflow can be run manually after this change reaches `main`. It will use
unreleased commits already in history; no additional source commit is needed.
The resulting draft release PR still needs review and merging. Running
`npm run release` is a write operation: it creates releases for merged release
PRs and opens or updates release PRs, using `GITHUB_REPOSITORY` and `GITHUB_TOKEN`.
