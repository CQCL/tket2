import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { Manifest } from 'release-please';
import { includeDependencyCommits, includePythonDependencies } from './release.mjs';

const root = new URL('../../', import.meta.url);
const read = path => readFile(new URL(path, root), 'utf8');

async function candidates(commits, mapDependencies = true) {
  const config = JSON.parse(await read('release-please-config.json'));
  const versions = JSON.parse(await read('.release-please-manifest.json'));
  // Keep the regression baseline at the release preceding the recorded fixes.
  versions['tket/tket-py'] = '0.15.11';
  // Exercise the real manifest and Python strategy for the wheel and one
  // independent package, using local files and a read-only GitHub fixture.
  const paths = ['tket/tket-py', 'tket/qis-compiler'];
  const github = {
    repository: { owner: 'Quantinuum', repo: 'tket2' },
    async getFileJson(path) {
      if (path === 'release-please-config.json') {
        return { ...config, packages: Object.fromEntries(paths.map(p => [p, config.packages[p]])) };
      }
      return Object.fromEntries(paths.map(p => [p, versions[p]]));
    },
    async *releaseIterator() {
      for (const path of paths) {
        yield { tagName: `${config.packages[path].component}-v${versions[path]}`, sha: `release-${path}` };
      }
    },
    async *mergeCommitIterator(branch, options) {
      assert.equal(branch, 'main');
      assert.equal(options.backfillFiles, true);
      yield* commits;
      for (const path of paths) {
        if (!commits.some(commit => commit.sha === `release-${path}`)) {
          yield { sha: `release-${path}`, message: 'chore: release', files: [] };
        }
      }
    },
    async getFileContentsOnBranch(path) {
      return { parsedContent: await read(path) };
    },
    async findFilesByFilenameAndRef() { return []; },
  };
  const manifest = await Manifest.fromManifest(
    mapDependencies ? includeDependencyCommits(github) : github, 'main',
  );
  return manifest.buildPullRequests();
}

const rustFix = {
  sha: 'b741472137760e0556fc5173587ad96e794d7315',
  message: 'fix(pytket): preserve opaque bit output dependencies (#2085)',
  files: ['tket/tket-rs/src/serialize/pytket/encoder.rs'],
};

test('Rust-only fixes create a Python release PR with the original changelog entry', async () => {
  assert.deepEqual(await candidates([rustFix], false), []);
  const prs = await candidates([rustFix]);
  assert.equal(prs.length, 1);
  const pr = prs[0];
  assert.match(pr.title.toString(), /tket-py 0\.15\.12/);
  assert.equal(pr.headRefName.toString(), 'release-please--branches--main--components--tket-py');
  const changelog = pr.updates.find(u => u.path === 'tket/tket-py/CHANGELOG.md');
  const text = changelog.updater.updateContent(await read(changelog.path));
  assert.match(text, /preserve opaque bit output dependencies/);
  assert.match(text, /b741472/);
  const pyproject = pr.updates.find(u => u.path === 'tket/tket-py/pyproject.toml');
  assert.match(pyproject.updater.updateContent(await read(pyproject.path)), /version = "0\.15\.12"/);
  const init = pr.updates.find(u => u.path === 'tket/tket-py/tket/__init__.py');
  assert.match(init.updater.updateContent(await read(init.path)), /__version__ = "0\.15\.12"/);
  const lock = pr.updates.find(u => u.path === 'uv.lock');
  assert.match(lock.updater.updateContent(await read(lock.path)), /name = "tket"\nversion = "0\.15\.12"/);
});

test('existing fixes since 0.15.11 are included without a new source commit', async () => {
  const commits = JSON.parse(await read('scripts/release/fixtures/rust-fixes.json'));
  const prs = await candidates(commits);
  const python = prs.find(pr => pr.title.toString().includes('tket-py'));
  assert.ok(python);
  const update = python.updates.find(u => u.path.endsWith('CHANGELOG.md'));
  const changelog = update.updater.updateContent(await read(update.path));
  assert.match(changelog, /preserve opaque bit output dependencies/);
  assert.match(changelog, /Avoid generated helper symbol collisions/);
});

test('independent Python packages remain independent', async () => {
  for (const file of ['tket/tket-eccs/pyproject.toml', 'tket/tket-exts/src/tket_exts/__init__.py', 'docs/index.md']) {
    assert.deepEqual(await candidates([{ sha: 'abc', message: 'fix: unrelated', files: [file] }]), []);
  }
  const prs = await candidates([{ sha: 'abc', message: 'fix: compiler', files: ['tket/qis-compiler/rust/lib.rs'] }]);
  assert.equal(prs.length, 1);
  assert.match(prs[0].title.toString(), /qis-compiler/);
});

test('dependency and shared build changes retain commit identity without mutating inputs', () => {
  for (const file of ['tket/tket-qsystem/src/lib.rs', 'tket/tket1-passes/Cargo.toml', 'pg-libs/tk-pg-core/src/lib.rs', 'Cargo.toml', 'Cargo.lock', 'hugrenv.lock', 'rust-toolchain.toml', 'pyproject.toml', 'uv.lock', '.github/workflows/python-wheels.yml']) {
    const commit = { ...rustFix, files: [file] };
    const mapped = includePythonDependencies(commit);
    assert.deepEqual(mapped.files, [file, 'tket/tket-py/Cargo.toml']);
    assert.equal(mapped.sha, commit.sha);
    assert.equal(mapped.message, commit.message);
    assert.deepEqual(commit.files, [file]);
  }
  const mixed = { ...rustFix, files: [...rustFix.files, 'tket/tket-py/tket/__init__.py'] };
  assert.equal(includePythonDependencies(mixed), mixed);
  assert.throws(() => includePythonDependencies({ sha: 'missing' }), /Missing changed files/);
});

test('dependency mapping respects the existing Python release boundary', async () => {
  const prs = await candidates([
    { sha: 'release-tket/tket-py', message: 'chore: release', files: [] },
    rustFix,
  ]);
  assert.deepEqual(prs, []);
});
