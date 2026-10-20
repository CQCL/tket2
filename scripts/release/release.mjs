import { pathToFileURL } from 'node:url';
import { GitHub, Manifest } from 'release-please';

// Local Rust dependencies compiled into tket's Python extension. Keep this list
// in sync with tket/tket-py/Cargo.toml, including transitive workspace crates.
const dependencyDirectories = [
  'tket/tket-rs/',
  'tket/tket-qsystem/',
  'tket/tket1-passes/',
  'pg-libs/',
];
const buildFiles = new Set([
  'Cargo.toml',
  'Cargo.lock',
  'rust-toolchain.toml',
  'hugrenv.lock',
  'pyproject.toml',
  'uv.lock',
  '.github/workflows/python-wheels.yml',
]);

export function includePythonDependencies(commit) {
  if (!commit.files) {
    throw new Error(`Missing changed files for commit ${commit.sha}`);
  }
  const affectsWheel = commit.files.some(
    file => buildFiles.has(file) || dependencyDirectories.some(dir => file.startsWith(dir)),
  );
  if (!affectsWheel || commit.files.some(file => file.startsWith('tket/tket-py/'))) {
    return commit;
  }
  // Release-please splits commits by package directory. Attribute the original
  // commit to the bindings too, without changing its SHA, message, or PR link.
  // This is only an in-memory change to detection; no repository file is edited.
  return { ...commit, files: [...commit.files, 'tket/tket-py/Cargo.toml'] };
}

export function includeDependencyCommits(github) {
  const iterate = github.mergeCommitIterator.bind(github);
  github.mergeCommitIterator = async function* (...args) {
    for await (const commit of iterate(...args)) {
      yield includePythonDependencies(commit);
    }
  };
  return github;
}

async function main() {
  const [owner, repo] = process.env.GITHUB_REPOSITORY.split('/');
  const github = includeDependencyCommits(await GitHub.create({
    owner,
    repo,
    token: process.env.GITHUB_TOKEN,
  }));
  // Reload after creating releases, as release-please-action does, so PR
  // generation sees newly created tags and releases.
  const releases = await Manifest.fromManifest(github, 'main');
  await releases.createReleases();
  const pullRequests = await Manifest.fromManifest(github, 'main');
  await pullRequests.createPullRequests();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
