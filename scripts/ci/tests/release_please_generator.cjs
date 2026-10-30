// Run with NODE_PATH pointing at an isolated release-please@17.1.3 + ajv@8 install.
// No authenticated API client or remote writes: all repository reads are fixture data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {Manifest} = require('release-please');
const {Version} = require('release-please/build/src/version');
const {Simple} = require('release-please/build/src/strategies/simple');
const {DefaultVersioningStrategy} = require('release-please/build/src/versioning-strategies/default');
const {TagName} = require('release-please/build/src/util/tag-name');
const {parseConventionalCommits} = require('release-please/build/src/commit');
const Ajv = require('ajv');
const addFormats = require('ajv-formats');
const root = path.resolve(__dirname, '../../..');
const config = JSON.parse(fs.readFileSync(path.join(root, 'release-please-config.json')));
assert.equal(require('release-please/package.json').version, '17.1.3');
const schema = require('release-please/schemas/config.json');
const ajv = new Ajv({strict: false});
addFormats(ajv);
const validate = ajv.compile(schema);
assert.ok(validate(config), JSON.stringify(validate.errors));
assert.equal(config['bootstrap-sha'], 'c76bb96e2284c8404b79daabe6df5fb10d11594b');
const pkg = config.packages['.'];
const github = {
  repository: {owner: 'example', repo: 'cuscan', defaultBranch: 'main'},
  getFileJson: async name => JSON.parse(fs.readFileSync(path.join(root, name), 'utf8')),
  getFileContentsOnBranch: async name => ({parsedContent: fs.readFileSync(path.join(root, name), 'utf8')}),
  async *releaseIterator() {},
  async *tagIterator() {},
  async *mergeCommitIterator() {
    yield {sha: 'a'.repeat(40), message: 'feat(ci): automate reviewed releases', files: ['README.md']};
    yield {sha: config['bootstrap-sha'], message: 'feat!: EXCLUDED_BOUNDARY', files: ['README.md']};
    yield {sha: 'c'.repeat(40), message: 'feat: EXCLUDED_OLD_HISTORY', files: ['README.md']};
  },
};
async function proposal(old, message, force = false) {
  const strategy = new Simple({
    github, targetBranch: 'main', packageName: 'cuscan',
    includeComponentInTag: false, includeVInTag: true,
    bumpMinorPreMajor: true, bumpPatchForMinorPreMajor: false,
    versioningStrategy: new DefaultVersioningStrategy({bumpMinorPreMajor: true, bumpPatchForMinorPreMajor: false}),
    releaseAs: force ? '0.1.0' : undefined,
    extraFiles: pkg['extra-files'], changelogSections: pkg['changelog-sections'],
  });
  const commits = parseConventionalCommits([{sha: 'a'.repeat(40), message, files: ['README.md']}]);
  return strategy.buildReleasePullRequest(commits, old ? {
    tag: new TagName(Version.parse(old)), sha: 'b'.repeat(40), notes: '',
  } : undefined);
}
(async () => {
  const cases = [
    ['0.1.0', 'fix: repair', '0.1.1'],
    ['0.1.0', 'feat: add capability', '0.2.0'],
    ['0.1.0', 'feat!: change API', '0.2.0'],
    ['1.0.0', 'feat!: change API', '2.0.0'],
    ['0.1.0', 'docs: instructions', undefined],
    ['0.1.0', 'chore: cleanup', undefined],
    ['0.1.0', 'test: checks', undefined],
  ];
  for (const [old, message, expected] of cases) {
    assert.equal((await proposal(old, message))?.version?.toString(), expected, message);
  }
  const manifest = await Manifest.fromManifest(github, 'main');
  const requests = await manifest.buildPullRequests();
  assert.equal(requests.length, 1);
  const updates = requests[0].updates;
  const content = name => updates.find(update => update.path === name).updater.updateContent(
    fs.readFileSync(path.join(root, name), 'utf8'));
  assert.equal(content('version.txt'), '0.1.0\n');
  assert.match(content('Cargo.toml'), /version = "0\.1\.0"/);
  assert.deepEqual(JSON.parse(content('.release-please-manifest.json')), {'.': '0.1.0'});
  const changelog = content('CHANGELOG.md');
  assert.equal((changelog.match(/^## 0\.1\.0 \(/gm) || []).length, 1);
  assert.ok(!changelog.includes('EXCLUDED_'));
  assert.ok(changelog.includes('Pre-automation project baseline'));
  const next = await proposal('0.1.0', 'fix: repair publishing');
  const nextLog = next.updates.find(update => update.path === 'CHANGELOG.md').updater.updateContent(changelog);
  assert.match(nextLog, /^## \[0\.1\.1\]\(https:\/\/github.com\/example\/cuscan\/compare\/v0\.1\.0\.\.\.v0\.1\.1\) \(\d{4}-\d{2}-\d{2}\)$/m);
  assert.equal((nextLog.match(/^## 0\.1\.0 \(/gm) || []).length, 1);
  console.log('Pinned schema, exclusive bootstrap, first proposal, workspace updater and version policy passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
