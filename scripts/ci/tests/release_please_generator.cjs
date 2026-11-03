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
assert.equal(config['bootstrap-sha'], undefined);
const pkg = config.packages['.'];
assert.equal(pkg['release-as'], undefined);
const released = JSON.parse(fs.readFileSync(path.join(root, '.release-please-manifest.json'), 'utf8'))['.'];
const releaseSha = 'b'.repeat(40);
const github = {
  repository: {owner: 'example', repo: 'cuscan', defaultBranch: 'main'},
  getFileJson: async name => JSON.parse(fs.readFileSync(path.join(root, name), 'utf8')),
  getFileContentsOnBranch: async name => ({parsedContent: fs.readFileSync(path.join(root, name), 'utf8')}),
  async *releaseIterator() {
    yield {tagName: `v${released}`, sha: releaseSha, notes: '', url: 'https://example.invalid'};
  },
  async *tagIterator() {},
  async *mergeCommitIterator() {
    yield {sha: 'a'.repeat(40), message: 'feat: add capability', files: ['README.md']};
    yield {sha: releaseSha, message: `chore(main): release ${released}`, files: ['README.md']};
    yield {sha: 'c'.repeat(40), message: 'feat: EXCLUDED_OLD_HISTORY', files: ['README.md']};
  },
};
async function proposal(old, message) {
  const bump = {
    bumpMinorPreMajor: pkg['bump-minor-pre-major'],
    bumpPatchForMinorPreMajor: pkg['bump-patch-for-minor-pre-major'],
  };
  const strategy = new Simple({
    github, targetBranch: 'main', packageName: pkg['package-name'],
    includeComponentInTag: pkg['include-component-in-tag'], includeVInTag: pkg['include-v-in-tag'],
    ...bump, versioningStrategy: new DefaultVersioningStrategy(bump),
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
    ['0.1.0', 'feat: add capability', '0.1.1'],
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
  const next = Version.parse(released);
  next.patch += 1;
  const nextVersion = next.toString();
  assert.equal(content('version.txt'), `${nextVersion}\n`);
  assert.match(content('Cargo.toml'), new RegExp(`version = "${nextVersion.replaceAll('.', '\\.')}"`));
  assert.deepEqual(JSON.parse(content('.release-please-manifest.json')), {'.': nextVersion});
  const changelog = content('CHANGELOG.md');
  assert.ok(!changelog.includes('EXCLUDED_'));
  assert.ok(changelog.includes(`compare/v${released}...v${nextVersion}`));
  assert.equal((changelog.match(new RegExp(`^## \\[?${released.replaceAll('.', '\\.')}`, 'gm')) || []).length, 1);
  console.log('Pinned schema, post-release proposal, workspace updater and version policy passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
