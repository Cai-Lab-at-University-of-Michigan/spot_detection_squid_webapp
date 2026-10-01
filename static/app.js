'use strict';

// ── small helpers ────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function fmtBytes(n) {
  if (!n) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1000)));
  return `${(n / 1000 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
}
function fmtDuration(s) {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j.detail) msg = j.detail; } catch (_) { /* not JSON */ }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return res.json();
}
const postJSON = (path, body) => api(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (_) { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) { /* storage off */ } },
};

// ── state ────────────────────────────────────────────────────────────────────
const S = {
  schema: null,
  inputMode: 'remote',
  selection: null,          // {kind: 'folder'|'single', root_name, tiles: [{key, relpath, size, file}], sidecars: [{relpath, file}]}
  uploaders: {},            // jobId -> Uploader (only for uploads running in this tab)
  jobs: [],
  selectedJob: null,
  log: { jobId: null, offset: 0, raw: '' },
};

// ── parameter form ───────────────────────────────────────────────────────────
function renderForm(values) {
  const form = $('#params-form');
  form.innerHTML = '';
  const groups = {};
  for (const f of S.schema.fields) (groups[f.group] ||= []).push(f);
  const openGroups = new Set(['Output', 'Detection', 'Stitching & genes']);
  for (const [group, fields] of Object.entries(groups)) {
    const det = document.createElement('details');
    det.className = 'card';
    det.open = openGroups.has(group);
    det.innerHTML = `<summary>${esc(group)}</summary>`;
    for (const f of fields) det.appendChild(renderField(f, values?.[f.name] ?? f.default));
    form.appendChild(det);
  }
}

function renderField(f, value) {
  const wrap = document.createElement('label');
  wrap.className = 'field';
  wrap.dataset.name = f.name;
  const help = f.help ? `<span class="help">${esc(f.help)}</span>` : '';
  if (f.kind === 'bool') {
    wrap.classList.add('check');
    wrap.innerHTML = `<input type="checkbox" name="${f.name}" ${value ? 'checked' : ''}>
      <span><span class="label">${esc(f.name)}</span>${help}</span>`;
    return wrap;
  }
  let control;
  if (f.kind === 'choice') {
    control = `<select name="${f.name}">${f.choices.map((c) =>
      `<option ${c === value ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>`;
  } else {
    const ph = f.kind === 'number' && f.nullable ? 'blank = None'
      : f.kind === 'literal' || f.kind === 'dir' || f.kind === 'file' ? 'blank = None' : '';
    control = `<input type="text" name="${f.name}" value="${esc(value)}" placeholder="${ph}">`;
  }
  if (f.kind === 'dir' || f.kind === 'file') {
    control = `<div class="path-row">${control}
      <button type="button" class="small" data-browse="${f.kind}">Browse remote…</button>
      ${f.kind === 'file' ? '<button type="button" class="small" data-upload>Upload local…</button><input type="file" hidden>' : ''}
    </div><span class="help uploaded muted"></span>`;
  }
  wrap.innerHTML = `<span class="label">${esc(f.name)}</span>${control}${help}`;

  const input = wrap.querySelector(`[name="${f.name}"]`);
  wrap.querySelector('[data-browse]')?.addEventListener('click', (e) => {
    e.preventDefault();
    openBrowser({ wantFile: f.kind === 'file', start: input.value, onChoose: (p) => { input.value = p; saveForm(); } });
  });
  const fileInput = wrap.querySelector('input[type=file]');
  wrap.querySelector('[data-upload]')?.addEventListener('click', (e) => { e.preventDefault(); fileInput.click(); });
  fileInput?.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    const note = wrap.querySelector('.uploaded');
    note.textContent = `Uploading ${file.name}…`;
    try {
      const r = await api(`/api/files?name=${encodeURIComponent(file.name)}`, { method: 'PUT', body: file });
      input.value = r.path;
      note.textContent = `Uploaded from this computer: ${file.name} (${fmtBytes(r.size)})`;
      saveForm();
    } catch (err) {
      note.textContent = `Upload failed: ${err.message}`;
    }
    fileInput.value = '';
  });
  return wrap;
}

function formValues() {
  const out = {};
  for (const f of S.schema.fields) {
    const el = $('#params-form').querySelector(`[name="${f.name}"]`);
    out[f.name] = f.kind === 'bool' ? el.checked : el.value;
  }
  return out;
}

function saveForm() {
  store.set('form', {
    params: formValues(), squid: $('#squid-folder').value, mode: S.inputMode,
    prefetch: $('#prefetch-tiles').value, parallel: $('#parallel-uploads').value,
    globusEndpoint: $('#globus-endpoint').value || S.globusEndpoint, globusPath: $('#globus-path').value,
    globusVerify: $('#globus-verify').checked,
  });
}

function setInputMode(mode) {
  S.inputMode = mode;
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.mode === mode));
  $('#input-remote').classList.toggle('hidden', mode !== 'remote');
  $('#input-globus').classList.toggle('hidden', mode !== 'globus');
  $('#input-upload').classList.toggle('hidden', mode !== 'upload');
  $('#prefetch-field').classList.toggle('hidden', mode === 'remote');
  if (mode === 'globus' && S.schema) refreshGlobus(false);
}

// ── Globus: readiness checks, endpoint choice, folder summary ───────────────
async function refreshGlobus(force) {
  const el = $('#globus-checks');
  if (force) el.innerHTML = '<li class="muted">Checking…</li>';
  try {
    S.globus = await api(`/api/globus/status${force ? '?refresh=1' : ''}`);
  } catch (err) {
    el.innerHTML = `<li class="error">${esc(err.message)}</li>`;
    return;
  }
  renderGlobus();
}

function renderGlobus() {
  const g = S.globus;
  if (!g) return;
  const r = g.remote;
  const item = (ok, text, fix = '') => `<li><span class="mark ${ok ? 'yes' : 'no'}">${ok ? '✓' : '✗'}</span>
    <span>${text}${!ok && fix ? `<br>${fix}` : ''}</span></li>`;
  const remoteName = r.api ? esc(r.api.display_name) : 'this machine';
  const localOk = g.local_endpoints.some((e) => e.gcp_connected);
  const rows = [
    item(g.logged_in, g.logged_in ? `Globus login on the remote machine (${esc(g.identity)})` : 'Globus login on the remote machine',
      `Run this once in a terminal on the remote machine, then check again:<br><code>${esc(g.login_command)}</code>`),
    item(r.process.connected && r.api?.gcp_connected,
      `Globus Connect Personal on the remote machine${r.process.connected ? ` (${remoteName})` : ''}`,
      r.process.installed
        ? `${esc(r.process.text.split('\n')[0] || 'not connected')} <button type="button" class="small" id="globus-start-remote">Start it</button>`
        : 'Not installed on the remote machine.'),
    item(r.jobs_dir_writable, 'Remote machine can receive files into the jobs folder',
      `Globus Connect Personal there is not allowed to write to <code>${esc(r.jobs_dir)}</code>. Add it as writable in ~/.globusonline/lta/config-paths.`),
    item(g.logged_in && localOk,
      localOk ? 'Globus Connect Personal on your computer' : 'Globus Connect Personal on your computer',
      g.logged_in ? (g.local_endpoints.length
        ? 'None of your computers\' endpoints is connected. Start Globus Connect Personal on your computer (and un-pause it), then check again.'
        : 'No Globus Connect Personal endpoint found for your account. Install it on your computer and log in with the same Globus account.')
        : 'Needs the Globus login above first.'),
  ];
  $('#globus-checks').innerHTML = `<ul class="checks">${rows.join('')}</ul>
    ${g.error ? `<div class="error">${esc(g.error)}</div>` : ''}
    <button type="button" class="small" id="globus-recheck">Check again</button>`;
  $('#globus-recheck').addEventListener('click', () => refreshGlobus(true));
  $('#globus-start-remote')?.addEventListener('click', async (e) => {
    e.target.disabled = true; e.target.textContent = 'Starting…';
    try { S.globus = await postJSON('/api/globus/start-remote', {}); } catch (err) { alert(err.message); }
    renderGlobus();
  });

  const sel = $('#globus-endpoint');
  const want = sel.value || S.globusEndpoint;
  sel.innerHTML = g.local_endpoints.length
    ? g.local_endpoints.map((e) => `<option value="${esc(e.id)}">${esc(e.display_name)} — ${e.gcp_connected ? (e.gcp_paused ? 'paused' : 'connected') : 'offline'}</option>`).join('')
    : '<option value="">(none found)</option>';
  const pick = g.local_endpoints.find((e) => e.id === want) || g.local_endpoints.find((e) => e.gcp_connected);
  if (pick) sel.value = pick.id;
}

async function inspectGlobusPath() {
  const el = $('#globus-summary');
  const endpoint = $('#globus-endpoint').value;
  const path = $('#globus-path').value.trim();
  S.globusInfo = null;
  if (!endpoint || !path) { el.className = 'summary muted'; el.textContent = 'Nothing chosen yet.'; return; }
  el.className = 'summary muted';
  el.textContent = 'Listing through Globus…';
  try {
    const r = await api(`/api/globus/inspect?endpoint=${encodeURIComponent(endpoint)}&path=${encodeURIComponent(path)}`);
    S.globusInfo = r;
    if (r.layout === 'single') {
      el.className = 'summary ok';
      el.innerHTML = `<b>${esc(r.root_name)}</b> · ${fmtBytes(r.total_bytes)} · fetched whole, then processed as a single tile.`;
      return;
    }
    const have = new Set(r.sidecars);
    const side = S.schema.sidecar_names.map((n) => `${have.has(n) ? '✓' : '✗'} ${esc(n)}`).join(' · ');
    const warnings = [];
    if (!have.has('coordinates.csv') && !have.has('0/coordinates.csv') && r.n_tiles > 1) warnings.push('No coordinates.csv: stitching more than one tile will fail.');
    if (r.layout === 'legacy') {
      if (!$('[name=resolution]').value.trim()) warnings.push('Old-layout TIFFs carry no pixel size: set resolution (Stitching &amp; genes) before starting.');
      const uneven = Object.entries(r.uneven_z || {});
      if (uneven.length) warnings.push(`Some tiles have fewer z-slices than others (${uneven.map(([ch, ns]) => `${esc(ch)}: ${ns.join('/')}`).join('; ')}). The pipeline uses the slices every channel of a tile has.`);
    }
    el.className = 'summary ok';
    el.innerHTML = `<b>${esc(r.root_name)}</b> · ${r.layout === 'legacy' ? 'old layout (one TIFF per slice)' : 'one OME-TIFF per tile'}
      · ${r.n_tiles} tiles${r.layout === 'legacy' ? ` (${r.n_files} files)` : ''} · ${fmtBytes(r.total_bytes)} total (largest tile ${fmtBytes(r.largest_tile)})<br>
      <span class="muted">${esc(r.root)}</span><br><span class="muted">Metadata: ${side}</span>
      ${warnings.map((w) => `<br><span class="error">${w}</span>`).join('')}`;
  } catch (err) {
    el.className = 'summary error';
    el.textContent = err.message;
  }
}

// ── local folder / file selection ────────────────────────────────────────────
// A selection is a list of tiles, each a list of files: one OME-TIFF in the
// new layout, one TIFF per (z, channel) in the old one.
const TILE_RE = /^(?:ome_tiff\/)?R0_(\d+)\.ome\.tiff?$/i;
const LEGACY_RE = /^0\/R0_(\d+)_(\d+)_(.+)\.tiff?$/i;
const LEGACY_AT_ROOT_RE = /^R0_\d+_\d+_.+\.tiff?$/i;

function makeTiles(groups) {
  return [...groups.entries()]
    .map(([key, files]) => ({
      key,
      files: files.sort((a, b) => a.relpath.localeCompare(b.relpath)),
      size: files.reduce((a, f) => a + f.size, 0),
    }))
    .sort((a, b) => a.key - b.key);
}

function scanFolder(fileList) {
  const files = Array.from(fileList);
  if (!files.length) return null;
  const root = files[0].webkitRelativePath.split('/')[0];
  const ome = new Map();
  const legacy = new Map();
  const sidecars = [];
  let legacyAtRoot = 0;
  for (const file of files) {
    const rel = file.webkitRelativePath.split('/').slice(1).join('/');
    const entry = { relpath: rel, size: file.size, file };
    let m;
    if ((m = rel.match(TILE_RE))) {
      const key = Number(m[1]);
      if (!ome.has(key)) ome.set(key, [entry]);   // ignore a duplicate .tif/.tiff spelling
    } else if ((m = rel.match(LEGACY_RE))) {
      const key = Number(m[1]);
      if (!legacy.has(key)) legacy.set(key, []);
      legacy.get(key).push(entry);
    } else if (S.schema.sidecar_names.includes(rel)) {
      sidecars.push(entry);
    } else if (LEGACY_AT_ROOT_RE.test(rel)) {
      legacyAtRoot++;
    }
  }
  // A folder with both is read the new way, as the pipeline does.
  const layout = ome.size ? 'ome' : legacy.size ? 'legacy' : null;
  const sel = {
    kind: 'folder', layout, root_name: root, legacyAtRoot, sidecars,
    tiles: layout ? makeTiles(layout === 'ome' ? ome : legacy) : [],
  };
  if (layout === 'legacy') sel.zcounts = legacyDepths(sel.tiles);
  return sel;
}

// For an old-layout selection: {channel: Set of z-counts across tiles}, to
// warn about tiles with missing slices before anything is sent.
function legacyDepths(tiles) {
  const out = {};
  for (const t of tiles) {
    const perChannel = {};
    for (const f of t.files) {
      const ch = f.relpath.match(LEGACY_RE)[3];
      perChannel[ch] = (perChannel[ch] || 0) + 1;
    }
    for (const [ch, n] of Object.entries(perChannel)) (out[ch] ||= new Set()).add(n);
  }
  return out;
}

function showSelection() {
  const el = $('#selection-summary');
  const sel = S.selection;
  if (!sel) return;
  if (sel.kind === 'single') {
    const t = sel.tiles[0];
    el.className = 'summary ok';
    el.innerHTML = `<b>${esc(t.files[0].relpath)}</b> · ${fmtBytes(t.size)} · uploaded whole, then processed as a single tile.`;
    return;
  }
  if (!sel.tiles.length) {
    el.className = 'summary error';
    el.textContent = sel.legacyAtRoot
      ? `${sel.root_name}: this looks like the per-slice 0/ folder itself. Pick the acquisition folder that contains it, so the metadata files come along.`
      : `${sel.root_name}: no tiles found. Pick the acquisition folder (the one containing ome_tiff/, or 0/ for the old layout).`;
    return;
  }
  const total = sel.tiles.reduce((a, t) => a + t.size, 0);
  const have = new Set(sel.sidecars.map((s) => s.relpath));
  const side = S.schema.sidecar_names.map((n) => `${have.has(n) ? '✓' : '✗'} ${esc(n)}`).join(' · ');
  const noCoords = !have.has('coordinates.csv') && !have.has('0/coordinates.csv');
  const nFiles = sel.tiles.reduce((a, t) => a + t.files.length, 0);
  const warnings = [];
  if (noCoords && sel.tiles.length > 1) warnings.push('No coordinates.csv: stitching more than one tile will fail.');
  if (sel.layout === 'legacy') {
    if (!$('[name=resolution]').value.trim()) {
      warnings.push('Old-layout TIFFs carry no pixel size: set resolution (Stitching &amp; genes) before starting.');
    }
    const uneven = Object.entries(sel.zcounts).filter(([, ns]) => ns.size > 1);
    if (uneven.length) {
      warnings.push(`Some tiles have fewer z-slices than others (${uneven.map(([ch, ns]) =>
        `${esc(ch)}: ${[...ns].sort((a, b) => a - b).join('/')}`).join('; ')}). The pipeline uses the slices every channel of a tile has.`);
    }
  }
  el.className = 'summary ok';
  el.innerHTML = `<b>${esc(sel.root_name)}</b> · ${sel.layout === 'legacy' ? 'old layout (one TIFF per slice)' : 'one OME-TIFF per tile'}
    · ${sel.tiles.length} tiles${sel.layout === 'legacy' ? ` (${nFiles} files)` : ''} · ${fmtBytes(total)} total
    (largest tile ${fmtBytes(Math.max(...sel.tiles.map((t) => t.size)))})<br>
    <span class="muted">Metadata: ${side}</span>
    ${warnings.map((w) => `<br><span class="error">${w}</span>`).join('')}`;
}

// Matches a selection against a job's manifest -- right after creating the
// job, or when re-selecting the folder to resume an interrupted upload.
// Returns {files: {tileKey: [File by file index]}, sidecars: {relpath: File}}.
function matchSelection(sel, manifest) {
  const byRel = new Map();
  for (const t of sel.tiles) for (const f of t.files) byRel.set(f.relpath, f);
  const files = {};
  for (const t of manifest.tiles) {
    files[t.key] = t.files.map((mf) => {
      const local = byRel.get(mf.relpath);
      if (!local || local.size !== mf.size) {
        throw new Error(`${mf.relpath} is missing or a different size in the selected folder.`);
      }
      return local.file;
    });
  }
  const sidecars = {};
  for (const s of sel.sidecars) sidecars[s.relpath] = s.file;
  return { files, sidecars };
}

// ── uploader: keeps the remote's tile queue full ─────────────────────────────
class Uploader {
  constructor(job, files, sidecars, parallel) {
    this.job = job;
    this.files = files;          // tile key -> [File by file index]
    this.sidecars = sidecars;    // relpath -> File
    this.parallel = Math.max(1, parallel);
    this.queue = [];
    this.inflight = new Set();
    this.stopped = false;
    this.refreshing = null;
    this.lastRefresh = 0;
    this.sent = [];              // [time, bytes] for the speed estimate
    this.totalSent = 0;
    this.window = [];
    this.message = 'starting';
    this.error = null;
  }

  async start() {
    try {
      for (const rel of this.job.manifest.sidecars) {
        const file = this.sidecars[rel];
        if (!file) continue;
        this.message = `sending ${rel}`;
        await api(`/api/jobs/${this.job.id}/sidecar?path=${encodeURIComponent(rel)}`, { method: 'PUT', body: file });
      }
    } catch (err) {
      this.error = `metadata upload failed: ${err.message}`;
      return;
    }
    this.message = 'uploading';
    await Promise.all(Array.from({ length: this.parallel }, () => this.worker()));
    this.message = 'done';
  }

  stop() { this.stopped = true; }

  speed() {
    const now = performance.now();
    this.sent = this.sent.filter(([t]) => now - t < 10000);
    if (this.sent.length < 2) return 0;
    const bytes = this.sent.reduce((a, [, b]) => a + b, 0);
    return bytes / Math.max(1, (now - this.sent[0][0]) / 1000);
  }

  async refresh() {
    if (this.refreshing) return this.refreshing;
    const wait = 1000 - (performance.now() - this.lastRefresh);
    this.refreshing = (async () => {
      if (wait > 0) await sleep(wait);
      try {
        const w = await api(`/api/jobs/${this.job.id}/wanted`);
        if (w.stop) { this.stopped = true; return; }
        this.chunkSize = w.chunk_size;
        this.window = w.tiles;
        const q = [];
        for (const t of w.tiles) {
          for (const f of t.files) {
            const got = new Set(f.received);
            for (let ci = 0; ci < f.n_chunks; ci++) {
              const id = `${t.key}:${f.fi}:${ci}`;
              if (!got.has(ci) && !this.inflight.has(id)) q.push({ id, key: t.key, fi: f.fi, ci, size: f.size });
            }
          }
        }
        this.queue = q;
        this.error = null;
      } catch (err) {
        this.error = `cannot reach server: ${err.message}`;
      } finally {
        this.lastRefresh = performance.now();
        this.refreshing = null;
      }
    })();
    return this.refreshing;
  }

  async worker() {
    while (!this.stopped) {
      const chunk = this.queue.shift();
      if (!chunk) {
        await this.refresh();
        if (!this.queue.length) await sleep(1500);
        continue;
      }
      if (this.inflight.has(chunk.id)) continue;
      this.inflight.add(chunk.id);
      try {
        await this.send(chunk);
      } finally {
        this.inflight.delete(chunk.id);
      }
    }
  }

  async send(chunk) {
    const file = this.files[chunk.key][chunk.fi];
    const start = chunk.ci * this.chunkSize;
    const blob = file.slice(start, Math.min(start + this.chunkSize, chunk.size));
    for (let attempt = 0; attempt < 6 && !this.stopped; attempt++) {
      try {
        await api(`/api/jobs/${this.job.id}/tiles/${chunk.key}/files/${chunk.fi}/chunks/${chunk.ci}`, { method: 'PUT', body: blob });
        this.sent.push([performance.now(), blob.size]);
        this.totalSent += blob.size;
        return;
      } catch (err) {
        if (err.status === 409 || err.status === 400) return;  // window moved on / job ended
        this.error = `retrying after: ${err.message}`;
        await sleep(1000 * 2 ** attempt);
      }
    }
  }
}

function startUploader(job, files, sidecars) {
  S.uploaders[job.id]?.stop();
  const up = new Uploader(job, files, sidecars, Number($('#parallel-uploads').value) || 4);
  S.uploaders[job.id] = up;
  up.start();
}

window.addEventListener('beforeunload', (e) => {
  if (Object.values(S.uploaders).some((u) => !u.stopped)) {
    e.preventDefault();
    e.returnValue = '';
  }
});

// ── remote file browser ──────────────────────────────────────────────────────
// `endpoint` set: browse your computer through Globus; otherwise this machine.
// `pick`: 'dir', 'file', or 'any' (a folder or a file).
const B = { pick: 'dir', endpoint: null, path: '', selected: null, onChoose: null };

function openBrowser({ wantFile, pick, endpoint = null, start, onChoose }) {
  B.pick = pick || (wantFile ? 'file' : 'dir');
  B.endpoint = endpoint;
  B.onChoose = onChoose;
  B.selected = null;
  const where = endpoint ? 'on your computer (via Globus)' : 'on the remote machine';
  $('#browser-title').textContent = `Choose ${{ dir: 'a folder', file: 'a file', any: 'a folder or file' }[B.pick]} ${where}`;
  $('#browser-choose').textContent = { dir: 'Use this folder', file: 'Use selected file', any: 'Use selected (or this folder)' }[B.pick];
  const key = endpoint ? `lastBrowse:${endpoint}` : 'lastBrowse';
  loadBrowser(start || store.get(key) || (endpoint ? '/~/' : S.schema.home));
  $('#browser').showModal();
}

async function loadBrowser(path) {
  $('#browser-error').textContent = '';
  if (B.endpoint) $('#browser-list').innerHTML = '<li class="file">Listing through Globus…</li>';
  // Globus listings take a second or two; only the latest request may draw.
  const seq = (B.seq = (B.seq || 0) + 1);
  try {
    const r = B.endpoint
      ? await api(`/api/globus/ls?endpoint=${encodeURIComponent(B.endpoint)}&path=${encodeURIComponent(path)}`)
      : await api(`/api/fs?path=${encodeURIComponent(path)}`);
    if (seq !== B.seq) return;
    B.path = r.path;
    B.parent = r.parent;
    B.selected = null;
    store.set(B.endpoint ? `lastBrowse:${B.endpoint}` : 'lastBrowse', r.path);
    // Leave the box alone while someone is typing a path into it.
    if (document.activeElement !== $('#browser-path')) $('#browser-path').value = r.path;
    const list = $('#browser-list');
    list.innerHTML = '';
    for (const e of r.entries) {
      const li = document.createElement('li');
      li.className = e.dir ? 'dir' : 'file';
      li.innerHTML = `<span>${e.dir ? '📁 ' : ''}${esc(e.name)}</span><span>${e.dir ? '' : fmtBytes(e.size)}</span>`;
      const full = `${r.path.replace(/\/$/, '')}/${e.name}`;
      li.addEventListener('click', () => {
        list.querySelectorAll('.selected').forEach((x) => x.classList.remove('selected'));
        li.classList.add('selected');
        B.selected = full;
        B.selectedIsDir = e.dir;
      });
      li.addEventListener('dblclick', () => {
        if (e.dir) loadBrowser(full);
        else if (B.pick !== 'dir') choose(full);
      });
      list.appendChild(li);
    }
    if (!r.entries.length) list.innerHTML = '<li class="file">(empty)</li>';
  } catch (err) {
    if (seq === B.seq) $('#browser-error').textContent = err.message;
  }
}

function choose(path) {
  $('#browser').close();
  B.onChoose?.(path);
}

// ── jobs ─────────────────────────────────────────────────────────────────────
async function refreshJobs() {
  try {
    S.jobs = await api('/api/jobs');
  } catch (err) {
    $('#jobs').innerHTML = `<div class="card error">${esc(err.message)}</div>`;
    return;
  }
  renderJobs();
}

function jobProgress(job) {
  if (!job.manifest || job.manifest.kind !== 'folder') return '';
  const n = job.manifest.n_tiles;
  const done = Object.keys(job.detected || {}).length;
  const pct = n ? (100 * job.released) / n : 0;
  const moved = job.globus ? `${job.transferred}/${n} tiles transferred · ` : '';
  return `<div class="bar"><div style="width:${pct.toFixed(1)}%"></div></div>
    <div class="upload-line">${moved}${job.released}/${n} tiles read · ${done} with spots detected</div>`;
}

function globusLine(job) {
  const active = ['starting', 'running'].includes(job.state);
  if (!active) return '';
  const lines = [];
  if (job.globus_note) lines.push(`<div class="upload-line warn">⚠ ${esc(job.globus_note)}</div>`);
  const entries = Object.entries(job.transfers || {}).filter(([, t]) => t.state === 'ACTIVE');
  let rate = 0;
  for (const [key, t] of entries) {
    rate += t.rate || 0;
    const what = key === 'sidecars' ? 'metadata' : `R0_${key}`;
    const pct = t.total ? ` ${Math.floor((100 * t.bytes) / t.total)}% (${fmtBytes(t.bytes)} of ${fmtBytes(t.total)})` : '';
    const ns = t.nice_status && t.nice_status !== 'OK' ? ` · <span class="warn">${esc(t.nice_status)}</span>` : '';
    const retry = t.attempts > 1 ? ` · attempt ${t.attempts}` : '';
    lines.push(`<div class="upload-line">⇣ ${what}:${pct}${ns}${retry}</div>`);
  }
  if (entries.length) lines.unshift(`<div class="upload-line">Globus: ${fmtBytes(rate)}/s average</div>`);
  else if (job.phase === 'detecting' || job.phase === 'waiting for first tile') {
    lines.push(`<div class="upload-line muted">Globus: ${job.transferred === job.manifest?.n_tiles ? 'all tiles transferred' : 'queue full, waiting for the pipeline'}</div>`);
  }
  return lines.join('');
}

function uploadLine(job) {
  if (job.globus) return globusLine(job);
  if (!job.manifest) return '';
  const up = S.uploaders[job.id];
  const active = ['starting', 'running'].includes(job.state);
  if (!up) {
    if (!active) return '';
    return `<div class="upload-line"><span class="error">Not uploading from this tab.</span>
      <button type="button" class="small" data-resume="${job.id}">Re-select ${job.manifest.kind === 'single' ? 'file' : 'folder'} to resume</button></div>`;
  }
  if (up.error) return `<div class="upload-line error">${esc(up.error)}</div>`;
  if (up.stopped || !active) return `<div class="upload-line muted">Upload finished (${fmtBytes(up.totalSent)} sent).</div>`;
  const sending = up.window.map((t) => {
    const partial = t.files.reduce((a, f) => a + f.received.length / f.n_chunks, 0);
    return `R0_${t.key} ${Math.floor((100 * (t.files_done + partial)) / t.n_files)}%`;
  }).join(', ');
  const waiting = (job.waiting_for || []).length ? ` · <b>pipeline waiting for tile ${job.waiting_for.join(', ')}</b>` : '';
  return `<div class="upload-line">⇧ ${fmtBytes(up.speed())}/s · ${fmtBytes(up.totalSent)} sent ·
    ${sending ? `sending ${esc(sending)}` : 'queue full, waiting for the pipeline'}${waiting}</div>`;
}

function renderJobs() {
  const el = $('#jobs');
  if (!S.jobs.length) {
    el.innerHTML = '<div class="card muted">No runs yet.</div>';
    return;
  }
  el.innerHTML = S.jobs.map((job) => {
    const active = ['starting', 'running'].includes(job.state);
    const elapsed = job.started ? fmtDuration((active ? Date.now() / 1000 : job.updated) - job.started) : '';
    const m = job.manifest;
    const source = job.mode === 'remote' ? job.squid_folder
      : job.globus ? `via Globus from ${job.globus.source_endpoint_name}: ${job.globus.source_root}${m?.kind === 'single' ? `/${m.root_name}` : ''} (${m?.n_tiles} tile${m?.n_tiles === 1 ? '' : 's'}${m?.layout === 'legacy' ? `, old layout, ${m.n_files} files` : ''}, ${fmtBytes(m?.total_bytes)})`
      : `uploaded from this computer: ${job.manifest?.root_name || ''} (${job.manifest?.n_tiles} tile${job.manifest?.n_tiles === 1 ? '' : 's'}${job.manifest?.layout === 'legacy' ? `, old layout, ${job.manifest.n_files} files` : ''}, ${fmtBytes(job.manifest?.total_bytes)})`;
    return `<div class="card job ${S.selectedJob === job.id ? 'selected' : ''}" data-job="${job.id}">
      <div class="job-head">
        <span class="job-name">${esc(job.name || job.id)}</span>
        <span class="badge state-${job.state}">${esc(job.state)}${active && job.phase ? ` · ${esc(job.phase)}` : ''}</span>
      </div>
      <div class="job-meta">${esc(source)}<br>→ ${esc(job.output_folder)}<br>
        ${new Date(job.created * 1000).toLocaleString()} · GPU ${job.gpu === '' || job.gpu == null ? 'default' : esc(job.gpu)}${elapsed ? ` · ${elapsed}` : ''}</div>
      ${job.error ? `<div class="error upload-line">${esc(job.error)}</div>` : ''}
      ${jobProgress(job)}
      ${uploadLine(job)}
      <div class="actions">
        ${active ? `<button type="button" class="small danger" data-cancel="${job.id}">Cancel</button>` : ''}
        ${job.state === 'finished' ? `<a href="/api/jobs/${job.id}/results.zip"><button type="button" class="small">Download results (.zip)</button></a>` : ''}
        <button type="button" class="small" data-clone="${job.id}">Copy settings to form</button>
        ${!active ? `<button type="button" class="small danger" data-delete="${job.id}">Delete</button>` : ''}
      </div>
    </div>`;
  }).join('');
}

$('#jobs').addEventListener('click', async (e) => {
  const t = e.target.closest('button, .job');
  if (!t) return;
  const ds = t.dataset;
  try {
    if (ds.cancel) {
      if (!confirm('Cancel this run?')) return;
      S.uploaders[ds.cancel]?.stop();
      await postJSON(`/api/jobs/${ds.cancel}/cancel`, {});
    } else if (ds.delete) {
      const job = S.jobs.find((j) => j.id === ds.delete);
      const inside = job.output_folder.includes(`/${job.id}/`) || job.output_folder.endsWith(`/${job.id}/output`);
      if (!confirm(`Delete this run's log and job folder?${inside ? '\n\nIts results are inside the job folder and will be deleted too.' : `\n\nResults in ${job.output_folder} are kept.`}`)) return;
      await api(`/api/jobs/${ds.delete}`, { method: 'DELETE' });
      if (S.selectedJob === ds.delete) closeDetail();
    } else if (ds.clone) {
      cloneJob(S.jobs.find((j) => j.id === ds.clone));
      return;
    } else if (ds.resume) {
      resumeJob(S.jobs.find((j) => j.id === ds.resume));
      return;
    } else if (ds.job && !e.target.closest('a')) {
      selectJob(ds.job);
      return;
    } else {
      return;
    }
  } catch (err) {
    alert(err.message);
  }
  refreshJobs();
});

function cloneJob(job) {
  renderForm(job.params);
  setInputMode(job.mode === 'remote' ? 'remote' : job.mode.startsWith('globus') ? 'globus' : 'upload');
  if (job.mode === 'remote') $('#squid-folder').value = job.squid_folder;
  if (job.globus) {
    S.globusEndpoint = job.globus.source_endpoint;
    $('#globus-endpoint').value = job.globus.source_endpoint;
    $('#globus-path').value = job.manifest?.kind === 'single'
      ? `${job.globus.source_root}/${job.manifest.root_name}` : job.globus.source_root;
    $('#globus-verify').checked = job.globus.verify_checksum !== false;
    inspectGlobusPath();
  }
  if (job.prefetch_tiles != null) $('#prefetch-tiles').value = job.prefetch_tiles;
  if (job.gpu != null) $('#gpu-select').value = job.gpu;
  saveForm();
  $('#new-run').scrollIntoView({ behavior: 'smooth' });
}

async function resumeJob(job) {
  const full = await api(`/api/jobs/${job.id}?full=1`);
  const input = document.createElement('input');
  input.type = 'file';
  if (full.manifest.kind === 'folder') input.webkitdirectory = true;
  input.addEventListener('change', () => {
    try {
      let sel;
      if (full.manifest.kind === 'folder') {
        sel = scanFolder(input.files);
      } else {
        sel = singleSelection(input.files[0]);
      }
      const { files, sidecars } = matchSelection(sel, full.manifest);
      startUploader(full, files, sidecars);
      renderJobs();
    } catch (err) {
      alert(err.message);
    }
  });
  input.click();
}

// ── job detail + log ─────────────────────────────────────────────────────────
function selectJob(id) {
  S.selectedJob = id;
  S.log = { jobId: id, offset: 0, raw: '' };
  $('#log').textContent = '';
  $('#job-detail').classList.remove('hidden');
  renderJobs();
  refreshDetail();
}

function closeDetail() {
  S.selectedJob = null;
  $('#job-detail').classList.add('hidden');
  renderJobs();
}

function renderLog(raw) {
  // tqdm redraws its bar with \r; show only the last redraw of each line.
  return raw.split('\n').map((l) => {
    const t = l.replace(/\r+$/, '');
    return t.slice(t.lastIndexOf('\r') + 1);
  }).join('\n');
}

async function refreshDetail() {
  const id = S.selectedJob;
  if (!id) return;
  const job = S.jobs.find((j) => j.id === id);
  if (job) {
    $('#detail-title').textContent = job.name || job.id;
    const p = job.params;
    $('#detail-info').innerHTML = `<div class="job-meta">Job ${esc(job.id)} · log tail below · settings:
      ${esc(Object.entries(p).filter(([, v]) => v !== '' && v !== false).map(([k, v]) => `${k}=${v}`).join(', '))}</div>`;
  }
  try {
    const r = await api(`/api/jobs/${id}/log?offset=${S.log.offset}`);
    if (S.log.jobId !== id) return;
    if (r.text) {
      const pre = $('#log');
      const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
      S.log.raw = (S.log.raw + r.text).slice(-1_500_000);
      pre.textContent = renderLog(S.log.raw);
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    }
    S.log.offset = r.offset;
  } catch (_) { /* job deleted */ }
}

// ── machine info ─────────────────────────────────────────────────────────────
async function refreshMachine(first) {
  let info;
  try { info = await api('/api/gpus'); } catch (_) { return; }
  const strip = info.gpus.map((g) =>
    `<span>GPU ${g.index}: ${(g.mem_used_mb / 1024).toFixed(0)}/${(g.mem_total_mb / 1024).toFixed(0)} GB · ${g.util}%${g.jobs.length ? ` · ${esc(g.jobs.join(', '))}` : ''}</span>`).join('');
  $('#machine').innerHTML = `<div class="gpu-strip">${strip}<span>Disk free: ${fmtBytes(info.disk_free)}</span></div>`;
  const sel = $('#gpu-select');
  const prev = sel.value;
  sel.innerHTML = info.gpus.map((g) =>
    `<option value="${g.index}">GPU ${g.index}: ${esc(g.name)} — ${(g.mem_used_mb / 1024).toFixed(1)}/${(g.mem_total_mb / 1024).toFixed(0)} GB used${g.jobs.length ? ` (running: ${esc(g.jobs.join(', '))})` : ''}</option>`).join('')
    + '<option value="">All GPUs visible (pipeline uses the first)</option>';
  if (first) {
    // Default to the GPU with the most free memory and no jobs from this app.
    const best = [...info.gpus].sort((a, b) =>
      (a.jobs.length - b.jobs.length) || ((b.mem_total_mb - b.mem_used_mb) - (a.mem_total_mb - a.mem_used_mb)))[0];
    sel.value = best ? String(best.index) : '';
  } else {
    sel.value = prev;
  }
}

function singleSelection(f) {
  return {
    kind: 'single', layout: 'single', root_name: f.name, sidecars: [],
    tiles: [{ key: 0, size: f.size, files: [{ relpath: f.name, size: f.size, file: f }] }],
  };
}

// ── start a run ──────────────────────────────────────────────────────────────
async function startRun() {
  const errEl = $('#start-error');
  errEl.textContent = '';
  const body = {
    params: formValues(), name: $('#job-name').value, gpu: $('#gpu-select').value,
    prefetch_tiles: Number($('#prefetch-tiles').value),
  };
  let local = null;
  if (S.inputMode === 'remote') {
    body.mode = 'remote';
    body.squid_folder = $('#squid-folder').value;
  } else if (S.inputMode === 'globus') {
    body.mode = 'globus';
    body.globus = { endpoint: $('#globus-endpoint').value, path: $('#globus-path').value.trim(),
                    verify_checksum: $('#globus-verify').checked };
    if (!body.globus.path) { errEl.textContent = 'Choose a folder or file on your computer first.'; return; }
  } else {
    const sel = S.selection;
    if (!sel || !sel.tiles.length) { errEl.textContent = 'Choose a folder or file to upload first.'; return; }
    body.mode = sel.kind === 'single' ? 'upload_single' : 'upload_folder';
    body.manifest = {
      layout: sel.layout, root_name: sel.root_name,
      tiles: sel.tiles.map((t) => ({ key: t.key, files: t.files.map((f) => ({ relpath: f.relpath, size: f.size })) })),
      sidecars: sel.sidecars.map((s) => s.relpath),
    };
    local = sel;
  }
  $('#start').disabled = true;
  try {
    const job = await postJSON('/api/jobs', body);
    if (local) {
      const full = await api(`/api/jobs/${job.id}?full=1`);
      startUploader(full, ...Object.values(matchSelection(local, full.manifest)));
    }
    saveForm();
    await refreshJobs();
    selectJob(job.id);
  } catch (err) {
    errEl.textContent = err.message;
  } finally {
    $('#start').disabled = false;
  }
}

// ── wiring ───────────────────────────────────────────────────────────────────
async function init() {
  S.schema = await api('/api/schema');
  const saved = store.get('form');
  renderForm(saved?.params);
  if (saved) {
    $('#squid-folder').value = saved.squid || '';
    if (saved.prefetch) $('#prefetch-tiles').value = saved.prefetch;
    if (saved.parallel) $('#parallel-uploads').value = saved.parallel;
    $('#globus-path').value = saved.globusPath || '';
    S.globusEndpoint = saved.globusEndpoint || null;
    if (saved.globusVerify === false) $('#globus-verify').checked = false;
  }
  setInputMode(saved?.mode || 'remote');
  if (S.inputMode === 'globus') refreshGlobus(false).then(inspectGlobusPath);

  document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => { setInputMode(t.dataset.mode); saveForm(); }));
  $('#params-form').addEventListener('change', () => {
    saveForm();
    showSelection();
    if (S.inputMode === 'globus' && S.globusInfo) inspectGlobusPath();
  });
  $('#squid-folder').addEventListener('change', saveForm);
  $('#globus-path').addEventListener('change', () => { saveForm(); inspectGlobusPath(); });
  $('#globus-endpoint').addEventListener('change', () => { S.globusEndpoint = $('#globus-endpoint').value; saveForm(); inspectGlobusPath(); });
  $('#globus-verify').addEventListener('change', saveForm);
  $('#browse-globus').addEventListener('click', () => {
    const endpoint = $('#globus-endpoint').value;
    if (!endpoint) { alert('No Globus endpoint for your computer is available yet; see the checks above.'); return; }
    openBrowser({
      pick: 'any', endpoint, start: $('#globus-path').value,
      onChoose: (p) => { $('#globus-path').value = p; saveForm(); inspectGlobusPath(); },
    });
  });
  $('#browse-squid').addEventListener('click', () => openBrowser({
    wantFile: false, start: $('#squid-folder').value,
    onChoose: (p) => { $('#squid-folder').value = p; saveForm(); },
  }));
  $('#pick-folder').addEventListener('click', () => $('#folder-input').click());
  $('#pick-file').addEventListener('click', () => $('#file-input').click());
  $('#folder-input').addEventListener('change', (e) => {
    S.selection = scanFolder(e.target.files);
    showSelection();
    e.target.value = '';
  });
  $('#file-input').addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (!f) return;
    S.selection = singleSelection(f);
    showSelection();
    e.target.value = '';
  });
  $('#start').addEventListener('click', startRun);
  $('#reset-defaults').addEventListener('click', () => { renderForm(null); saveForm(); });
  $('#detail-close').addEventListener('click', closeDetail);

  $('#browser-up').addEventListener('click', () => loadBrowser(B.parent || '/'));
  $('#browser-go').addEventListener('click', () => loadBrowser($('#browser-path').value));
  $('#browser-path').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); loadBrowser(e.target.value); } });
  $('#browser-choose').addEventListener('click', () => {
    if (B.pick === 'file') {
      if (!B.selected || B.selectedIsDir) { $('#browser-error').textContent = 'Select a file first.'; return; }
      choose(B.selected);
    } else if (B.pick === 'any') {
      choose(B.selected || B.path);
    } else {
      choose(B.selected && B.selectedIsDir ? B.selected : B.path);
    }
  });

  await Promise.all([refreshMachine(true), refreshJobs()]);
  setInterval(refreshJobs, 2000);
  setInterval(refreshDetail, 2000);
  setInterval(() => refreshMachine(false), 10000);
  setInterval(() => { if (S.inputMode === 'globus' && !document.hidden) refreshGlobus(false); }, 20000);
}

init().catch((err) => { document.body.innerHTML = `<p class="error" style="padding:20px">${esc(err.message)}</p>`; });
