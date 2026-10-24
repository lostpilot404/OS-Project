/* Adaptive CPU Scheduling — simulation console.
 *
 * The browser holds no scheduling, detection or metric logic. It sends requests to the
 * Python API, renders the returned values, and injects the SVG charts that the backend
 * draws from the same trace. Navigation is hash-based; every page renders from one shared
 * run state, so pages can never show data from different simulations.
 */
(() => {
  'use strict';

  // ---------------------------------------------------------------- state

  const state = {
    scenarios: [],
    scenarioId: null,
    options: {},
    running: false,
    sequence: 0,
    run: null,                 // { run_id, result, source: 'live' | 'restored' }
    comparison: null,          // { result, runId }
    comparisonBusy: false,
    research: null,
    researchError: null,
    health: null,
    online: false,
    recentRuns: [],
    sessionRuns: null,         // registry size, from /api/runs; the only run counter shown
    teachingPolicy: null,
    sort: { key: 'pid', direction: 1 },
  };

  let lastRenderedRoute = null;

  const ROUTES = ['overview', 'studio', 'workload', 'recovery', 'compare', 'research', 'about'];
  const DEFAULT_ROUTE = 'overview';

  const POLICY_CLASS = { FCFS: 'fcfs', SJF: 'sjf', 'Round Robin': 'rr', Priority: 'prio' };

  const $ = (sel, root = document) => root.querySelector(sel);
  const view = () => document.getElementById('view');
  const el = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- helpers

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function fmt(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
    const n = Number(value);
    return Number.isInteger(n) ? String(n) : n.toFixed(digits);
  }

  function listText(value) {
    return esc(String(value ?? '').replace(/\s*,\s*/g, ', '));
  }

  function signed(value, digits = 2) {
    if (value === null || value === undefined) return 'n/a';
    const n = Number(value);
    return (n > 0 ? '+' : '') + fmt(n, digits);
  }

  function pct(value) {
    return value === null || value === undefined ? 'n/a' : `${Number(value).toFixed(2)}%`;
  }

  function timeAgo(iso) {
    if (!iso) return '';
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return '';
    const s = Math.max(0, Math.round((Date.now() - then) / 1000));
    if (s < 5) return 'just now';
    if (s < 60) return `${s}s ago`;
    const m = Math.round(s / 60);
    if (m < 60) return `${m} min ago`;
    return `${Math.round(m / 60)} h ago`;
  }

  class ApiError extends Error {
    constructor(message, status, field, kind) {
      super(message);
      this.status = status;
      this.field = field;
      this.kind = kind;
    }
  }

  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
    } catch (error) {
      throw new ApiError('The backend could not be reached. Is `tools/serve_dashboard.py` running?', 0, null, 'offline');
    }
    const text = await response.text();
    let body = null;
    if (text) { try { body = JSON.parse(text); } catch { body = null; } }
    if (!response.ok) {
      const detail = body && body.error ? body.error : null;
      const message = detail && detail.message ? detail.message
        : (body && body.detail ? String(body.detail) : `Request failed with HTTP ${response.status}.`);
      throw new ApiError(message, response.status, detail && detail.field, detail && detail.type);
    }
    return body;
  }

  // ---------------------------------------------------------------- router

  function currentRoute() {
    const hash = (window.location.hash || '').replace(/^#\/?/, '').split('?')[0];
    return ROUTES.includes(hash) ? hash : DEFAULT_ROUTE;
  }

  function navigate(route, options = {}) {
    const target = ROUTES.includes(route) ? route : DEFAULT_ROUTE;
    if (currentRoute() === target && !options.force) {
      render();
      return;
    }
    window.location.hash = `#/${target}`;
  }

  function render() {
    // In-page fragments such as the skip link's #view are not routes: leave the
    // current page alone so a keyboard jump never navigates away from it.
    if (window.location.hash && !window.location.hash.startsWith('#/')) return;
    const route = currentRoute();
    document.querySelectorAll('.rail-nav a').forEach((link) => {
      if (link.dataset.route === route) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    const page = PAGES[route] || PAGES[DEFAULT_ROUTE];
    view().innerHTML = page();
    document.title = `${page.title} — Adaptive CPU Scheduling`;
    if (lastRenderedRoute !== null && lastRenderedRoute !== route) window.scrollTo(0, 0);
    lastRenderedRoute = route;
    if (route === 'research' && !state.research && !state.researchError) loadResearch();
  }

  // ---------------------------------------------------------------- boot

  async function boot() {
    bindEvents();
    await refreshHealth();
    await loadExamples();
    await loadRecentRuns();
    if (!state.scenarioId && state.scenarios.length) selectScenario(state.scenarios[0].id, { quiet: true });
    render();
    setInterval(() => { if (!state.running) refreshHealth({ quiet: true }); }, 25000);
  }

  function bindEvents() {
    window.addEventListener('hashchange', render);
    const skip = document.querySelector('.skip-link');
    if (skip) {
      skip.addEventListener('click', (event) => {
        event.preventDefault();
        view().focus();
      });
    }
    el('health-retry').addEventListener('click', async () => {
      const ok = await refreshHealth();
      if (ok) {
        clearBanner();
        await loadExamples();
        await loadRecentRuns();
        render();
      }
    });

    // The banner lives outside #view, so the delegated handler below never sees its Dismiss
    // button. It needs its own listener or the control does nothing.
    el('banner').addEventListener('click', (event) => {
      if (event.target.closest('[data-action="dismiss-banner"]')) clearBanner();
    });

    view().addEventListener('click', (event) => {
      const scenario = event.target.closest('[data-action="select-scenario"]');
      if (scenario) { selectScenario(scenario.dataset.id); return; }

      const action = event.target.closest('[data-action]');
      if (action) {
        const name = action.dataset.action;
        if (name === 'run') runSimulation();
        else if (name === 'compare') runComparison();
        else if (name === 'teaching-policy') applyTeachingPolicy(action.dataset.policy);
        else if (name === 'sort') sortWorkload(action.dataset.key);
        else if (name === 'toggle-chart-fit') toggleChartFit(action);
        else if (name === 'add-process') addProcess();
        else if (name === 'remove-process') removeProcess(Number(action.dataset.index));
        else if (name === 'reset-processes') resetProcesses();
        else if (name === 'dismiss-banner') clearBanner();
        else if (name === 'open-run') openRun(action.dataset.runId);
        else if (name === 'retry-research') { state.researchError = null; loadResearch(); }
        return;
      }

      const slice = event.target.closest('g.slice');
      if (slice) inspectSlice(slice);
    });

    view().addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const slice = event.target.closest('g.slice');
      if (!slice) return;
      event.preventDefault();
      inspectSlice(slice);
    });

    view().addEventListener('change', (event) => {
      const field = event.target;
      if (field.dataset.option) {
        const key = field.dataset.option;
        let value = field.type === 'checkbox' ? field.checked : field.value;
        if (field.type === 'number') value = Number(value);
        state.options[key] = value;
        if (key === 'policy_view' && state.run && state.run.result.kind === 'teaching') {
          applyTeachingPolicy(String(value));
          return;
        }
        if (key === 'healing') syncHealingPolicyField();
        validateOptions();
        updateRunSummary();
      }
    });

    view().addEventListener('input', (event) => {
      const input = event.target.closest('input[data-key]');
      if (!input) return;
      const row = input.closest('tr[data-row]');
      const entry = state.options.processes[Number(row.dataset.row)];
      entry[input.dataset.key] = input.value === '' ? NaN : Number(input.value);
      validateOptions();
      updateRunSummary();
    });
  }

  async function refreshHealth({ quiet = false } = {}) {
    const pill = el('engine-status');
    const label = el('engine-status-text');
    try {
      const health = await api('/api/health', { cache: 'no-store' });
      state.health = health;
      state.online = health.status === 'ok';
      pill.className = `rail-status ${state.online ? 'ok' : 'bad'}`;
      label.textContent = state.online ? `Engine ready · ${health.engine.python}` : 'Engine degraded';
      el('health-retry').hidden = state.online;
      renderCounters();
      return state.online;
    } catch (error) {
      state.online = false;
      state.health = null;
      pill.className = 'rail-status bad';
      label.textContent = 'Backend unavailable';
      el('health-retry').hidden = false;
      renderCounters();
      if (!quiet) console.warn('health check failed', error);
      return false;
    }
  }

  /**
   * Write the session run count everywhere it appears.
   *
   * There is exactly one counter: ``state.sessionRuns``, read from ``GET /api/runs``, whose
   * ``count`` is the registry's own size (the same figure ``/api/health`` calls
   * ``stored_runs``). The footer used to read its count from the health payload while the
   * overview read a different one from the runs list, so after a run the two disagreed until
   * the next health poll.
   */
  function renderCounters() {
    const node = el('footer-engine');
    if (!node) return;
    const health = state.health;
    if (!health) {
      node.textContent = '';
      return;
    }
    const parts = [`${health.service} ${health.version}`, `probe ${health.engine.probe_ms} ms`];
    if (state.sessionRuns !== null) parts.push(runCountText('kept in memory'));
    node.textContent = parts.join(' · ');
  }

  function runCountText(suffix) {
    if (state.sessionRuns === null) return 'run history unavailable';
    const count = `${state.sessionRuns} run${state.sessionRuns === 1 ? '' : 's'}`;
    return suffix ? `${count} ${suffix}` : count;
  }

  async function loadExamples() {
    try {
      const payload = await api('/api/examples');
      state.scenarios = payload.scenarios || [];
    } catch (error) {
      state.scenarios = [];
      showBanner('error', `<strong>Scenarios could not be loaded.</strong> ${esc(error.message)}`);
    }
  }

  async function loadRecentRuns() {
    try {
      const payload = await api('/api/runs');
      state.recentRuns = payload.runs || [];
      // ``count`` is the registry size; ``returned`` only differs if the page is truncated.
      state.sessionRuns = typeof payload.count === 'number' ? payload.count : state.recentRuns.length;
    } catch (error) {
      state.recentRuns = [];
      state.sessionRuns = null;
    }
    renderCounters();
  }

  async function loadResearch() {
    try {
      state.research = await api('/api/research');
      state.researchError = null;
    } catch (error) {
      state.researchError = error;
    }
    if (currentRoute() === 'research') render();
  }

  // ---------------------------------------------------------------- scenario & config

  function currentScenario() {
    return state.scenarios.find((s) => s.id === state.scenarioId) || null;
  }

  function scenarioCategory(scenario) {
    if (scenario.id === 'teaching') return 'Teaching example';
    if (scenario.id === 'custom') return 'Custom workload';
    return 'Self-healing recovery';
  }

  function selectScenario(id, { quiet = false } = {}) {
    if (state.running) return;
    const scenario = state.scenarios.find((s) => s.id === id);
    if (!scenario) return;
    state.scenarioId = id;
    state.options = JSON.parse(JSON.stringify(scenario.defaults || {}));
    state.teachingPolicy = null;
    if (!quiet) {
      clearBanner();
      render();
    }
  }

  /** Enable or disable the custom policy selector to match the self-healing checkbox. */
  function syncHealingPolicyField() {
    const select = el('policy-select');
    const note = el('policy-healing-note');
    const healing = Boolean(state.options.healing);
    if (select) select.disabled = healing;
    if (note) note.hidden = !healing;
  }

  function validateOptions() {
    const error = el('config-error');
    if (!error) return null;
    const scenario = currentScenario();
    if (!scenario) return null;
    if (scenario.kind !== 'custom') {
      error.textContent = '';
      return null;
    }
    const processes = state.options.processes || [];
    const message = (() => {
      if (!processes.length) return 'Add at least one process.';
      if (processes.length > 40) return 'At most 40 processes per run.';
      const seen = new Set();
      for (let i = 0; i < processes.length; i += 1) {
        const row = processes[i];
        const label = `Process ${i + 1}`;
        for (const key of ['pid', 'arrival_time', 'burst_time', 'priority']) {
          if (!Number.isInteger(row[key])) return `${label}: ${key.replace('_', ' ')} must be a whole number.`;
        }
        if (row.pid < 0) return `${label}: id must be 0 or more.`;
        if (row.arrival_time < 0) return `${label}: arrival must be 0 or more.`;
        if (row.burst_time < 1) return `${label}: burst must be at least 1 time unit.`;
        if (row.priority < 0) return `${label}: priority must be 0 or more.`;
        if (seen.has(row.pid)) return `Duplicate process id ${row.pid}.`;
        seen.add(row.pid);
      }
      if (!(state.options.quantum >= 1 && state.options.quantum <= 64)) {
        return 'Round-robin quantum must be between 1 and 64.';
      }
      if (!(state.options.switching_cost >= 0 && state.options.switching_cost <= 10)) {
        return 'Switch cost must be between 0 and 10 time units.';
      }
      return null;
    })();
    error.textContent = message || '';
    return message;
  }

  function updateRunSummary() {
    const target = el('run-summary');
    if (!target) return;
    const scenario = currentScenario();
    target.innerHTML = runSummaryRows(scenario);
  }

  function addProcess() {
    const processes = state.options.processes || [];
    if (processes.length >= 40) return;
    const next = processes.reduce((m, p) => Math.max(m, Number(p.pid) || 0), 0) + 1;
    processes.push({ pid: next, arrival_time: 0, burst_time: 4, priority: 2 });
    render();
  }

  function removeProcess(index) {
    if ((state.options.processes || []).length <= 1) return;
    state.options.processes.splice(index, 1);
    render();
  }

  function resetProcesses() {
    const scenario = currentScenario();
    state.options.processes = JSON.parse(JSON.stringify(scenario.defaults.processes));
    render();
  }

  function sortWorkload(key) {
    state.sort = { key, direction: state.sort.key === key ? -state.sort.direction : 1 };
    render();
  }

  // ---------------------------------------------------------------- run

  async function runSimulation() {
    if (state.running || !state.scenarioId) return;
    const scenario = currentScenario();
    if (!scenario) return;
    if (!state.online) {
      showBanner('error', '<strong>The backend is not reachable.</strong> Start it with <code>python tools/serve_dashboard.py</code>, then retry.');
      return;
    }
    if (scenario.kind === 'custom') {
      const problem = validateOptions();
      if (problem) {
        showBanner('error', `<strong>Check the workload.</strong> ${esc(problem)}`);
        return;
      }
    }

    const sequence = ++state.sequence;
    state.running = true;
    clearBanner();
    setRunCardBusy(true, scenario);
    const started = performance.now();
    const timer = setInterval(() => {
      const node = el('run-status');
      if (node) node.textContent = `Executing… ${((performance.now() - started) / 1000).toFixed(1)} s`;
    }, 100);

    try {
      const options = { ...state.options };
      if (scenario.kind === 'custom' && options.healing) {
        // The loop starts from FCFS; sending a different policy would be a request the
        // engine cannot honour (the API rejects it).
        delete options.policy;
      }
      const payload = await api('/api/simulations', {
        method: 'POST',
        body: JSON.stringify({ scenario: state.scenarioId, options }),
      });
      if (sequence !== state.sequence) return;         // superseded by a newer request
      state.run = { run_id: payload.run_id, result: payload.result, source: 'live' };
      state.comparison = null;
      state.teachingPolicy = payload.result.kind === 'teaching' ? payload.result.teaching.selected : null;
      await loadRecentRuns();
      render();
      const done = el('run-status');
      if (done) {
        const ms = Math.round(performance.now() - started);
        done.textContent = `Completed in ${ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`} · run ${payload.run_id.slice(0, 8)}`;
        done.scrollIntoView({ block: 'nearest' });
      }
    } catch (error) {
      if (sequence !== state.sequence) return;
      if (error.status === 0) markOffline();
      showBanner('error',
        `<strong>The simulation did not run.</strong> ${esc(error.message)}` +
        (error.field ? ` <span class="field-error">(${esc(error.field)})</span>` : '') +
        (state.run ? ' The pages still show the previous successful run.' : ''));
      render();
    } finally {
      clearInterval(timer);
      if (sequence === state.sequence) {
        state.running = false;
        setRunCardBusy(false, scenario);
      }
    }
  }

  function markOffline() {
    state.online = false;
    state.health = null;
    el('engine-status').className = 'rail-status bad';
    el('engine-status-text').textContent = 'Backend unavailable';
    el('health-retry').hidden = false;
  }

  function setRunCardBusy(busy, scenario) {
    const button = $('[data-action="run"]');
    if (button) {
      button.disabled = busy;
      button.textContent = busy ? 'Running…' : 'Run simulation';
    }
    const status = el('run-status');
    if (status) {
      status.className = busy ? 'run-status busy' : 'run-status';
      status.textContent = busy ? 'Executing… 0.0 s' : status.textContent;
    }
    document.querySelectorAll('[data-action="select-scenario"]').forEach((card) => {
      card.disabled = busy;
      card.setAttribute('aria-disabled', String(busy));
    });
  }

  async function openRun(runId) {
    try {
      const payload = await api(`/api/simulations/${runId}`);
      state.run = { run_id: payload.run_id, result: payload.result, source: 'restored' };
      state.comparison = null;
      state.teachingPolicy = payload.result.kind === 'teaching' ? payload.result.teaching.selected : null;
      render();
      showBanner('info', 'Restored a result from this session’s history. It is not a new execution.');
    } catch (error) {
      showBanner('error', `<strong>That run is no longer available.</strong> ${esc(error.message)} Run it again from the Simulation Studio.`);
    }
  }

  // ---------------------------------------------------------------- comparison

  function currentComparison() {
    if (!state.run || !state.comparison) return null;
    return state.comparison.runId === state.run.run_id ? state.comparison.result : null;
  }

  function comparisonLabel() {
    return currentComparison() ? 'Run comparison again' : 'Run comparison';
  }

  /** Reflect the in-flight state immediately, without rebuilding the page. */
  function setComparisonBusy(busy) {
    const button = $('[data-action="compare"]');
    if (!button) return;
    button.disabled = busy;
    button.textContent = busy ? 'Running four policies…' : comparisonLabel();
  }

  function isCurrentRun(runId) {
    return Boolean(state.run && state.run.run_id === runId);
  }

  async function runComparison() {
    if (!state.run || state.comparisonBusy) return;      // one request at a time
    const runId = state.run.run_id;
    state.comparisonBusy = true;
    state.comparison = null;                              // a new comparison is authoritative
    setComparisonBusy(true);
    try {
      const payload = await api('/api/comparisons', { method: 'POST', body: JSON.stringify({ run_id: runId }) });
      if (!isCurrentRun(runId)) return;                   // a newer run replaced this page
      state.comparison = { result: payload.result, runId };
    } catch (error) {
      if (!isCurrentRun(runId)) return;
      showBanner('error', `<strong>The comparison could not run.</strong> ${esc(error.message)}`);
    } finally {
      // The flag must be cleared before the final paint, or the page keeps rendering the
      // in-flight button ("Running four policies…", disabled) after the work has finished.
      state.comparisonBusy = false;
      if (isCurrentRun(runId)) render();
    }
  }

  // ---------------------------------------------------------------- inspection

  function toggleChartFit(button) {
    const frame = button.closest('.chart-frame');
    if (!frame) return;
    const fit = frame.dataset.fit !== 'true';
    frame.dataset.fit = fit ? 'true' : 'false';
    button.setAttribute('aria-pressed', String(fit));
    button.textContent = fit ? 'Fit to width' : 'Actual size';
  }

  function inspectSlice(node) {
    view().querySelectorAll('g.slice.selected').forEach((other) => other.classList.remove('selected'));
    node.classList.add('selected');
    const panel = el('inspector');
    if (!panel) return;
    panel.classList.remove('is-empty');
    const d = node.dataset;
    const waiting = d.waiting === '' || d.waiting === undefined
      ? '<span class="not-recorded">not recorded for this dispatch</span>'
      : `${esc(d.waiting)} u`;
    panel.innerHTML = `
      <h3>P${esc(d.pid)} <span class="badge plain neutral">${esc(d.policy)}</span></h3>
      <dl>
        <div><dt>Executed</dt><dd>t=${esc(d.start)} → t=${esc(d.end)} (${esc(d.duration)} u)</dd></div>
        <div><dt>Switch before</dt><dd>${esc(d.switch)} u</dd></div>
        <div><dt>Waiting added</dt><dd>${waiting}</dd></div>
        <div><dt>Arrival</dt><dd>${esc(d.arrival ?? '—')}</dd></div>
        <div><dt>Burst / priority</dt><dd>${esc(d.burst ?? '—')} / ${esc(d.priority ?? '—')}</dd></div>
        <div><dt>Completion</dt><dd>${esc(d.completion ?? '—')}</dd></div>
        <div><dt>Process waiting</dt><dd>${esc(d.processWaiting ?? '—')} u</dd></div>
      </dl>`;
  }

  function applyTeachingPolicy(name) {
    if (!state.run || state.run.result.kind !== 'teaching') return;
    const teaching = state.run.result.teaching;
    if (!teaching.policies.some((policy) => policy.name === name)) return;
    // One source of truth for the displayed trace: the tab, the Studio selector and the
    // next request all read the same field.
    state.teachingPolicy = name;
    state.options = { ...state.options, policy_view: name };
    render();
  }

  // ---------------------------------------------------------------- shared markup

  function pageHead({ title, lead, actions = '' }) {
    return `
      <header class="page-head">
        <h1>${esc(title)}</h1>
        ${lead ? `<p class="lead">${lead}</p>` : ''}
        ${actions ? `<div class="actions">${actions}</div>` : ''}
      </header>`;
  }

  function emptyState({ title, body, primary = '', secondary = '' }) {
    return `
      <div class="empty">
        <h2>${esc(title)}</h2>
        <p>${body}</p>
        <div class="actions">
          ${primary ? `<a class="btn btn-primary" href="#/studio">${esc(primary)}</a>` : ''}
          ${secondary}
        </div>
      </div>`;
  }

  function noRunPage(title, what) {
    return `
      ${pageHead({ title })}
      ${emptyState({
        title: 'No simulation is open',
        body: `${what} Nothing is shown rather than placeholder values: start a run in the Simulation Studio, or reopen one of this session’s results from the Overview. A page reload closes the open run but leaves the server’s history intact.`,
        primary: 'Open Simulation Studio',
        secondary: '<a class="btn btn-secondary" href="#/overview">See session activity</a>',
      })}`;
  }

  function runProvenance(result, runId, source) {
    return `
      <div class="provenance">
        <span><b>${source === 'live' ? 'Executed now' : 'Restored from this session'}</b></span>
        <span>run <b>${esc(runId.slice(0, 8))}</b></span>
        <span>${esc(result.kind)} scenario</span>
        <span>${esc(result.duration_ms)} ms engine time</span>
        <span>${esc(timeAgo(result.executed_at))}</span>
        <span><a href="/api/simulations/${encodeURIComponent(runId)}/export" download>Export JSON</a></span>
      </div>`;
  }

  function metricRow(metrics, baseline) {
    const diff = baseline ? metrics.avg_waiting_time - baseline.avg_waiting_time : null;
    const reduction = baseline && baseline.avg_waiting_time
      ? (diff / baseline.avg_waiting_time) * 100 : null;
    const items = [];
    items.push(`
      <div><dt>Mean waiting</dt>
        <dd>${fmt(metrics.avg_waiting_time)}<small>u</small>
          <span class="sub">${baseline ? `FCFS ${fmt(baseline.avg_waiting_time)}` : 'per process'}</span></dd></div>`);
    if (baseline) {
      items.push(`
        <div><dt>vs FCFS</dt>
          <dd class="${diff < 0 ? 'good' : diff > 0 ? 'warn' : ''}">${signed(diff)}<small>u</small>
            <span class="sub">${diff === 0 ? 'identical' : reduction === null ? 'comparison' : `${Math.abs(reduction).toFixed(1)}% ${diff < 0 ? 'lower' : 'higher'}`}</span></dd></div>`);
    }
    items.push(`
      <div><dt>Turnaround</dt><dd>${fmt(metrics.avg_turnaround_time)}<small>u</small></dd></div>
      <div><dt>Response</dt>
        <dd>${fmt(metrics.avg_response_time)}<small>u</small>
          <span class="sub">${baseline ? `FCFS ${fmt(baseline.avg_response_time)}` : 'first start'}</span></dd></div>
      <div><dt>Switches</dt><dd>${fmt(metrics.context_switches)}
          <span class="sub">${baseline ? `FCFS ${fmt(baseline.context_switches)}` : 'recorded'}</span></dd></div>
      <div><dt>CPU busy</dt><dd>${pct(metrics.cpu_utilization)}
          <span class="sub">${fmt(metrics.total_elapsed_time)} u elapsed</span></dd></div>`);
    return `<dl class="metric-row">${items.join('')}</dl>`;
  }

  function legendRow() {
    return `
      <div class="legend" aria-label="Chart legend">
        <span><i class="fcfs"></i>FCFS</span>
        <span><i class="sjf"></i>SJF</span>
        <span><i class="rr"></i>Round Robin</span>
        <span><i class="prio"></i>Priority</span>
        <span><i class="switch"></i>context switch</span>
        <span><i class="idle"></i>CPU idle</span>
        <span>◦ arrival</span>
      </div>`;
  }

  function ganttPanel(result, { schedule = result.schedule } = {}) {
    const teaching = result.kind === 'teaching' ? result.teaching : null;
    const active = teaching ? (teachingPolicyView(result) || {}).name : null;
    const svg = teaching
      ? (result.charts.gantt_variants[active] || result.charts.gantt)
      : result.charts.gantt;
    const summary = [
      `Initial policy ${schedule.initial_policy}`,
      schedule.final_policy !== schedule.initial_policy ? `final ${schedule.final_policy}` : '',
      `${schedule.intervals.length} dispatch intervals`,
      `${schedule.switch_units} u switch time`,
      `${schedule.idle_units} u idle`,
    ].filter(Boolean).join(' · ');

    return `

      <p class="section-head" style="margin:0 0 4px"><span class="hint">${esc(summary)}</span></p>
      <div class="chart-frame" data-fit="true">
        <div class="chart-tools">
          <span class="hint">Small screen: the trace is fitted to the width. “Actual size” shows full detail with sideways scrolling.</span>
          <button class="chart-toggle" type="button" data-action="toggle-chart-fit" aria-pressed="true">Fit to width</button>
        </div>
        <div class="chart-scroll" id="gantt" aria-label="Dispatch timeline; the same intervals are listed as text in the process table">${svg}</div>
      </div>
      ${legendRow()}
      <div class="inspector is-empty" id="inspector">
        <p class="hint">Select a dispatch block — click it, or focus it and press Enter — to read the recorded interval.</p>
      </div>`;
  }

  // ---------------------------------------------------------------- pages: overview

  const PAGES = {};

  PAGES.overview = () => {
    PAGES.overview.title = 'Overview';
    const health = state.health;
    const engine = health
      ? `${health.engine.platform.split('-')[0]} · Python ${health.engine.python}`
      : state.online === false ? 'backend unavailable' : 'checking…';
    const lastRun = state.run;
    const recent = state.recentRuns;
    const earlier = recent.filter((run) => !lastRun || run.run_id !== lastRun.run_id);

    return `
      ${pageHead({
        title: 'A self-healing CPU scheduler, observed end to end',
        lead: 'This console drives a single-CPU discrete-event simulator. A workload is generated, a scheduling policy dispatches it, a monitor watches the waiting time it actually achieves, and a control loop detects degradation, diagnoses it, changes policy and then verifies whether the change worked.',
        actions: `
          <a class="btn btn-primary" href="#/studio">Start a simulation</a>
          <a class="btn btn-secondary" href="#/research">Read the research results</a>`,
      })}

      <section class="section">
        <div class="panel">
          <dl class="facts">
            <div><dt>Engine</dt><dd>${esc(engine)}</dd></div>
            <div><dt>Policies</dt><dd>${esc((health && health.engine.policies || ['FCFS', 'SJF', 'Round Robin', 'Priority']).join(' · '))}</dd></div>
            <div><dt>Time units</dt><dd>Simulated, not seconds</dd></div>
            <div><dt>Session history</dt><dd>${runCountText('kept in memory')}</dd></div>
          </dl>
        </div>
      </section>

      <section class="section">
        <div class="section-head"><h2>Current simulation</h2>
          ${lastRun ? `<span class="hint">${esc(timeAgo(lastRun.result.executed_at))}</span>` : ''}</div>
        <div class="panel">
          ${lastRun ? overviewCurrentRun(lastRun) : `
            <p class="panel-intro">No simulation has been run in this session. The Studio offers a three-process teaching workload, two recorded recovery scenarios and a custom workload editor.</p>
            <p style="margin-top:16px"><a class="btn btn-primary" href="#/studio">Open Simulation Studio</a></p>`}
        </div>
      </section>

      <section class="section">
        <div class="panel flush">
          <div class="section-head" style="padding:20px 24px 0;margin:0">
            <h2>${lastRun ? 'Earlier runs in this session' : 'Runs in this session'}</h2>
            <span class="hint">Newest first · ${historyHint()}</span>
          </div>
          ${earlier.length ? `
            <div class="table-scroll">
              <table class="data" style="min-width:700px">
                <thead><tr>
                  <th scope="col">Scenario</th>
                  <th scope="col">Result</th>
                  <th scope="col" class="num">Mean waiting</th>
                  <th scope="col">Final policy</th>
                  <th scope="col">When</th>
                  <th scope="col"><span class="visually-hidden">Open</span></th>
                </tr></thead>
                <tbody>
                  ${earlier.map((run) => `
                    <tr>
                      <td class="name">${esc(run.scenario_name)}</td>
                      <td>${run.verdict ? `<span class="badge ${esc(run.verdict)}">${esc(run.verdict)}</span>` : '<span class="muted">completed</span>'}</td>
                      <td class="num">${fmt(run.mean_waiting_time)}</td>
                      <td>${esc(run.final_policy || 'n/a')}</td>
                      <td class="muted">${esc(timeAgo(run.executed_at))}</td>
                      <td class="num"><button class="linkish" type="button" data-action="open-run" data-run-id="${esc(run.run_id)}">Open</button></td>
                    </tr>`).join('')}
                </tbody>
              </table>
            </div>` : `
            <p class="panel-intro" style="padding:var(--s4) 24px var(--s5);margin:0">
              ${recent.length ? 'This run is the only one in this session so far.' : 'Completed runs appear here for as long as this server stays up. Each entry is a re-openable result, not a new execution.'}
            </p>`}
        </div>
      </section>`;
  };

  function historyHint() {
    const shown = state.recentRuns.length;
    const total = state.sessionRuns;
    if (typeof total === 'number' && total > shown) {
      return `showing the ${shown} most recent of ${total} · re-openable results, not new executions`;
    }
    return 're-openable results, not new executions';
  }

  function overviewCurrentRun(run) {
    const result = run.result;
    const view = teachingPolicyView(result);
    const metrics = view ? view.metrics : result.metrics;
    const schedule = view ? view.schedule : result.schedule;
    const verdict = result.healing && result.healing.triggered ? result.healing.incident.verdict : null;
    const diff = result.baseline ? metrics.avg_waiting_time - result.baseline.avg_waiting_time : null;
    return `
      <div class="current-run">
        <div>
          <div class="verdict-row">
            <strong>${esc(result.scenario.name)}</strong>
            ${verdict ? `<span class="badge ${esc(verdict)}">${esc(verdict)}</span>` : ''}
            <span class="badge plain neutral">${run.source === 'live' ? 'this session' : 'restored'}</span>
          </div>
          <p class="panel-intro" style="margin-top:10px">
            ${result.workload.process_count} processes under ${esc(schedule.initial_policy)}${schedule.final_policy !== schedule.initial_policy ? ` → ${esc(schedule.final_policy)}` : ''}.
            All analysis pages read this trace until another run replaces it.
          </p>
          <div class="navigate" style="margin-top:14px">
            <a href="#/workload">Inspect the workload and timeline →</a>
            ${result.healing ? '<a href="#/recovery">Read the recovery decision →</a>' : ''}
            <a href="#/compare">Compare policies on this workload →</a>
          </div>
        </div>
        <dl class="facts facts-pairs">
          <div><dt>Mean waiting</dt><dd>${fmt(metrics.avg_waiting_time)} u</dd></div>
          <div><dt>Final policy</dt><dd>${esc(schedule.final_policy)}</dd></div>
          <div><dt>Against FCFS</dt><dd>${diff === null ? 'n/a' : `${signed(diff)} u`}</dd></div>
          <div><dt>CPU busy</dt><dd>${pct(metrics.cpu_utilization)}</dd></div>
        </dl>
      </div>`;
  }

  // ---------------------------------------------------------------- pages: studio

  PAGES.studio = () => {
    PAGES.studio.title = 'Simulation Studio';
    const scenario = currentScenario();
    const groups = new Map();
    state.scenarios.forEach((s) => {
      const key = scenarioCategory(s);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(s);
    });
    const library = [...groups.entries()].map(([group, items]) => `
      <div class="library-group">
        <h3 class="library-label">${esc(group)}</h3>
        <div class="scenario-list">
          ${items.map((s) => `
            <button class="scenario" type="button" data-action="select-scenario" data-id="${esc(s.id)}"
                    aria-pressed="${s.id === state.scenarioId}" ${state.running ? 'disabled' : ''}>
              <span class="s-name">${esc(s.name)}</span>
              <span class="s-desc">${esc(s.tagline || s.description)}</span>
              <span class="s-state">${s.id === state.scenarioId ? '✓ Selected' : 'Select'}</span>
            </button>`).join('')}
        </div>
      </div>`).join('');

    return `
      ${pageHead({
        title: 'Simulation Studio',
        lead: 'Choose what to execute, review the settings the engine will receive, then run it. The trace is read in the Workload Explorer and the recovery decision in Recovery Analysis; the run itself stays on this page.',
      })}
      <section class="section">
        <div class="panel">
          <span class="step-label"><span class="n">1</span> Choose a scenario</span>
          <div class="library">${library || '<p class="panel-intro">Scenarios are not available — the backend could not be reached.</p>'}</div>
        </div>
      </section>

      <div class="split">
        <section class="section">
          <div class="panel">
            <span class="step-label"><span class="n">2</span> Review the configuration</span>
            ${scenario ? configPanel(scenario) : '<p class="panel-intro">Select a scenario first.</p>'}
          </div>
        </section>

        <aside class="run-card" aria-label="Execute the simulation">
          <span class="step-label"><span class="n">3</span> Execute</span>
          <h2>Run the engine</h2>
          <p class="panel-intro" style="margin-top:6px">The backend schedules the workload with the settings on the left and returns the recorded trace.</p>
          <dl class="summary" id="run-summary">${runSummaryRows(scenario)}</dl>
          <button class="btn btn-primary btn-large" type="button" data-action="run"
                  ${!state.online || state.running || !scenario ? 'disabled' : ''}>
            ${state.running ? 'Running…' : 'Run simulation'}
          </button>
          <p class="run-status ${state.running ? 'busy' : ''}" id="run-status" role="status" aria-live="polite"></p>
          ${studioResult()}
        </aside>
      </div>`;
  };

  function runSummaryRows(scenario) {
    if (!scenario) return '';
    const rows = scenario.kind === 'custom'
      ? [
          ['Scenario', 'Custom workload'],
          ['Processes', `${(state.options.processes || []).length}`],
          ['Policy', state.options.healing
            ? 'FCFS, then the recovery rule'
            : esc(state.options.policy || 'FCFS')],
          ['Quantum / switch', `${esc(state.options.quantum ?? 4)} / ${esc(state.options.switching_cost ?? 0)}`],
          ['Self-healing', state.options.healing ? 'on' : 'off'],
        ]
      : [
          ['Scenario', esc(scenario.name)],
          ['Workload', esc((scenario.fixed_parameters || {}).family || 'three processes')],
          ['Parameters', Object.entries(scenario.fixed_parameters || {})
            .filter(([k]) => k !== 'family')
            .map(([k, v]) => `${k.replace(/_/g, ' ')} ${v}`).join(', ') || 'fixed'],
          ['Recovery rule', (scenario.fixed_parameters || {}).rule ? 'waiting-time rule' : 'not used'],
        ];
    return rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join('');
  }

  function studioResult() {
    if (!state.run) return '';
    const result = state.run.result;
    const view = teachingPolicyView(result);
    const metrics = view ? view.metrics : result.metrics;
    const schedule = view ? view.schedule : result.schedule;
    const healing = result.healing;
    const verdict = healing && healing.triggered ? healing.incident.verdict : null;
    return `
      <div class="result-note">
        <div class="verdict-row">
          <span class="badge plain neutral">${state.run.source === 'live' ? 'Executed' : 'Restored'}</span>
          ${verdict ? `<span class="badge ${esc(verdict)}">${esc(verdict)}</span>` : ''}
        </div>
        <p><strong>${esc(result.scenario.name)}</strong> — mean waiting ${fmt(metrics.avg_waiting_time)} u${result.baseline ? `, FCFS ${fmt(result.baseline.avg_waiting_time)} u` : ''}. ${view ? `Showing ${esc(view.name)}.` : `Final policy ${esc(schedule.final_policy)}.`}</p>
        <div class="navigate">
          <a href="#/workload">Workload Explorer →</a>
          ${result.healing ? '<a href="#/recovery">Recovery Analysis →</a>' : ''}
          <a href="#/compare">Policy Comparison →</a>
        </div>
      </div>`;
  }

  function configPanel(scenario) {
    if (scenario.kind === 'custom') return customConfig(scenario);
    const fixed = scenario.fixed_parameters || {};
    const facts = Object.entries(fixed).map(([key, value]) => `
      <div><dt>${esc(key.replace(/_/g, ' '))}</dt><dd class="mono">${esc(value)}</dd></div>`).join('');
    const policyField = scenario.kind === 'teaching' ? `
      <div class="field" style="max-width:280px">
        <label for="policy-view">Policy view</label>
        <select id="policy-view" data-option="policy_view">
          ${['FCFS', 'SJF', 'Priority', 'Round Robin'].map((name) => `
            <option value="${esc(name)}" ${(state.options.policy_view || 'FCFS') === name ? 'selected' : ''}>${esc(name)}</option>`).join('')}
        </select>
        <span class="help">All four policies run on the same three processes. The selection only decides which executed trace is shown first.</span>
      </div>` : `
      <p class="panel-intro">This fixture is generated from the frozen configuration, so the same seed always reproduces the same trace. The recovery rule, thresholds and observation window are fixed by the engine and cannot be changed here.</p>`;
    return `
      <p class="panel-intro" style="max-width:70ch">${esc(scenario.description)}</p>
      <p class="panel-intro" style="max-width:70ch;margin-top:8px"><strong>Expected behaviour.</strong> ${esc(scenario.expectation)}</p>
      <dl class="facts" style="margin-top:24px">${facts}</dl>
      <div style="margin-top:24px">${policyField}</div>
      <p class="field-error" id="config-error"></p>`;
  }

  function customConfig(scenario) {
    const processes = state.options.processes || scenario.defaults.processes;
    const rows = processes.map((p, index) => `
      <tr data-row="${index}">
        <td><input type="number" min="0" step="1" data-key="pid" value="${esc(p.pid)}" aria-label="Process ${index + 1} id"></td>
        <td><input type="number" min="0" step="1" data-key="arrival_time" value="${esc(p.arrival_time)}" aria-label="Process ${index + 1} arrival time"></td>
        <td><input type="number" min="1" step="1" data-key="burst_time" value="${esc(p.burst_time)}" aria-label="Process ${index + 1} burst time"></td>
        <td><input type="number" min="0" step="1" data-key="priority" value="${esc(p.priority)}" aria-label="Process ${index + 1} priority"></td>
        <td><button class="btn-quiet" type="button" data-action="remove-process" data-index="${index}"
                    aria-label="Remove process ${index + 1}" style="color:var(--muted);padding:2px 6px">×</button></td>
      </tr>`).join('');
    const policies = ['FCFS', 'SJF', 'Priority', 'Round Robin'].map((name) => `
      <option value="${esc(name)}" ${(state.options.policy || 'FCFS') === name ? 'selected' : ''}>${esc(name)}</option>`).join('');
    const healingOn = Boolean(state.options.healing);
    return `
      <p class="panel-intro" style="max-width:70ch">${esc(scenario.description)}</p>
      <div class="field-grid" style="margin-top:24px">
        <div class="field">
          <label for="policy-select">Scheduling policy</label>
          <select id="policy-select" data-option="policy" ${healingOn ? 'disabled' : ''}>${policies}</select>
          <span class="help" id="policy-healing-note" ${healingOn ? '' : 'hidden'}>The self-healing loop always starts from FCFS and chooses the policy itself, so this selector is used only when the loop is off.</span>
        </div>
        <div class="field">
          <label for="quantum-input">Round-robin quantum</label>
          <input id="quantum-input" type="number" min="1" max="64" step="1" data-option="quantum" value="${esc(state.options.quantum ?? 4)}">
        </div>
        <div class="field">
          <label for="switch-input">Switch cost (units)</label>
          <input id="switch-input" type="number" min="0" max="10" step="1" data-option="switching_cost" value="${esc(state.options.switching_cost ?? 0)}">
        </div>
      </div>
      <label class="check" style="margin-top:20px">
        <input type="checkbox" data-option="healing" ${state.options.healing ? 'checked' : ''}>
        <span>Run the self-healing loop on this workload
          <small>Uses the frozen detector, waiting-time rule and verifier. A short workload may legitimately produce no incident.</small></span>
      </label>

      <div class="section-head" style="margin-top:32px"><h2 style="font-size:13px">Processes</h2>
        <span class="hint">1–40 processes · burst ≥ 1 · arrival ≥ 0 · priority ≥ 0</span></div>
      <div class="process-editor">
        <table>
          <thead><tr><th>ID</th><th>Arrival</th><th>Burst</th><th>Priority</th><th></th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div style="margin-top:12px;display:flex;gap:16px">
        <button class="linkish" type="button" data-action="add-process">+ Add process</button>
        <button class="linkish" type="button" data-action="reset-processes">Reset to the default workload</button>
      </div>
      <p class="field-error" id="config-error"></p>`;
  }

  // ---------------------------------------------------------------- pages: workload

  PAGES.workload = () => {
    PAGES.workload.title = 'Workload Explorer';
    if (!state.run) return noRunPage('Workload Explorer', 'The timeline and the process table describe the current run. Run a simulation in the Studio to produce a trace.');
    const { result, run_id: runId, source } = state.run;
    const view = teachingPolicyView(result);
    const metrics = view ? view.metrics : result.metrics;
    const schedule = view ? view.schedule : result.schedule;
    const rows = workloadRows(result);
    const sorted = [...rows].sort((a, b) => {
      const l = a[state.sort.key];
      const r = b[state.sort.key];
      if (l === undefined || r === undefined) return 0;
      return (l > r ? 1 : l < r ? -1 : 0) * state.sort.direction;
    });
    const columns = [
      ['pid', 'PID', false], ['arrival_time', 'Arrival', true], ['burst_time', 'Burst', true],
      ['priority', 'Priority', true], ['start_time', 'First start', true], ['completion_time', 'Completion', true],
      ['waiting_time', 'Waiting', true], ['turnaround_time', 'Turnaround', true], ['response_time', 'Response', true],
    ];
    const head = columns.map(([key, label, numeric]) => `
      <th scope="col" class="${numeric ? 'num' : ''} sortable" data-action="sort" data-key="${key}"
          aria-sort="${state.sort.key === key ? (state.sort.direction === 1 ? 'ascending' : 'descending') : 'none'}">${label}</th>`).join('');
    const body = sorted.map((row) => `
      <tr>
        <td data-label="PID"><span class="process-key ${POLICY_CLASS[row.policy] || ''}">P${esc(row.pid)}</span></td>
        ${columns.slice(1).map(([key, label]) => `<td class="num" data-label="${esc(label)}">${esc(row[key] ?? '—')}</td>`).join('')}
      </tr>`).join('');

    return `
      ${pageHead({
        title: 'Workload Explorer',
        lead: `Recorded execution of ${result.workload.process_count} processes for “${esc(result.scenario.name)}” under ${esc(schedule.initial_policy)}${schedule.final_policy !== schedule.initial_policy ? ` → ${esc(schedule.final_policy)}` : ''}. Charts replay the completed trace; nothing is animated.`,
      })}
      ${runProvenance(result, runId, source)}
      ${teachingPolicyTabs(result)}

      <section class="section" style="margin-top:${view ? '20px' : '28px'}">
        ${metricRow(metrics, result.baseline)}
      </section>

      <section class="section">
        <div class="panel">
          <div class="section-head">
            <h2>Dispatch timeline</h2>
            <span class="hint">Simulated time units</span>
          </div>
          ${ganttPanel(result, { schedule })}
        </div>
      </section>

      <section class="section">
        <div class="panel">
          <div class="section-head">
            <h2>Processes</h2>
            <span class="hint">${sorted.length} processes · total waiting ${rows.reduce((s, r) => s + (r.waiting_time || 0), 0)} u · click a column to sort</span>
          </div>
          <div class="table-scroll">
            <table class="data stacked">
              <thead><tr>${head}</tr></thead>
              <tbody>${body}</tbody>
            </table>
          </div>
          <p class="table-note">Waiting = completion − arrival − burst · turnaround = completion − arrival ·
            response = first start − arrival. All values are computed by <code>evaluation/metrics.py</code>
            from the executed trace.</p>
        </div>
      </section>`;
  };

  function teachingPolicyView(result) {
    const teaching = result.kind === 'teaching' ? result.teaching : null;
    if (!teaching) return null;
    const name = state.teachingPolicy || teaching.selected;
    return teaching.policies.find((policy) => policy.name === name) || teaching.policies[0];
  }

  function teachingPolicyTabs(result) {
    if (result.kind !== 'teaching' || !result.teaching) return '';
    const active = (teachingPolicyView(result) || {}).name;
    return `
      <div class="policy-tabs" role="tablist" aria-label="Teaching example policy view">
        <span class="tabs-label">Policy view</span>
        ${result.teaching.policies.map((policy) => `
          <button class="tab" role="tab" type="button" data-action="teaching-policy" data-policy="${esc(policy.name)}"
                  aria-selected="${policy.name === active}">${esc(policy.name)}</button>`).join('')}
        <span class="tabs-note">Each class was executed separately on the same three processes.</span>
      </div>`;
  }

  function workloadRows(result) {
    const view = teachingPolicyView(result);
    if (view) return view.outcomes.map((row) => ({ ...row, policy: view.name }));
    // The chip colour names the policy that first dispatched each process, not the policy
    // in force at the end of the trace: after a mid-trace recovery the two differ, and
    // painting all fifteen chips with the final policy misreports where the work ran.
    const firstDispatch = new Map();
    (result.schedule.intervals || []).forEach((interval) => {
      if (!firstDispatch.has(interval.pid)) firstDispatch.set(interval.pid, interval.policy);
    });
    return (result.outcomes || []).map((row) => ({
      ...row,
      policy: firstDispatch.get(row.pid) || result.schedule.final_policy,
    }));
  }

  // ---------------------------------------------------------------- pages: recovery

  PAGES.recovery = () => {
    PAGES.recovery.title = 'Recovery Analysis';
    if (!state.run) return noRunPage('Recovery Analysis', 'The control-loop record belongs to the current run. Run a recovery scenario in the Simulation Studio to produce one.');
    const { result, run_id: runId, source } = state.run;
    const healing = result.healing;

    if (!healing) {
      return `
        ${pageHead({ title: 'Recovery Analysis', lead: 'The self-healing control loop was not part of this run.' })}
        ${runProvenance(result, runId, source)}
        <section class="section" style="margin-top:28px">
          <div class="panel">
            <h2 style="font-size:15px">No control loop in this execution</h2>
            <p class="panel-intro" style="margin-top:8px;max-width:70ch">
              This run executed <strong>${esc(result.settings.policy || 'a single policy')}</strong> directly, so no degradation was
              monitored and no intervention was attempted. Choose a recovery scenario, or enable the self-healing loop on a
              custom workload, to produce a detection → verification record.
            </p>
            <p style="margin-top:20px"><a class="btn btn-secondary" href="#/studio">Open Simulation Studio</a></p>
          </div>
        </section>`;
    }

    if (!healing.triggered) {
      return `
        ${pageHead({
          title: 'Recovery Analysis',
          lead: 'The control loop ran for this workload but declared no incident.',
        })}
        ${runProvenance(result, runId, source)}
        <section class="section" style="margin-top:28px">
          <div class="verdict held">
            <h2>No intervention was attempted <span class="badge neutral plain">not triggered</span></h2>
            <p>${esc(healing.summary)}</p>
            <p class="panel-intro" style="margin-top:10px">${esc(healing.final_text)}</p>
          </div>
        </section>
        <section class="section">
          <div class="panel">
            <div class="section-head"><h2>Timeline</h2><span class="hint">${esc(result.settings.policy || '')}</span></div>
            ${ganttPanel(result, { schedule: result.schedule })}
          </div>
        </section>`;
    }

    const incident = healing.incident;
    const tone = incident.verdict;
    const stages = healing.stages.map((stage) => `
      <li data-tone="${esc(stage.tone_key || stageTone(stage))}">
        <span class="marker" aria-hidden="true"></span>
        <h3>${esc(stage.title)} ${stageMeta(stage)}</h3>
        <p>${esc(stage.detail || '')}</p>
        <div class="figures">${stage.lines.map((line) => `<span>${esc(line)}</span>`).join('')}</div>
      </li>`).join('');

    return `
      ${pageHead({
        title: 'Recovery Analysis',
        lead: `What the controller observed, decided and verified for this trace: ${esc(incident.settings_before_policy)} → ${esc(incident.settings_after_policy)}, closing at t=${esc(incident.window_end_time)}.`,
      })}
      ${runProvenance(result, runId, source)}

      <section class="section" style="margin-top:28px">
        <div class="verdict ${esc(tone)}">
          <h2>
            <span class="badge ${esc(tone)}">${esc(tone)}</span>
            ${esc(incident.settings_before_policy)} → ${esc(incident.settings_after_policy)}
          </h2>
          <p>${esc(healing.summary)}</p>
          <div class="meta">
            <span>reason <b>${esc(incident.verifier_reason)}</b></span>
            <span>window <b>t=${esc(incident.simulated_time)} → t=${esc(incident.window_end_time)}</b></span>
            <span>final policy <b>${esc(result.schedule.final_policy)}</b></span>
            <span>confirmed failure <b>${incident.confirmed_failure ? 'yes' : 'no'}</b></span>
          </div>
        </div>
      </section>

      <section class="section">
        <div class="panel">
          <div class="section-head"><h2>Verification</h2><span class="hint">Residual against the declared null range</span></div>
          <div class="chart-scroll">${result.charts.verification || ''}</div>
          <p class="chart-caption">The accept and reject boundaries come from the incident record; the marker is the residual this run recorded.</p>
        </div>
      </section>

      <section class="section">
        <div class="panel">
          <div class="section-head"><h2>Event timeline</h2><span class="hint">Recorded stages, in order</span></div>
          <div class="lifecycle"><ol>${stages}</ol></div>
        </div>
      </section>

      <section class="section">
        <div class="panel">
          <div class="section-head"><h2>Recorded detail</h2><span class="hint">Values as stored by the controller</span></div>
          <div class="detail-cols">
            <section>
              <h3>Detection</h3>
              <dl>
                <dt>Decision</dt><dd>d${esc(incident.decision_index)} · t=${esc(incident.simulated_time)}</dd>
                <dt>Window mean</dt><dd>${fmt(incident.window_mean_at_trigger, 3)}</dd>
                <dt>Threshold</dt><dd>${fmt(incident.trigger_threshold, 3)}</dd>
                <dt>Reference ± band</dt><dd>${fmt(incident.reference_at_trigger, 3)} ± ${fmt(incident.band_at_trigger, 3)}</dd>
                <dt>Suspicion evaluations</dt><dd>${esc(incident.suspicion_evaluations)}</dd>
                <dt>Ready at trigger</dt><dd>${esc(incident.ready_count_at_trigger)}</dd>
                <dt>Detector reason</dt><dd>${esc(incident.detector_reason)}</dd>
              </dl>
            </section>
            <section>
              <h3>Diagnosis and action</h3>
              <dl>
                <dt>Diagnosis</dt><dd>${esc(incident.diagnosis)}</dd>
                <dt>Matched</dt><dd>${listText(incident.diagnosis_matched || 'none')}</dd>
                <dt>Context state</dt><dd>${esc(incident.context_state_index)}/${esc(incident.state_space_size ?? 648)}</dd>
                <dt>Offered actions</dt><dd>${listText(incident.offered_actions)}</dd>
                <dt>Selected</dt><dd>${esc(incident.action_name)}</dd>
                <dt>Rule note</dt><dd>${esc(incident.request_note)}</dd>
              </dl>
            </section>
            <section>
              <h3>Verification</h3>
              <dl>
                <dt>Residual</dt><dd>${signed(incident.closure_residual, 4)}</dd>
                <dt>Accept ≥</dt><dd>${signed(incident.accept_residual_factor)}</dd>
                <dt>Reject ≤</dt><dd>${signed(incident.reject_residual_factor)}</dd>
                <dt>Window decisions</dt><dd>${esc(incident.verification_decisions)}</dd>
                <dt>Completions</dt><dd>${esc(incident.verification_completions)}</dd>
                <dt>Rollback target</dt><dd>${esc(incident.rollback_target_policy || 'none — configuration kept')}</dd>
                <dt>Reward</dt><dd>${signed(incident.reward, 4)}</dd>
              </dl>
            </section>
          </div>
          <p class="table-note">Diagnosis statement as recorded: “${esc(incident.diagnosis_statement)}”</p>
        </div>
      </section>

      <p class="page-foot-link">The dispatch timeline for this trace, with the same control events marked, is on the
        <a href="#/workload">Workload Explorer</a>.</p>`;
  };

  /**
   * Fallback tone for a stage record that predates ``tone_key``.
   *
   * The live payload always carries a semantic key; this only covers a result restored
   * from an older server. "Final decision" is deliberately not hard-coded to the success
   * colour: an inconclusive rollback must not be painted as a recovery.
   */
  function stageTone(stage) {
    const title = stage.title.toLowerCase();
    if (title.includes('degradation')) return 'warn';
    if (title.includes('verification') || title.includes('final decision')) {
      if (stage.lines && /accepted/.test(stage.lines[0] || '')) return 'ok';
      if (stage.lines && /rolled back|inconclusive/.test(stage.lines[0] || '')) return 'warn';
    }
    if (title.includes('monitoring') || title.includes('diagnosis')) return 'muted';
    return 'accent';
  }

  function stageMeta(stage) {
    const when = (stage.lines || []).find((line) => line.startsWith('t=') || line.includes('window'));
    return when ? `<span class="when">${esc(when)}</span>` : '';
  }

  // ---------------------------------------------------------------- pages: compare

  PAGES.compare = () => {
    PAGES.compare.title = 'Policy Comparison';
    if (!state.run) return noRunPage('Policy Comparison', 'A comparison runs on the workload of the current run. Run a simulation in the Studio first.');
    const { result, run_id: runId, source } = state.run;
    const comparison = currentComparison();

    return `
      ${pageHead({
        title: 'Policy Comparison',
        lead: 'FCFS, SJF, Round Robin and Priority executed separately on the same process list, so the difference between them belongs to the policy rather than to the workload.',
      })}
      ${runProvenance(result, runId, source)}
      <section class="section" style="margin-top:24px">
        <div class="split-wide">
          <div class="panel">
            <div class="section-head">
              <h2>Workload under comparison</h2>
              <span class="hint">${result.workload.process_count} processes</span>
            </div>
            <dl class="facts facts-pairs">
              <div><dt>Source run</dt><dd class="mono">${esc(runId.slice(0, 8))}</dd></div>
              <div><dt>Quantum</dt><dd class="mono">${esc((result.settings || {}).quantum ?? 4)}</dd></div>
              <div><dt>Fingerprint</dt><dd class="mono">${esc(result.workload.fingerprint)}</dd></div>
              <div><dt>Switch cost</dt><dd class="mono">${esc((result.settings || {}).switching_cost ?? 0)} u</dd></div>
            </dl>
            <p class="table-note">These are the exact conditions the current run was executed under; the comparison reuses them.</p>
          </div>
          <div class="panel">
            <h2 style="font-size:15px">Execute the comparison</h2>
            <p class="panel-intro" style="margin-top:8px">Four simulations run now on this workload. Nothing is reused from earlier comparisons.</p>
            <button class="btn btn-primary" type="button" data-action="compare" style="margin-top:16px"
              ${state.comparisonBusy ? 'disabled' : ''}>${state.comparisonBusy ? 'Running four policies…' : comparisonLabel()}</button>
          </div>
        </div>
      </section>
      ${comparison ? comparisonBody(comparison) : `
        <section class="section">
          <div class="panel">
            <p class="panel-intro">No comparison has been executed for this run yet. The button above runs all four policies on the same processes.</p>
          </div>
        </section>`}`;
  };

  function comparisonBody(comparison) {
    const policies = comparison.policies;
    const best = comparison.best_by_waiting;
    const fcfs = policies.find((p) => p.name === 'FCFS');
    const bestRow = policies.find((p) => p.name === best);
    return `
      <section class="section">
        <p class="panel-intro" style="max-width:78ch">
          On this workload <strong>${esc(best)}</strong> recorded the lowest mean waiting time
          (${fmt(bestRow.metrics.avg_waiting_time)} u against ${fmt(fcfs.metrics.avg_waiting_time)} u for FCFS).
          A single workload is a demonstration, not evidence of general superiority.
        </p>
      </section>
      <section class="section">
        <div class="panel">
          <div class="section-head"><h2>Mean waiting and response time</h2>
            <span class="hint">executed ${esc(timeAgo(comparison.executed_at))} · ${esc(comparison.duration_ms)} ms</span></div>
          <div class="chart-scroll">${comparison.charts.waiting}</div>
          <div class="chart-scroll" style="margin-top:24px">${comparison.charts.response}</div>
          <p class="chart-caption">Bars are simulated time units per process. Lower is better for both metrics.</p>
        </div>
      </section>
      <section class="section">
        <div class="panel flush">
          <div class="section-head" style="padding:20px 24px 0;margin:0">
            <h2>Policy results</h2>
            <span class="hint">Four separate executions · times in simulated units</span>
          </div>
          <div class="table-scroll">
            <table class="data stacked" style="min-width:660px">
              <thead><tr>
                <th scope="col">Policy</th>
                <th scope="col" class="num">Mean waiting</th>
                <th scope="col" class="num">Turnaround</th>
                <th scope="col" class="num">Response</th>
                <th scope="col" class="num">Switches</th>
                <th scope="col" class="num">CPU busy</th>
                <th scope="col" class="num">Elapsed</th>
              </tr></thead>
              <tbody>
                ${policies.map((row) => `
                  <tr>
                    <td class="name" data-label="Policy">${esc(row.name)}${row.name === best ? ' <span class="badge accepted plain">best</span>' : ''}</td>
                    <td class="num ${row.name === best ? 'best' : ''}" data-label="Mean waiting">${fmt(row.metrics.avg_waiting_time)}</td>
                    <td class="num" data-label="Turnaround">${fmt(row.metrics.avg_turnaround_time)}</td>
                    <td class="num" data-label="Response">${fmt(row.metrics.avg_response_time)}</td>
                    <td class="num" data-label="Switches">${fmt(row.metrics.context_switches)}</td>
                    <td class="num" data-label="CPU busy">${pct(row.metrics.cpu_utilization)}</td>
                    <td class="num" data-label="Elapsed">${fmt(row.metrics.total_elapsed_time)}</td>
                  </tr>`).join('')}
              </tbody>
            </table>
          </div>
          <p class="table-note" style="padding:16px 24px 20px">Metrics come from <code>evaluation/metrics.py</code> for each execution.
            Round-robin results depend on the quantum shown above; switch cost is charged per preemption.</p>
        </div>
      </section>`;
  }

  // ---------------------------------------------------------------- pages: research

  PAGES.research = () => {
    PAGES.research.title = 'Research Results';
    const head = pageHead({
      title: 'Research Results',
      lead: 'Aggregate measurements from the committed held-out study. These are not measurements of the run shown elsewhere in this console, and they were not produced during this session.',
    });
    if (state.researchError) {
      return `${head}<div class="panel"><h2 style="font-size:15px">The study artifacts could not be read</h2>
        <p class="panel-intro" style="margin-top:8px">${esc(state.researchError.message)}</p>
        <button class="btn btn-secondary" type="button" data-action="retry-research" style="margin-top:16px">Retry</button></div>`;
    }
    if (!state.research) return `${head}<div class="panel"><p class="panel-intro">Reading the committed study tables…</p></div>`;

    const r = state.research;
    const primary = r.arms.filter((a) => ['fixed_fcfs', 'fixed_sjf', 'rule_wait', 'q_recovery'].includes(a.arm));
    const learned = r.arms.find((a) => a.arm === 'q_recovery');
    const rule = r.arms.find((a) => a.arm === 'rule_wait');
    const comparison = r.comparisons.find((c) => c.target === 'q_recovery' && c.reference === 'rule_wait' && c.metric === 'avg_waiting_time');
    const limits = r.measured_limits;
    const share = (value) => (value === null || value === undefined ? 'n/a' : `${(Number(value) * 100).toFixed(1)}%`);

    return `
      ${head}
      <section class="section">
        <div class="panel">
          <div class="section-head"><h2>Design and provenance</h2><span class="hint">committed artifacts, read only</span></div>
          <dl class="facts">
            <div><dt>Source file</dt><dd class="mono">${esc(r.source)}</dd></div>
            <div><dt>Split</dt><dd>${esc(r.split)}</dd></div>
            <div><dt>Workloads per arm</dt><dd class="mono">${esc(r.workloads ?? 'n/a')}</dd></div>
            <div><dt>Final-test seed</dt><dd class="mono">${esc(r.final_test_seed ?? 'n/a')}</dd></div>
          </dl>
        </div>
      </section>

      <section class="section">
        <div class="panel">
          <div class="section-head"><h2>Mean waiting time by arm</h2><span class="hint">simulated time units per process</span></div>
          <div class="chart-scroll">${r.charts.waiting}</div>
          <p class="chart-caption">Every bar is a committed aggregate over the whole held-out split. The learned-recovery arms are shaded separately from the fixed policies and the predeclared rules.</p>
        </div>
      </section>

      <section class="section">
        <div class="research-grid">
          <div class="panel">
            <div class="section-head"><h2>Primary comparison</h2></div>
            <ul class="arm-list">
              ${primary.map((arm) => `<li><span class="k">${esc(arm.label)}</span><span class="v">${fmt(arm.mean_waiting_time)}</span></li>`).join('')}
            </ul>
            ${learned && rule ? `
              <p class="table-note">Q-learning recovery ${fmt(learned.mean_waiting_time)} u against the waiting-time rule
                ${fmt(rule.mean_waiting_time)} u${comparison ? ` — difference ${signed(comparison.difference)} u
                (95% CI ${signed(comparison.ci_low)} to ${signed(comparison.ci_high)})` : ''}.</p>` : ''}
          </div>
          <div class="panel">
            <div class="section-head"><h2>Interpretation limits</h2></div>
            <ul class="notes">
              ${(r.caveats || []).map((line) => `<li>${esc(line)}</li>`).join('')}
            </ul>
          </div>
        </div>
      </section>

      ${limits ? `
      <section class="section">
        <div class="panel">
          <div class="section-head"><h2>What the verification measured</h2><span class="hint">counterfactual audit, same split</span></div>
          <dl class="facts">
            <div><dt>Interventions audited</dt><dd>${fmt(limits.interventions)}</dd></div>
            <div><dt>Accepted windows</dt><dd>${fmt(limits.accepted)}${limits.accepted_helped_share === 1 ? ' · all helpful' : ` · ${share(limits.accepted_helped_share)} helpful`}</dd></div>
            <div><dt>Helpful changes certified</dt><dd>${share(limits.certifiable_share)}</dd></div>
            <div><dt>Incidents on the control</dt><dd>${fmt(limits.control_incidents_per_trace)} per trace</dd></div>
          </dl>
          <p class="table-note">
            The verifier is a safety filter, not a benefit estimator. It accepted
            ${fmt(limits.accepted)} of ${fmt(limits.interventions)} interventions; the audit found
            ${share(limits.base_helped_share)} of all interventions helpful, and the ${share(limits.certifiable_share)}
            figure is the share of those helpful changes the verifier could certify at its declared
            false-accept budget of ${share(limits.declared_false_accept_rate)}. The detector also fires on
            stable control traces (${fmt(limits.control_incidents_per_trace)} per trace), so it is not a
            zero-false-alarm detector. Method, denominators and limitations are recorded in
            <code>docs/EVALUATION_RESULTS.md</code> and <code>docs/LIMITATIONS.md</code>.
          </p>
        </div>
      </section>` : ''}
    `;
  };

  // ---------------------------------------------------------------- pages: about

  PAGES.about = () => {
    PAGES.about.title = 'About & Method';
    return `
      ${pageHead({
        title: 'About & Method',
        lead: 'What is simulated, how a trace is produced, what the numbers mean, and what this project does not claim.',
      })}
      <div class="doc-layout">
        <div class="panel doc">
          <h2>Purpose</h2>
          <p>
            The project studies a safety-aware self-healing scheduler on dynamic workloads. A discrete-event
            simulator runs a reproducible workload, the scheduler dispatches it, a monitor measures the waiting
            time actually achieved, a detector declares degradation when the rolling statistic leaves the trace’s
            own stable band, a diagnosis names a likely cause, a recovery policy chooses one action, and a verifier
            decides after an observation window whether the change worked.
          </p>
          <p>
            The research question is not whether reinforcement learning can find a better policy, but whether an
            adaptive scheduler can recognise degradation, recover safely, and avoid harmful interventions under
            uncertainty.
          </p>

          <h2>Architecture</h2>
          <div class="arch">
            <ol class="layers">
              <li><b>Workload</b> — <code>workload/</code> generates processes from a declared family and seed and fingerprints them, so identical inputs are provable.</li>
              <li><b>Scheduler</b> — <code>scheduler/</code> holds FCFS, SJF, Round Robin and Priority plus the runtime simulator that records each dispatch as a slice.</li>
              <li><b>Control loop</b> — <code>healing/</code> monitors, detects, diagnoses, selects an action, verifies the window and rolls back when required.</li>
              <li><b>Metrics and evaluation</b> — <code>evaluation/</code> computes waiting, turnaround, response, utilisation, throughput and switches from the recorded result.</li>
              <li><b>Presentation</b> — this console and the static report builder read those results; they never schedule, detect or verify anything themselves.</li>
            </ol>
          </div>

          <h2>Definitions</h2>
          <dl>
            <dt>Waiting time</dt><dd>Completion − arrival − burst, per process; the table mean is the sum divided by the process count.</dd>
            <dt>Turnaround</dt><dd>Completion − arrival.</dd>
            <dt>Response</dt><dd>First start − arrival.</dd>
            <dt>CPU utilisation</dt><dd>Busy time as a percentage of the elapsed time of the trace.</dd>
            <dt>Throughput</dt><dd>Completed processes per simulated time unit.</dd>
            <dt>Context switch</dt><dd>A change of the running process. The optional per-switch cost is charged as idle switch time.</dd>
            <dt>Inconclusive</dt><dd>The verification residual stayed inside the declared null range: the change is neither proven nor rejected, so the declared posture rolls it back without recording a confirmed failure.</dd>
          </dl>

          <h2>Units and assumptions</h2>
          <ul>
            <li>All times are simulated integer time units, not seconds, and the machine is a single CPU.</li>
            <li>Burst times are known to the scheduler unless the uncertainty extension is enabled; the live demo uses the rule-based recovery policy, not a trained model.</li>
            <li>Research aggregates are committed artifacts from the frozen final-test split; they are never recomputed or mixed with a live run in this console.</li>
            <li>Each API run executes the repository’s own engine in an isolated temporary directory; committed research evidence is only read.</li>
          </ul>

          <h2>Further reading</h2>
          <ul>
            <li><code>docs/LIVE_DASHBOARD.md</code> — API reference, startup and troubleshooting.</li>
            <li><code>docs/TEACHER_LIVE_DEMO.md</code> — classroom sequence and the hand-worked example.</li>
            <li><code>docs/ARCHITECTURE.md</code> — layer-by-layer design and data flow.</li>
            <li><code>docs/LIMITATIONS.md</code> — recorded scope and validation boundaries.</li>
          </ul>
        </div>

        <aside class="doc-aside">
          <div class="panel">
            <h2>What this does not claim</h2>
            <ul class="notes">
              <li>This is a synthetic simulation, not a kernel scheduler: no real processes, no real time, no hardware measurement.</li>
              <li>Charts replay completed traces. Nothing is animated as if it were executing live.</li>
              <li>A policy comparison on one workload demonstrates behaviour for that workload only.</li>
              <li>The study did not establish that Q-learning recovery outperforms the waiting-time rule.</li>
            </ul>
          </div>
          <div class="panel">
            <h2>Run it locally</h2>
            <p>From the repository root, in a virtual environment with <code>requirements-dashboard.txt</code> installed:</p>
            <pre class="cmd">.\\.venv\\Scripts\\python.exe tools\\serve_dashboard.py</pre>
            <p class="hint">PowerShell (documented default)</p>
            <pre class="cmd">.venv/bin/python tools/serve_dashboard.py</pre>
            <p class="hint">Linux or macOS, then open <code>127.0.0.1:8000</code> in a browser</p>
          </div>
        </aside>
      </div>`;
  };

  // ---------------------------------------------------------------- banners

  function showBanner(kind, html) {
    const banner = el('banner');
    banner.className = `banner ${kind}`;
    banner.innerHTML = `<div class="banner-text">${html}</div>
      <button class="btn-quiet" type="button" data-action="dismiss-banner">Dismiss</button>`;
    banner.hidden = false;
  }

  function clearBanner() {
    const banner = el('banner');
    banner.hidden = true;
    banner.innerHTML = '';
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
