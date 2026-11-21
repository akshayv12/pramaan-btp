import { TERMINAL, safeUrl, domainOf, readableTitle, citationParts, formatSeconds, runLabel, copyText, generationLabel, verificationRows } from './ui.js';

const $ = id => document.getElementById(id);
const state = { health: null, job: null, answer: null, currentId: null, poll: null, clock: null, toast: null, sources: new Map(), inputMode: 'live' };

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers }, signal: controller.signal });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The backend could not complete this request.');
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The backend took too long to respond. Check that it is still running.');
    if (error instanceof TypeError) throw new Error('The backend is unreachable. Start the local server and try again.');
    throw error;
  } finally { clearTimeout(timer); }
}

function toast(message) {
  clearTimeout(state.toast);
  $('toast').textContent = message;
  $('toast').hidden = false;
  state.toast = setTimeout(() => { $('toast').hidden = true; }, 4200);
}

function busy(value) {
  $('search-form').setAttribute('aria-busy', String(value));
  $('search-submit').disabled = value;
  $('example-button').disabled = value;
  $('upload-button').disabled = value;
  $('submit-label').textContent = value ? 'Researching' : 'Search';
  document.body.classList.toggle('is-researching', value);
}

function preferences() {
  return { language: $('language').value || null, ...(state.inputMode === 'live' ? {ranker: $('ranker').value} : {}), provider: $('provider').value, verification: $('verification').value, judge_provider:$('judge-provider').value };
}

function updateModeLabel() {
  if (!state.health) return;
  const ranking = state.inputMode === 'live' ? `Task 4: ${$('ranker').selectedOptions[0].textContent}` : 'Task 4 ranking supplied';
  const provider = $('provider').value === 'auto' ? state.health.generation_label : $('provider').selectedOptions[0].textContent;
  $('mode-label').textContent = `${ranking} · ${provider}`;
  $('verification').disabled = $('provider').value === 'extractive';
  $('judge-provider').disabled = $('provider').value === 'extractive';
}

function setInputMode(mode) {
  state.inputMode = mode;
  $('ranker-control').hidden = mode !== 'live';
  $('ranker').disabled = mode !== 'live';
  $('supplied-ranking-note').hidden = mode === 'live';
  updateModeLabel();
}

function applyHealth(health) {
  state.health = health;
  $('ranker').value = health.default_ranker;
  $('ranker').querySelector('[value="production"]').disabled = !health.semantic_ranking;
  $('provider').value = health.default_provider;
  $('provider').querySelector('[value="gemini"]').disabled = !health.gemini;
  $('provider').querySelector('[value="ollama"]').disabled = !health.ollama;
  $('verification').value = health.default_verification || 'standard';
  $('judge-provider').querySelector('[value="gemini"]').disabled = !health.gemini;
  $('judge-provider').value = health.default_judge_provider || 'same';
  $('gemini-model').value = health.gemini_model || '';
  $('options-note').textContent = `${health.notice || ''} ${health.nli ? 'Independent NLI verifier is installed.' : 'Independent NLI verifier is unavailable; model review remains available.'}`;
  $('example-button').disabled = !health.saved_example;
  updateModeLabel();
}

function showError(message, title = 'This search could not finish', job = null) {
  $('error-title').textContent = title;
  $('error-message').textContent = message;
  $('error-section').hidden = false;
  $('error-log').hidden = !job || job.status === 'cancelled';
  if (job) $('error-log').href = `/api/jobs/${job.id}/log`;
}

function stopPolling() {
  clearTimeout(state.poll);
  clearInterval(state.clock);
  state.poll = state.clock = null;
}

function updateElapsed() {
  if (!state.job) return;
  const date = Date.parse(state.job.started_at || state.job.created_at);
  $('elapsed').textContent = formatSeconds((Date.now() - date) / 1000);
}

function renderStages(job) {
  $('stage-list').replaceChildren();
  for (const stage of job.stages) {
    const item = document.createElement('div');
    item.className = `stage-item ${stage.state}`;
    const marker = document.createElement('span');
    marker.className = 'stage-marker';
    marker.setAttribute('aria-hidden', 'true');
    marker.textContent = ['completed', 'supplied'].includes(stage.state) ? '✓' : stage.state === 'running' ? '' : '·';
    const label = document.createElement('span');
    label.textContent = stage.label;
    item.append(marker, label);
    $('stage-list').append(item);
  }
  const running = job.stages.find(stage => stage.state === 'running');
  $('progress-note').textContent = job.status === 'queued' ? 'Your question is queued for research.' : running ? `${running.label}… This may take a moment while sources are collected and read.` : 'Finishing the research result…';
}

function sourceCard(source, index) {
  const card = document.createElement('article');
  card.className = 'source-card';
  card.id = `source-passage-${index}`;
  const topline = document.createElement('div');
  topline.className = 'source-topline';
  const id = document.createElement('span');
  id.className = 'source-id'; id.textContent = source.id;
  const domain = document.createElement('span');
  domain.className = 'source-domain'; domain.textContent = domainOf(source.url);
  topline.append(id, domain);
  const url = safeUrl(source.url);
  const title = document.createElement(url ? 'a' : 'p');
  title.className = 'source-title';
  title.textContent = readableTitle(source.title || domainOf(source.url) || `Source ${source.id}`);
  if (url) { title.href = url; title.target = '_blank'; title.rel = 'noopener noreferrer'; }
  const detail = document.createElement('p'); detail.className = 'source-detail';
  const extraction = /ocr/i.test(source.extraction_method || '') ? 'OCR text' : 'Digital text';
  const metadata = [source.page ? `Page ${source.page}` : '', extraction];
  if (/ocr/i.test(source.extraction_method || '')) metadata.push(source.ocr_confidence == null ? 'OCR confidence unknown' : `OCR confidence ${Number(source.ocr_confidence).toFixed(2)}`);
  detail.textContent = metadata.filter(Boolean).join(' · ');
  const details = document.createElement('details');
  const summary = document.createElement('summary'); summary.textContent = 'View supporting passage';
  details.append(summary);
  for (const quote of source.quotes || []) {
    const block = document.createElement('blockquote'); block.textContent = quote; details.append(block);
  }
  card.append(topline, title, detail, details);
  state.sources.set(source.id, { card, details });
  return card;
}

function focusSource(id) {
  for (const source of state.sources.values()) source.card.classList.remove('is-highlighted');
  const source = state.sources.get(id);
  if (!source) return;
  source.details.open = true;
  source.card.classList.add('is-highlighted');
  source.card.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'nearest' });
}

function renderAnswer(answer) {
  state.answer = answer;
  // Show the resolved language, including a query instruction overriding a
  // previous selector or a different item in an uploaded question batch.
  if (['hi','en'].includes(answer.language)) $('language').value = answer.language;
  $('result-question').textContent = answer.query;
  $('answer-text').lang = answer.language || 'en';
  $('sources-list').replaceChildren();
  state.sources.clear();
  const sources = answer.sources || [];
  sources.forEach((source, i) => $('sources-list').append(sourceCard(source, i)));
  $('source-count').textContent = `${sources.length} passage${sources.length === 1 ? '' : 's'}`;
  if (!sources.length) {
    const message = document.createElement('p'); message.className = 'empty-sources';
    message.textContent = 'No usable supporting passages were found. Try a more specific question or review the collected evidence.';
    $('sources-list').append(message);
  }
  $('answer-text').replaceChildren();
  for (const part of citationParts(answer.answer)) {
    if (part.type === 'text') { $('answer-text').append(document.createTextNode(part.text)); continue; }
    for (const id of part.ids) {
      if (!state.sources.has(id)) { $('answer-text').append(document.createTextNode(`[${id}]`)); continue; }
      const citation = document.createElement('button'); citation.className = 'citation'; citation.type = 'button';
      citation.textContent = id; citation.setAttribute('aria-label', `Inspect supporting passage ${id}`);
      citation.addEventListener('click', () => focusSource(id)); $('answer-text').append(citation);
    }
  }
  const rating = String(answer.reliability || 'LOW').toLowerCase();
  $('reliability-badge').className = `reliability-badge ${['high', 'medium', 'low'].includes(rating) ? rating : 'low'}`;
  $('reliability-badge').textContent = `${rating.charAt(0).toUpperCase() + rating.slice(1)} support`;
  $('why-text').textContent = answer.why;
  $('answer-status').hidden = !['limited', 'insufficient_evidence'].includes(answer.answer_status);
  $('answer-status').textContent = answer.answer_status === 'limited' ? 'Limited answer: the available excerpts may not cover the full question.' : 'Insufficient evidence: the system could not support a factual answer.';
  const mode = answer.generation_mode;
  $('generation-label').textContent = generationLabel(answer);
  $('generation-label').title=answer.model || '';
  const warnings = [...new Set((answer.warnings || []).filter(w => typeof w === 'string'))];
  if (answer.is_simulated) warnings.unshift('This evidence is a constructed test fixture, not a live factual evaluation.');
  $('warnings-list').replaceChildren();
  for (const warning of warnings) { const item = document.createElement('li'); item.textContent = warning; $('warnings-list').append(item); }
  $('limitations').hidden = !warnings.length;
  $('verification-details').hidden = !answer.verification && !answer.factors?.semantic_review;
  $('verification-list').replaceChildren();
  for (const [label,value] of verificationRows(answer)) {
    const term = document.createElement('dt'); term.textContent = label;
    const description = document.createElement('dd'); description.textContent = value;
    $('verification-list').append(term,description);
  }
  $('verification-explanation').textContent = answer.verification?.criteria?.explanation || '';
}

function renderJob(job) {
  const newlyOpened = state.job?.id !== job.id;
  state.job = job;
  if (newlyOpened) {
    const provider = $('provider').querySelector(`option[value="${job.request.provider}"]`);
    if (provider && !provider.disabled) $('provider').value = job.request.provider;
    $('language').value = job.request.language || '';
    $('verification').value = job.request.verification || 'standard';
    $('judge-provider').value = job.request.judge_provider || 'same';
    if (job.request.ranker && job.request.ranker !== 'supplied') $('ranker').value = job.request.ranker;
  }
  setInputMode(job.request.mode);
  const terminal = TERMINAL.has(job.status);
  busy(!terminal);
  $('error-section').hidden = true;
  $('progress-section').hidden = terminal;
  renderStages(job);
  if (!terminal) {
    $('result-section').hidden = true;
    document.body.classList.remove('has-results');
    $('progress-title').textContent = job.request.mode === 'live' ? 'Following the evidence' : 'Reading the supplied evidence';
    updateElapsed();
    if (!state.clock) state.clock = setInterval(updateElapsed, 1000);
    return;
  }
  clearInterval(state.clock); state.clock = null;
  if (['completed', 'partial'].includes(job.status) && job.answers?.length) {
    document.body.classList.add('has-results');
    $('result-section').hidden = false;
    $('previous-result-notice').hidden = job.result_is_current !== false;
    const trace = job.pipeline_trace;
    $('pipeline-trace').hidden = !trace;
    $('pipeline-trace-list').replaceChildren();
    if (trace) {
      const fields = [['Query and passage IDs preserved', trace.handoff_verified ? 'Verified' : 'Could not verify'], ['Documents read', trace.documents_processed], ['Candidate passages', trace.candidate_passages], ['Passages received by Task 5', trace.passages_received_by_task5], ['Passages selected by Task 5', trace.passages_selected_by_task5], ['Accepted answer statements', trace.accepted_claims], ['Completed model calls', trace.completed_model_calls], ['Completed support reviews', trace.model_reviews_completed], ['Model', trace.models?.join(', ') || null], ['Answer implementation', trace.implementation_version]];
      for (const [label,value] of fields) {
        if (value == null) continue;
        const term=document.createElement('dt'); term.textContent=label;
        const description=document.createElement('dd'); description.textContent=String(value);
        $('pipeline-trace-list').append(term,description);
      }
    }
    $('result-mode').textContent = runLabel(job.request.mode);
    $('result-meta').textContent = `${formatSeconds(job.duration_seconds)}${job.status === 'partial' ? ' · Some documents could not be read' : ''}`;
    $('answer-selector-wrap').hidden = job.answers.length < 2;
    $('answer-selector').replaceChildren();
    job.answers.forEach((answer, index) => {
      const option = document.createElement('option'); option.value = String(index); option.textContent = answer.query;
      $('answer-selector').append(option);
    });
    renderAnswer(job.answers[0]);
    $('download-answer').href = `/api/jobs/${job.id}/text`;
    $('download-json').href = `/api/jobs/${job.id}/json`;
  } else {
    $('result-section').hidden = true;
    showError(job.error || (job.status === 'cancelled' ? 'You can edit your question and search again.' : 'The backend stopped before this search finished.'), job.status === 'cancelled' ? 'Search cancelled' : 'This search could not finish', job);
  }
}

async function pollJob(id) {
  try {
    const job = await api(`/api/jobs/${id}`);
    if (state.currentId !== id) return;
    renderJob(job);
    if (!TERMINAL.has(job.status)) state.poll = setTimeout(() => pollJob(id), 1000);
  } catch (error) {
    if (state.currentId !== id) return;
    busy(false); stopPolling();
    showError(`${error.message} Your search may still be running; open Recent searches to reconnect.`);
  }
}

async function submit(mode, payload = {}) {
  if ($('search-submit').disabled) return;
  if (!state.health) { toast('The backend is still connecting. Try again in a moment.'); return; }
  setInputMode(mode);
  $('error-section').hidden = true;
  stopPolling(); state.currentId = null; busy(true);
  $('result-section').hidden = true;
  try {
    const path = mode === 'live' ? '/api/search' : mode === 'example' ? '/api/example' : '/api/task4';
    const job = await api(path, { method: 'POST', body: JSON.stringify({ ...preferences(), ...payload }) });
    state.currentId = job.id;
    $('query').value = job.request.query;
    const url = new URL(location.href); url.searchParams.set('run', job.id); history.replaceState({}, '', url);
    renderJob(job); await pollJob(job.id);
  } catch (error) { busy(false); showError(error.message); }
}

async function openHistory() {
  $('history-dialog').showModal();
  $('history-list').replaceChildren();
  const loading = document.createElement('p'); loading.className = 'history-empty'; loading.textContent = 'Loading searches…'; $('history-list').append(loading);
  try {
    const data = await api('/api/jobs'); $('history-list').replaceChildren();
    if (!data.jobs.length) { loading.textContent = 'Your research will appear here after your first search.'; $('history-list').append(loading); }
    for (const job of data.jobs) {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'history-item';
      const title = document.createElement('strong'); title.textContent = job.request.query;
      const meta = document.createElement('span');
      meta.textContent = `${runLabel(job.request.mode)} · ${job.status} · ${new Date(job.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}`;
      button.append(title, meta);
      button.addEventListener('click', () => {
        $('history-dialog').close(); stopPolling(); state.currentId = job.id; $('query').value = job.request.query;
        const url = new URL(location.href); url.searchParams.set('run', job.id); history.replaceState({}, '', url);
        pollJob(job.id);
      }); $('history-list').append(button);
    }
  } catch (error) { loading.textContent = error.message; $('history-list').replaceChildren(loading); }
}

$('search-form').addEventListener('submit', event => {
  event.preventDefault(); const query = $('query').value.trim();
  if (query.length < 2 || query.length > 500) { toast('Enter a question between 2 and 500 characters.'); $('query').focus(); return; }
  submit('live', { query });
});
$('query').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('search-form').requestSubmit(); } });
for (const button of document.querySelectorAll('[data-suggestion]')) button.addEventListener('click', () => { $('query').value = button.dataset.suggestion; $('query').focus(); });
$('example-button').addEventListener('click', () => submit('example'));
$('upload-button').addEventListener('click', () => $('task4-file').click());
$('task4-file').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  try { if (file.size > 8000000) throw new Error('Choose a Task 4 JSON file smaller than 8 MB.'); await submit('task4', { payload: JSON.parse(await file.text()) }); }
  catch (error) { toast(error instanceof SyntaxError ? 'That file is not valid JSON.' : error.message); }
  finally { event.target.value = ''; }
});
$('options-toggle').addEventListener('click', () => { const expanded = $('options-toggle').getAttribute('aria-expanded') === 'true'; $('options-toggle').setAttribute('aria-expanded', String(!expanded)); $('search-options').hidden = expanded; });
for (const id of ['ranker','provider']) $(id).addEventListener('change',updateModeLabel);
$('gemini-settings-button').addEventListener('click', () => { $('gemini-status').textContent = state.health?.gemini_message || ''; $('gemini-dialog').showModal(); });
$('gemini-dialog').addEventListener('close', () => { $('gemini-key').value = ''; });
$('gemini-form').addEventListener('submit', async event => {
  event.preventDefault(); $('gemini-save').disabled = true; $('gemini-status').textContent = 'Checking model access…';
  try {
    const response = await api('/api/gemini/configure', {method:'POST',body:JSON.stringify({api_key:$('gemini-key').value,model:$('gemini-model').value.trim()})});
    $('gemini-key').value = ''; applyHealth(response.health); $('provider').value = 'gemini'; updateModeLabel();
    $('gemini-status').textContent = response.message; toast('Gemini synthesis is selected.');
  } catch (error) { $('gemini-status').textContent = error.message; }
  finally { $('gemini-save').disabled = false; }
});
$('how-button').addEventListener('click', () => $('how-dialog').showModal());
$('history-button').addEventListener('click', openHistory);
for (const button of document.querySelectorAll('[data-close]')) button.addEventListener('click', () => $(button.dataset.close).close());
for (const dialog of document.querySelectorAll('dialog')) dialog.addEventListener('click', event => { if (event.target === dialog) { const bounds = dialog.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close(); } });
$('cancel-button').addEventListener('click', async () => { if (!state.currentId) return; try { const job = await api(`/api/jobs/${state.currentId}/cancel`, { method: 'POST', body: '{}' }); stopPolling(); renderJob(job); } catch (error) { toast(error.message); } });
$('new-search-button').addEventListener('click', () => { $('query').value = ''; $('query').focus(); window.scrollTo({ top: 0, behavior: 'smooth' }); });
$('answer-selector').addEventListener('change', () => {
  const answer = state.job?.answers?.[Number($('answer-selector').value)];
  if (answer) renderAnswer(answer);
});
$('copy-button').addEventListener('click', async () => { if (!state.answer) return; try { await navigator.clipboard.writeText(copyText(state.answer)); toast('Answer and source references copied.'); } catch { toast('Copy is unavailable in this browser. Use Download answer instead.'); } });

async function initialize() {
  try {
    const health = await api('/api/health'); applyHealth(health);
    $('backend-status').classList.add('connected'); $('backend-label').textContent = 'Backend connected';
    const id = new URLSearchParams(location.search).get('run');
    if (id && /^[a-f0-9]{32}$/.test(id)) { state.currentId = id; await pollJob(id); if (state.job) $('query').value = state.job.request.query; }
  } catch (error) {
    $('backend-status').classList.add('offline'); $('backend-label').textContent = 'Backend offline';
    $('mode-label').textContent = 'Start the local Python backend to search';
    showError(error.message, 'Connect to your research backend');
  } finally {
    // Saved-query restoration must finish before the user can edit. Otherwise
    // its late response can overwrite (or append to) a freshly typed question.
    $('query').disabled = false;
    if (!state.job || TERMINAL.has(state.job.status)) $('search-submit').disabled = false;
  }
}
initialize();
