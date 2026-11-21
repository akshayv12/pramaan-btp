export const TERMINAL = new Set(['completed', 'partial', 'failed', 'cancelled', 'interrupted']);

export function generationLabel(answer) {
  if (answer.answer_status === 'insufficient_evidence' || answer.generation_mode === 'abstained') return 'No usable evidence';
  if (answer.generation_mode === 'extractive_fallback') return 'Extraction fallback';
  if (answer.generation_mode === 'ollama') {
    if (answer.verification?.judge_provider === 'gemini') return 'Qwen draft · Gemini review';
    return answer.model?.startsWith('qwen3:4b') ? 'Local Qwen3 4B' : 'Local LLM';
  }
  return answer.generation_mode === 'gemini' ? 'Gemini synthesis' : 'Source extraction';
}

export function verificationRows(answer) {
  const review = answer.verification || answer.factors?.semantic_review || {};
  const nli = review.nli || {};
  const accepted = new Set(nli.accepted_claim_indices || []);
  const checked = (nli.claims || []).filter(row => accepted.has(row.claim_index));
  const consistency = review.consistency || {};
  const repeated = (consistency.claims || []).filter(row => (consistency.accepted_claim_indices || []).includes(row.claim_index));
  const criteria = review.criteria || {};
  const retrieval = review.retrieval || {};
  const support = answer.factors?.llm_support_assessment;
  const planning = review.evidence_planning || {};
  return [
    ...(support ? [['Support assessment basis','LLM rubric with citation, contradiction and OCR guards'],
      ['LLM support judgment',support.judge_proposed_label],['Claim grounding',support.assessment.grounding],
      ['Question coverage',support.assessment.question_coverage],['Source quality judgment',support.assessment.source_quality],
      ['Support policy validation',support.calibration_status === 'model_judged_pilot' ? `${support.calibration.query_count} query pilot; human calibration pending` : 'Prototype; human calibration pending']] : []),
    ...(planning.status === 'completed' ? [['Question-focused evidence',`${planning.selected_passage_ids.length} selected from ${planning.candidates_shown} Task 4 passages`]] : []),
    ['Independent claim check', nli.status === 'completed' ? `${checked.filter(row => row.verdict === 'entailed').length} entailed; ${checked.filter(row => row.verdict === 'borderline').length} borderline` : nli.status === 'failed' ? 'Verifier failed' : 'Not run / unavailable'],
    ['Criteria-based judge', review.status === 'completed' ? (Object.keys(criteria).length ? 'Completed' : 'Basic support review only') : 'Not run'],
    ['Judge model', review.judge_model || answer.model || 'Not applicable'],
    ['Sampling consistency', consistency.status === 'completed' ? `${repeated.filter(row => row.unstable).length} of ${repeated.length} statements varied across ${consistency.additional_drafts} alternative draft(s)` : 'Not measured'],
    ['Human calibration', nli.thresholds?.calibration_status === 'reviewed_holdout_evaluated' ? 'Reviewed labels and holdout evaluation' : 'Pending; support labels remain heuristic'],
    ['Retrieval relevance', retrieval.low_relevance ? (retrieval.document_topic_match ? 'Document topic matches; upstream scores are low' : criteria.answer_relevance === 'pass' ? 'Judge relevance passed; upstream scores are low' : 'Low upstream scores; relevance needs review') : retrieval.status === 'measured' ? 'Measured separately from generation' : 'Not measured'],
    ['Ranking distinction', retrieval.flat_scores ? 'Leading scores closely clustered' : retrieval.status === 'measured' ? 'No flat-score flag' : 'Not measured'],
    ...['answer_relevance', 'evidence_coverage', 'language_fidelity', 'readability'].filter(key => criteria[key]).map(key => [key.replaceAll('_', ' '), criteria[key]])
  ];
}

export function safeUrl(value) {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

export function domainOf(value) {
  try { return new URL(value).hostname.replace(/^www\./, ''); } catch { return 'Source provenance'; }
}

export function readableTitle(value) {
  return String(value || 'Source passage').replace(/&#(x[0-9a-f]+|[0-9]+);/gi, (original, number) => {
    const code = number.toLowerCase().startsWith('x') ? parseInt(number.slice(1), 16) : parseInt(number, 10);
    return code >= 0 && code <= 0x10ffff ? String.fromCodePoint(code) : original;
  }).replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&apos;/g, "'");
}

export function citationParts(text) {
  const pattern = /\[([A-Za-z0-9_.:-]+(?:\s*,\s*[A-Za-z0-9_.:-]+)*)\]/g;
  const result = []; let position = 0;
  for (const match of String(text).matchAll(pattern)) {
    if (match.index > position) result.push({ type: 'text', text: text.slice(position, match.index) });
    result.push({ type: 'citation', ids: [...new Set(match[1].split(',').map(id => id.trim()))], raw: match[0] });
    position = match.index + match[0].length;
  }
  if (position < text.length) result.push({ type: 'text', text: text.slice(position) });
  return result;
}

export function formatSeconds(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  return value < 60 ? `${value}s` : `${Math.floor(value / 60)}m ${value % 60}s`;
}

export function runLabel(mode) {
  return mode === 'example' ? 'Saved research example' : mode === 'task4' ? 'Imported evidence' : 'Live web research';
}

export function copyText(answer) {
  const sources = (answer.sources || []).map(source => {
    const url = safeUrl(source.url);
    return `[${source.id}] ${readableTitle(source.title)}${url ? '\n' + url : ''}`;
  });
  return `${answer.query}\n\n${answer.answer}\n\nReliability: ${answer.reliability}\nWhy: ${answer.why}\n\nSupporting sources:\n${sources.join('\n\n') || 'None'}`;
}
