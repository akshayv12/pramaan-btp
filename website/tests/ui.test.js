import test from 'node:test';
import assert from 'node:assert/strict';
import { safeUrl, readableTitle, citationParts, formatSeconds, runLabel, TERMINAL, copyText, generationLabel, verificationRows } from '../static/ui.js';

test('source URLs cannot execute script or carry credentials', () => {
  for (const url of ['javascript:alert(1)', 'data:text/html,test', 'file:///etc/passwd', 'https://user:secret@example.com', '/relative']) {
    assert.equal(safeUrl(url), null);
  }
  assert.equal(safeUrl('https://example.com/path?q=1'), 'https://example.com/path?q=1');
});

test('verification display distinguishes an accepted borderline claim from measured consistency and calibration', () => {
  const rows = new Map(verificationRows({verification:{status:'completed',nli:{status:'completed',accepted_claim_indices:[1],claims:[{claim_index:0,verdict:'contradicted'},{claim_index:1,verdict:'borderline'}]},criteria:{answer_relevance:'pass'},consistency:{status:'not_run'},retrieval:{status:'measured',flat_scores:true}}}));
  assert.equal(rows.get('Independent claim check'),'0 entailed; 1 borderline');
  assert.equal(rows.get('Sampling consistency'),'Not measured');
  assert.match(rows.get('Human calibration'),/Pending/);
  assert.match(rows.get('Ranking distinction'),/clustered/);
});

test('thorough mode exposes unstable statements without presenting a truth percentage', () => {
  const rows = new Map(verificationRows({verification:{consistency:{status:'completed',additional_drafts:1,accepted_claim_indices:[0,1],claims:[{claim_index:0,unstable:false},{claim_index:1,unstable:true}]}}}));
  assert.equal(rows.get('Sampling consistency'),'1 of 2 statements varied across 1 alternative draft(s)');
  assert.equal([...rows.values()].some(value => value.includes('%')),false);
});

test('low upstream scores are distinguished from a verified document-topic match', () => {
  const rows = new Map(verificationRows({verification:{retrieval:{status:'measured',low_relevance:true,document_topic_match:true}}}));
  assert.equal(rows.get('Retrieval relevance'),'Document topic matches; upstream scores are low');
});

test('judge relevance and raw retrieval scores remain separate', () => {
  const review = {status:'completed',criteria:{answer_relevance:'pass'},retrieval:{status:'measured',low_relevance:true,document_topic_match:false}};
  assert.equal(new Map(verificationRows({verification:review})).get('Retrieval relevance'),'Judge relevance passed; upstream scores are low');
  assert.equal(new Map(verificationRows({verification:{...review,criteria:{}}})).get('Retrieval relevance'),'Low upstream scores; relevance needs review');
});

test('LLM support and pilot validation are visible without claiming human calibration', () => {
  const rows = new Map(verificationRows({factors:{llm_support_assessment:{judge_proposed_label:'High',assessment:{grounding:'strong',question_coverage:'complete',source_quality:'adequate'},calibration_status:'model_judged_pilot',calibration:{query_count:30}}}}));
  assert.equal(rows.get('LLM support judgment'),'High');
  assert.equal(rows.get('Question coverage'),'complete');
  assert.equal(rows.get('Support policy validation'),'30 query pilot; human calibration pending');
});

test('Hindi entities remain readable without interpreting markup', () => {
  assert.equal(readableTitle('&#2352;&#x93e;&#2350; &amp; text'), 'राम & text');
  assert.equal(readableTitle('<script>text</script>'), '<script>text</script>');
  assert.equal(readableTitle('&#99999999;'), '&#99999999;');
});

test('citations preserve surrounding Hindi and English evidence text', () => {
  assert.deepEqual(citationParts('दावा। [S1, S2, S1] Next [S3].'), [
    { type: 'text', text: 'दावा। ' },
    { type: 'citation', ids: ['S1', 'S2'], raw: '[S1, S2, S1]' },
    { type: 'text', text: ' Next ' },
    { type: 'citation', ids: ['S3'], raw: '[S3]' },
    { type: 'text', text: '.' }
  ]);
});

test('saved and imported evidence are clearly distinguished from live search', () => {
  assert.equal(runLabel('example'), 'Saved research example');
  assert.equal(runLabel('task4'), 'Imported evidence');
  assert.equal(runLabel('live'), 'Live web research');
});

test('elapsed time and final states do not imply a fake completion percentage', () => {
  assert.equal(formatSeconds(62), '1m 2s');
  assert.equal(formatSeconds(-1), '0s');
  assert.equal(formatSeconds(undefined), '0s');
  assert.equal(TERMINAL.has('cancelled'), true);
  assert.equal(TERMINAL.has('running'), false);
});

test('copied report text includes traceable source links and support explanation', () => {
  const value = copyText({query:'Question', answer:'Answer [S1]', reliability:'LOW', why:'Limited evidence', sources:[{id:'S1',title:'Source &amp; title',url:'https://example.com/evidence'}]});
  assert.match(value, /Why: Limited evidence/);
  assert.match(value, /\[S1\] Source & title\nhttps:\/\/example.com\/evidence/);
});

test('actual generation mode determines the label even when a model call preceded a fallback', () => {
  assert.equal(generationLabel({generation_mode:'ollama',model:'qwen3:4b-instruct-2507-q4_K_M'}),'Local Qwen3 4B');
  assert.equal(generationLabel({generation_mode:'extractive_fallback',model:'qwen3:4b-instruct-2507-q4_K_M'}),'Extraction fallback');
  assert.equal(generationLabel({generation_mode:'ollama',model:'another-model'}),'Local LLM');
  assert.equal(generationLabel({generation_mode:'abstained',answer_status:'insufficient_evidence'}),'No usable evidence');
  assert.equal(generationLabel({generation_mode:'ollama',model:'qwen3:4b-instruct-2507-q4_K_M',verification:{judge_provider:'gemini'}}),'Qwen draft · Gemini review');
});
