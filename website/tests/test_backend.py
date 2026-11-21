"""Contract and lifecycle tests for the website's pipeline adapter."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from website.backend import (ROOT, TERMINAL, JobStore, atomic_json, command_for, default_python, capabilities,
                             public_answers, read_json, validate_request, pipeline_trace, pipeline_environment)

READY = {"connected": True, "live_search": True, "gemini": False,
         "semantic_ranking": False, "saved_example": True,
         "default_ranker": "lexical", "default_provider": "extractive"}
SAMPLE = ROOT / "task5_standalone/examples/supplied_task4_output.json"


def wait_for(store, identity, statuses=TERMINAL, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = store.snapshot(identity, details=True)
        if value["status"] in statuses:
            return value
        time.sleep(.02)
    raise AssertionError("Job did not reach expected state: " + repr(value))


class RequestTests(unittest.TestCase):
    def test_fresh_install_defaults_to_available_source_extraction(self):
        local={'enabled':True,'ready':False,'model':'qwen3:4b-instruct-2507-q4_K_M','message':'Not downloaded'}
        gemini={'ready':False,'configured':False,'model':'unconfigured','message':'Configure Gemini'}
        with patch('website.backend.subprocess.run') as probe,patch('website.backend.local_status',return_value=local),patch('website.backend.shutil.which',return_value=None):
            probe.return_value.returncode=0;probe.return_value.stdout='{}'
            self.assertEqual(capabilities(sys.executable,gemini)['default_provider'],'extractive')

    def test_pipeline_interpreter_uses_windows_venv_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);python=root/'.integration-venv/Scripts/python.exe'
            python.parent.mkdir(parents=True);python.touch()
            with patch('website.backend.ROOT',root),patch('website.backend.os.name','nt'),patch.dict('os.environ',{},clear=True):
                self.assertEqual(default_python(),str(python))

    def test_explicit_pipeline_interpreter_takes_priority(self):
        with patch.dict('os.environ',{'PIPELINE_PYTHON':'/chosen/python'}):
            self.assertEqual(default_python(),'/chosen/python')

    def test_explicit_query_language_overrides_selector_and_reaches_cli(self):
        request=validate_request({'query':'tell me 5 points on ramayana in hindi','language':'en'},READY)
        self.assertEqual(request['language'],'hi')
        command=command_for(request,Path('/tmp/language-test'),sys.executable)
        self.assertEqual(command[command.index('--language')+1],'hi')

    def test_explicit_english_overrides_a_previous_hindi_selector(self):
        request=validate_request({'query':'रामायण के बारे में बताइए, answer in English','language':'hi'},READY)
        self.assertEqual(request['language'],'en')

    def test_uploaded_single_question_resolves_its_requested_language(self):
        payload=copy.deepcopy(read_json(SAMPLE))
        payload['task4_output']['queries'][0]['query']='Tell me about Rama in Hindi'
        request=validate_request({'payload':payload,'language':'en'},READY,'task4')
        self.assertEqual(request['language'],'hi')

    def test_review_provider_is_explicit_and_reaches_child_environment(self):
        ready={**READY,'gemini':True,'default_provider':'gemini','default_judge_provider':'gemini'}
        request=validate_request({'query':'What is Sundara Kanda?'},ready)
        self.assertEqual(request['judge_provider'],'gemini')
        self.assertEqual(pipeline_environment(request)['TASK5_JUDGE_PROVIDER'],'gemini')
        local=validate_request({'query':'A question','judge_provider':'same'},ready)
        self.assertEqual(local['judge_provider'],'same')

    def test_unavailable_gemini_review_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Configure Gemini'):
            validate_request({'query':'A question','judge_provider':'gemini'},READY)

    def test_evidence_selection_does_not_replace_the_reported_generation_model(self):
        result={'answer':'Answer [S1]','target_language':'en','reliability':'HIGH','reliability_explanation':'Supported',
                'cited_sources':['S1'],'generation_mode':'ollama','diagnostics':{'model_calls':[
                    {'status':'completed','purpose':'evidence_selection','resolved_model':'gemini-3.5-flash-lite'},
                    {'status':'completed','purpose':'draft','resolved_model':'qwen3:4b-instruct-2507-q4_K_M'}]}}
        value=public_answers({'task5_output':{'queries':[{'query':'A question','result':result}]}})
        self.assertEqual(value[0]['model'],'qwen3:4b-instruct-2507-q4_K_M')

    def test_task_local_ocr_models_reach_child_processes_without_replacing_user_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'website/.ocr/tessdata';data.mkdir(parents=True)
            for name in ('eng','hin'):(data/(name+'.traineddata')).write_bytes(b'test model')
            with patch('website.backend.ROOT',root),patch.dict('os.environ',{},clear=True):
                self.assertEqual(pipeline_environment()['TESSDATA_PREFIX'],str(data))
            with patch('website.backend.ROOT',root),patch.dict('os.environ',{'TESSDATA_PREFIX':'/chosen/models'},clear=True):
                self.assertEqual(pipeline_environment()['TESSDATA_PREFIX'],'/chosen/models')

    def test_literal_query_is_not_a_shell_or_cli_option(self):
        query = "--provider=gemini $(echo unsafe); हिन्दी"
        request = validate_request({"query": query}, READY)
        command = command_for(request, Path("/tmp/example job"), sys.executable)
        self.assertIn("--query=" + query, command)
        self.assertEqual(command.count("--provider"), 1)
        self.assertIn("--lexical-only", command)
        self.assertEqual(command[command.index("--task4-config") + 1], str(ROOT / "website/configs/lexical.json"))

    def test_unavailable_model_is_not_silently_selected(self):
        for preferences in ({"provider": "gemini"}, {"ranker": "production"}):
            with self.subTest(preferences=preferences), self.assertRaises(ValueError):
                validate_request({"query": "Indian agriculture", **preferences}, READY)

    def test_invalid_request_values(self):
        for value in ([], {}, {"query": "a"}, {"query": "x" * 501},
                      {"query": 42}, {"query": "hello", "language": "fr"},
                      {"query": "hello", "provider": "other"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_request(value, READY)

    def test_upload_uses_existing_task4_contract(self):
        value = validate_request({"payload": read_json(SAMPLE)}, READY, "task4")
        self.assertTrue(value["query"])
        self.assertNotIn("payload", value)
        malformed = copy.deepcopy(read_json(SAMPLE))
        malformed["task4_output"]["queries"][0]["traceability"]["selected_evidence_ids"] = []
        with self.assertRaises(ValueError):
            validate_request({"payload": malformed}, READY, "task4")

    def test_empty_or_excessive_batches_are_rejected(self):
        for queries in ([], [read_json(SAMPLE)["task4_output"]["queries"][0]] * 21):
            with self.assertRaises(ValueError):
                validate_request({"payload": {"task4_output": {"schema_version": "1.0", "queries": queries}}}, READY, "task4")

    def test_missing_runtime_is_reported(self):
        with self.assertRaisesRegex(ValueError, "environment is incomplete"):
            validate_request({"query": "Indian agriculture"}, {**READY, "live_search": False})


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            store.close()
        self.temp.cleanup()

    def store(self, **kwargs):
        store = JobStore(directory=self.temp.name, python=sys.executable, ready=READY, **kwargs)
        self.stores.append(store)
        return store

    def test_real_task4_to_task5_and_downloads(self):
        """Exercise the unchanged CLI, rather than a canned backend result."""
        before = SAMPLE.read_bytes()
        store = self.store()
        job = wait_for(store, store.submit({}, "example")["id"])
        self.assertEqual(job["status"], "completed", job.get("error"))
        answer = job["answers"][0]
        self.assertEqual(answer["reliability"], "LOW")
        self.assertEqual(answer["language"], "hi")
        self.assertTrue(answer["sources"])
        self.assertIn("[S", answer["answer"])
        self.assertEqual([s["state"] for s in job["stages"]], ["supplied"] * 4 + ["completed"])
        raw, mime, _ = store.download(job["id"], "json")
        exported = json.loads(raw)
        self.assertEqual(public_answers(exported), job["answers"])
        self.assertIn("application/json", mime)
        text, _, _ = store.download(job["id"], "text")
        self.assertIn("Reliability: LOW", text.decode())
        self.assertIn(answer["sources"][0]["url"], text.decode())
        self.assertEqual(SAMPLE.read_bytes(), before)
        self.assertEqual(store.history()[0]["id"], job["id"])
        with self.assertRaises(KeyError):
            store.download(job["id"], "../../input.json")

    def test_multiple_uploaded_questions_survive_the_adapter(self):
        payload = read_json(SAMPLE)
        second = copy.deepcopy(payload["task4_output"]["queries"][0])
        second["query"] = "राम पर उपलब्ध साक्ष्य क्या बताते हैं?"
        payload["task4_output"]["queries"].append(second)
        store = self.store()
        job = wait_for(store, store.submit({"payload": payload}, "task4")["id"])
        self.assertEqual(job["status"], "completed", job.get("error"))
        self.assertEqual(len(job["answers"]), 2)
        self.assertEqual(job["answers"][1]["query"], second["query"])

    def test_handoff_trace_detects_bad_mapping_and_flags_old_results(self):
        store=self.store()
        job=wait_for(store,store.submit({},'example')['id'])
        self.assertTrue(job['pipeline_trace']['handoff_verified'])
        self.assertGreater(job['pipeline_trace']['passages_received_by_task5'],0)
        self.assertGreater(job['pipeline_trace']['accepted_claims'],0)
        self.assertTrue(job['result_is_current'])
        folder=Path(self.temp.name)/job['id']
        payload=read_json(folder/'task5_output.json')
        payload['task5_output']['queries'][0]['citation_to_task4_passage_id']['S1']='wrong passage'
        self.assertFalse(pipeline_trace(folder,payload)['handoff_verified'])
        payload['task5_output']['implementation_version']='3.0'
        atomic_json(folder/'task5_output.json',payload)
        self.assertFalse(store.snapshot(job['id'],details=True)['result_is_current'])

    def test_failure_is_not_a_saved_answer_fallback(self):
        store = self.store(command_builder=lambda *_: [sys.executable, "-c", "raise SystemExit(2)"])
        job = wait_for(store, store.submit({"query": "live question"})["id"])
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["answers"], [])

    def test_model_verification_failure_is_explained_without_claiming_generation_succeeded(self):
        builder=lambda *_:[sys.executable,'-c','print("[INTEGRATION] ProviderError: The support review rejected every proposed answer statement"); raise SystemExit(2)']
        store=self.store(command_builder=builder)
        job=wait_for(store,store.submit({'query':'A live question'})['id'])
        self.assertEqual(job['status'],'failed');self.assertEqual(job['answers'],[])
        self.assertIn('Source extraction',job['error'])

    def test_cancel_and_timeout_terminate_pipeline_process(self):
        builder = lambda *_: [sys.executable, "-c", "import time; time.sleep(30)"]
        store = self.store(command_builder=builder, timeout=.12)
        cancelled = store.submit({"query": "cancel me"})
        store.cancel(cancelled["id"])
        self.assertEqual(wait_for(store, cancelled["id"])["status"], "cancelled")
        timed = wait_for(store, store.submit({"query": "timeout case"})["id"])
        self.assertEqual(timed["status"], "failed")
        self.assertIn("timed out", timed["error"])

    def test_restart_marks_unfinished_jobs_interrupted(self):
        identity = "a" * 32
        folder = Path(self.temp.name) / identity
        folder.mkdir()
        atomic_json(folder / "job.json", {"id": identity, "status": "running", "created_at": "2026-01-01", "request": {"mode": "live"}})
        store = self.store()
        self.assertEqual(store.snapshot(identity)["status"], "interrupted")
        self.assertEqual(read_json(folder / "job.json")["status"], "interrupted")

    def test_existing_pipeline_files_are_unchanged(self):
        # The public release omits historical reports/private run data from the
        # old workspace snapshot. Verify its explicitly published source scope.
        from scripts.check_release import errors
        self.assertEqual(errors(ROOT), [])


if __name__ == "__main__":
    unittest.main()
