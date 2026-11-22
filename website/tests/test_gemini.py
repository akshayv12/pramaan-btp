"""Gemini configuration, secret handling and Task 4 import behavior."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from website.gemini import probe
from website.backend import ROOT,JobStore,validate_request,command_for,pipeline_environment
from test_backend import READY

MODEL='gemini-test-flash'


class GeminiTests(unittest.TestCase):
    def test_persistent_configuration_uses_one_shared_ignored_environment(self):
        from website import configure_gemini
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);environment=root/'.env'
            environment.write_text('BRAVE_API_KEY=test-brave-key\n')
            state={'ready':True,'model':MODEL}
            output=io.StringIO()
            with patch.object(configure_gemini,'ROOT',root),patch.object(configure_gemini,'getpass',return_value='test-gemini-secret'),patch.object(configure_gemini,'probe',return_value=state),patch('sys.argv',['configure_gemini.py','--model',MODEL]),patch('sys.stdout',output):
                self.assertEqual(configure_gemini.main(),0)
            settings=environment.read_text()
            self.assertIn('BRAVE_API_KEY=test-brave-key',settings)
            self.assertIn('GEMINI_API_KEY=test-gemini-secret',settings)
            self.assertEqual(list(root.iterdir()),[environment])
            self.assertNotIn('test-gemini-secret',output.getvalue())
            if os.name=='posix':self.assertEqual(environment.stat().st_mode & 0o777,0o600)

    def test_missing_key_is_explicit_and_never_calls_google(self):
        with patch.dict(os.environ,{},clear=True),patch('website.gemini.urlopen') as request:
            value=probe(MODEL)
        self.assertFalse(value['ready']);self.assertFalse(value['configured']);request.assert_not_called()

    def test_readiness_uses_a_header_and_public_state_never_contains_the_key(self):
        response=io.BytesIO(json.dumps({'supportedGenerationMethods':['generateContent']}).encode())
        with patch('website.gemini.urlopen',return_value=response) as request:value=probe(MODEL,'unit-test-secret')
        self.assertTrue(value['ready']);self.assertNotIn('unit-test-secret',json.dumps(value))
        sent=request.call_args.args[0]
        self.assertNotIn('unit-test-secret',sent.full_url);self.assertIn('unit-test-secret',sent.headers.values())

    def test_model_ids_cannot_redirect_a_key_to_another_host(self):
        with self.assertRaises(ValueError):probe('https://untrusted.example/model','unit-test-secret')

    def test_quota_errors_are_actionable_without_dumping_provider_responses(self):
        error=HTTPError('https://generativelanguage.googleapis.com',429,'quota',{},None)
        with patch('website.gemini.urlopen',side_effect=error):value=probe(MODEL,'unit-test-secret')
        self.assertFalse(value['ready']);self.assertIn('quota',value['message']);self.assertNotIn('unit-test-secret',json.dumps(value))

    def test_auto_prefers_gemini_when_both_providers_are_ready(self):
        ready={**READY,'gemini':True,'gemini_model':MODEL,'ollama':True,'ollama_model':'local-test'}
        value=validate_request({'query':'What is Mahabharata?','provider':'auto'},ready)
        self.assertEqual(value['provider'],'gemini');self.assertEqual(value['model'],MODEL)
        command=command_for(value,Path('/tmp/test-job'),'python')
        self.assertEqual(command[command.index('--model')+1],MODEL);self.assertIn('--strict',command)

    def test_imported_task4_evidence_ignores_upstream_ranking_options(self):
        payload=json.loads((ROOT/'task5/examples/supplied_task4_output.json').read_text())
        request=validate_request({'payload':payload,'ranker':'production'},READY,'task4')
        self.assertEqual(request['ranker'],'supplied')
        command=command_for(request,Path('/tmp/test-job'),'python')
        self.assertIn('--task4-input',command);self.assertNotIn('--lexical-only',command);self.assertNotIn('--task4-config',command)

    def test_verification_choice_reaches_task5_without_changing_task4(self):
        request=validate_request({'query':'What is Mahabharata?','verification':'thorough'},READY)
        self.assertEqual(pipeline_environment(request)['TASK5_VERIFICATION'],'thorough')
        with self.assertRaises(ValueError):validate_request({'query':'A question','verification':'invented'},READY)

    def test_session_configuration_keeps_the_key_out_of_public_state_and_files(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{},clear=True),patch('task5.config.load_env_file'):
            store=JobStore(directory=directory,ready=dict(READY))
            state={'configured':True,'ready':True,'model':MODEL,'message':'Ready'}
            with patch('website.backend.gemini_probe',return_value=state),patch('website.backend.capabilities',return_value={**READY,'gemini':True,'gemini_model':MODEL,'default_provider':'gemini'}):
                result=store.configure_gemini({'api_key':'unit-test-secret','model':MODEL})
            self.assertEqual(os.getenv('GEMINI_API_KEY'),'unit-test-secret');self.assertNotIn('unit-test-secret',json.dumps(result))
            self.assertEqual(list(Path(directory).iterdir()),[]);store.close()


if __name__=='__main__':unittest.main()
