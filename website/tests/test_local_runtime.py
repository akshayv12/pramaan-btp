"""Local service ownership and website model selection must be explicit."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from website.backend import command_for,validate_request
from website.local_llm import LocalRuntime
from test_backend import READY

LOCAL={**READY,'ollama':True,'ollama_model':'qwen3:4b-instruct-2507-q4_K_M','default_provider':'ollama'}


class LocalRuntimeTests(unittest.TestCase):
    def test_selected_local_model_is_passed_explicitly_and_strictly(self):
        request=validate_request({'query':'What is Mahabharata?'},LOCAL)
        command=command_for(request,Path('/tmp/job'),'python')
        self.assertEqual(request['provider'],'ollama')
        self.assertIn('--strict',command)
        self.assertEqual(command[command.index('--model')+1],LOCAL['ollama_model'])

    def test_automatic_mode_prefers_the_installed_local_model(self):
        self.assertEqual(validate_request({'query':'A question','provider':'auto'},LOCAL)['provider'],'ollama')

    def test_missing_local_service_does_not_silently_become_extraction(self):
        with self.assertRaisesRegex(ValueError,'local model'):
            validate_request({'query':'A question'},{**LOCAL,'ollama':False})

    def test_an_existing_service_is_never_owned_or_terminated(self):
        with patch('website.local_llm.status',return_value={'running':True,'ready':True}),patch('website.local_llm.subprocess.Popen') as spawn:
            runtime=LocalRuntime();runtime.start();runtime.close()
            spawn.assert_not_called()

    def test_a_service_started_by_the_website_is_stopped_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            process=Mock();process.poll.return_value=None
            with patch('website.local_llm.MODEL_DIRECTORY',Path(directory)/'ollama'),patch('website.local_llm.shutil.which',return_value='/local/ollama'),patch('website.local_llm.status',side_effect=[{'running':False},{'running':True,'ready':True}]),patch('website.local_llm.subprocess.Popen',return_value=process):
                runtime=LocalRuntime();runtime.start();runtime.close()
            process.terminate.assert_called_once();process.wait.assert_called_once()


if __name__=='__main__':unittest.main()
