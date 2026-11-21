"""Web job orchestration. All research is delegated to the existing pipeline CLI."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from website.local_llm import settings as local_settings,status as local_status
from website.gemini import probe as gemini_probe,key_from_environment
from task5.config import DEFAULT_GEMINI_MODEL

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "integrated_pipeline"
TERMINAL = {"completed", "partial", "failed", "cancelled", "interrupted"}
STAGES = [("query", "Understand question", "task1_input.json"),
          ("search", "Find sources", "task3_input.json"),
          ("extract", "Read documents", "task3_output.json"),
          ("rank", "Select evidence", "task4_output.json"),
          ("answer", "Write answer", "task5_output.json")]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        path = Path(path)
        if path.stat().st_size > 20_000_000:
            return None
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def atomic_json(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def default_python():
    if os.environ.get("PIPELINE_PYTHON"):
        return os.environ["PIPELINE_PYTHON"]
    candidate = ROOT / ".integration-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(candidate) if candidate.is_file() else sys.executable


def pipeline_environment(request=None):
    environment=dict(os.environ)
    tessdata=ROOT/'website/.ocr/tessdata'
    if all((tessdata/(language+'.traineddata')).is_file() for language in ('hin','eng')):
        environment.setdefault('TESSDATA_PREFIX',str(tessdata))
    local=local_settings()
    if local['enabled']:
        environment.update(OLLAMA_BASE_URL=local['base_url'],OLLAMA_MODEL=local['model'],
                           OLLAMA_NUM_CTX=str(local['num_ctx']),OLLAMA_NUM_PREDICT=str(local['num_predict']),
                           OLLAMA_KEEP_ALIVE=local['keep_alive'])
    if request:
        environment['TASK5_VERIFICATION']=request.get('verification','standard')
        if request.get('judge_provider'):environment['TASK5_JUDGE_PROVIDER']=request['judge_provider']
    return environment


def capabilities(python,gemini_state=None):
    from task5.config import load_env_file
    load_env_file(PIPELINE / ".env")
    load_env_file(ROOT / "task5_standalone/.env")
    probe = "import importlib.util,json; print(json.dumps({m:bool(importlib.util.find_spec(m)) for m in ['numpy','sklearn','requests','bs4','pymupdf','ddgs','fastapi','torch','sentence_transformers','onnxruntime','tokenizers']}))"
    try:
        result = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=15)
        libraries = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        libraries = {}
    config = read_json(PIPELINE / "configs/deployment.json") or {}
    model = Path(config.get("semantic_model", {}).get("name", "not-configured"))
    model = model if model.is_absolute() else PIPELINE / model
    semantic = all((model / p).is_file() for p in ("modules.json", "config.json", "model.safetensors")) and all(libraries.get(m) for m in ("torch", "sentence_transformers"))
    live = all(libraries.get(m) for m in ("numpy", "sklearn", "requests", "bs4", "pymupdf", "ddgs", "fastapi"))
    gemini_state=gemini_state or gemini_probe()
    gemini=gemini_state['ready']
    from task5.config import DEFAULT_NLI_DIRECTORY
    nli=all(libraries.get(name) for name in ('numpy','onnxruntime','tokenizers')) and all((Path(DEFAULT_NLI_DIRECTORY)/name).is_file() for name in ('model.onnx','tokenizer.json','config.json','manifest.json'))
    local=local_status()
    languages=[]
    executable=shutil.which('tesseract')
    if executable:
        try:
            response=subprocess.run([executable,'--list-langs'],capture_output=True,text=True,timeout=5,env=pipeline_environment())
            languages=[line.strip() for line in response.stdout.splitlines() if line.strip() in {'hin','eng','osd'}]
        except (OSError,subprocess.TimeoutExpired):pass
    ocr=all(language in languages for language in ('hin','eng'))
    notices=[]
    if gemini:notices.append('Gemini synthesis is ready. The selected evidence is sent to Google for generation and judging.')
    else:notices.append(gemini_state['message'])
    if not gemini:
        if local['enabled'] and not local['ready']:notices.append(local['message'])
        elif local['ready']:notices.append('Answers are phrased by the local model with source quotes and a support review. The first answer may take longer while the model loads.')
        else:notices.append('Source extraction preserves the language of the evidence. Configure a local model or Gemini for translated synthesis.')
    if not ocr:notices.append('OCR runtime/language data is unavailable; image-only sources may fail.')
    return {"connected": True, "live_search": live, "semantic_ranking": bool(semantic), "gemini": gemini,
            'gemini_configured':gemini_state['configured'],'gemini_model':gemini_state['model'],'gemini_message':gemini_state['message'],
            'nli':bool(nli),'default_verification':'standard','default_judge_provider':'gemini' if gemini else 'same',
            "ollama":local['ready'],"ollama_model":local['model'],"ollama_message":local['message'],
            "ocr":ocr,"ocr_languages":languages,
            "default_ranker": "production" if semantic else "lexical", "default_provider": "gemini" if gemini else "ollama" if local['ready'] else "extractive",
            "ranking_label": "Semantic ranking" if semantic else "Lexical baseline",
            "generation_label": 'Gemini synthesis' if gemini else ("Local Qwen3 4B" if local['model'].startswith('qwen3:4b') else 'Local '+local['model']) if local['ready'] else "Local model unavailable" if local['enabled'] else "Source extraction",
            "saved_example": (ROOT / "task5_standalone/examples/supplied_task4_output.json").is_file(),
            "notice":' '.join(notices) or None}


def validate_request(data, ready, mode="live"):
    if not isinstance(data, dict):
        raise ValueError("Send a JSON object")
    if mode not in {"live", "example", "task4"}:
        raise ValueError("Unsupported search mode")
    language = data.get("language") or None
    if language not in {None, "hi", "en"}:
        raise ValueError("Choose automatic language, Hindi or English")
    provider = data.get("provider") or ready["default_provider"]
    if provider not in {"auto", "extractive", "gemini", "ollama"}:
        raise ValueError("Unsupported answer provider")
    if provider=='auto':provider='gemini' if ready['gemini'] else 'ollama' if ready.get('ollama') else 'extractive'
    if provider=='ollama' and not ready.get('ollama'):
        raise ValueError(ready.get('ollama_message') or 'The local model is unavailable. Run python3 website/setup_llm.py and restart the website.')
    if provider == "gemini" and not ready["gemini"]:
        raise ValueError(ready.get('gemini_message') or 'Configure Gemini with a backend API key and an available model.')
    ranker = (data.get("ranker") or ready["default_ranker"]) if mode=='live' else 'supplied'
    if ranker not in {"production", "lexical",'supplied'} or mode=='live' and ranker=='supplied':
        raise ValueError("Unsupported evidence ranker")
    verification=data.get('verification') or ready.get('default_verification','basic')
    if verification not in {'basic','standard','thorough'}:raise ValueError('Choose basic, standard or thorough verification')
    request = {"mode": mode, "language": language, "provider": provider, "ranker": ranker,'verification':verification}
    judge=data.get('judge_provider') or ready.get('default_judge_provider','same')
    if judge not in {'same','gemini'}:raise ValueError('Choose same-provider or Gemini review')
    if judge=='gemini' and not ready.get('gemini'):raise ValueError('Configure Gemini before selecting Gemini review')
    request['judge_provider']='same' if provider=='extractive' else judge
    if provider=='ollama':request['model']=ready['ollama_model']
    elif provider=='gemini':request['model']=ready.get('gemini_model') or DEFAULT_GEMINI_MODEL
    if mode == "live":
        query = data.get("query")
        if not isinstance(query, str) or not 2 <= len(query.strip()) <= 500:
            raise ValueError("Enter a question between 2 and 500 characters")
        if not ready["live_search"]:
            raise ValueError("The pipeline environment is incomplete. Install integrated_pipeline/requirements-integration.txt.")
        if ranker == "production" and not ready["semantic_ranking"]:
            raise ValueError("BGE-M3 is not installed. Choose the existing lexical baseline in search options.")
        request["query"] = query.strip()
    elif mode == "example":
        if not ready["saved_example"]:
            raise ValueError("The saved example is unavailable")
        request["query"] = "राम कैसे थे? राम को क्यों भगवान मानते हैं लोग?"
    else:
        payload = data.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Upload a Task 4 JSON object")
        from task5.handoff import validate_envelope
        batch = payload.get("task4_output")
        if batch is not None:
            if not isinstance(batch, dict) or batch.get("schema_version") != "1.0" or not isinstance(batch.get("queries"), list):
                raise ValueError("Expected Task 4 batch schema 1.0")
            envelopes = batch["queries"]
        else:
            envelopes = [payload]
        if not 1 <= len(envelopes) <= 20:
            raise ValueError("Upload one to twenty Task 4 questions")
        for envelope in envelopes:
            validate_envelope(envelope)
        request["query"] = envelopes[0]["query"]
    # Explicit language instructions in the user's question override a stale
    # selector. Each Task 5 batch item resolves independently as well.
    from task5.language import explicit_answer_language
    if mode!='task4' or len(envelopes)==1:
        request['language']=explicit_answer_language(request['query']) or language
    return request


def command_for(request, directory, python):
    command = [python, "-u", str(PIPELINE / "run_integrated_pipeline.py"), "--output-dir", str(directory),
               "--provider", request["provider"], "--task4-top-k", "24"]
    if request["mode"] == "live":
        # The equals form treats even a query beginning with '--' as one literal value.
        command += ["--query=" + request["query"], "--max-queries", "2", "--top-k", "3"]
        if request["ranker"] == "lexical":
            # A threshold calibrated for semantic scores is unsuitable for TF-IDF.
            # Keep Task 4's existing top-k ranking, then Task 5's relevance checks.
            # This does not change the deployment configuration or task interfaces.
            command += ["--lexical-only", "--task4-config", str(ROOT / "website/configs/lexical.json")]
    else:
        command += ["--task4-input", str(directory / "input.json")]
    if request["language"]:
        command += ["--language", request["language"]]
    if request['provider'] in {'ollama','gemini'}:command += ['--model',request['model']]
    if request["provider"] in {"gemini","ollama"}:
        command.append("--strict")
    return command


def public_answers(payload):
    answers = []
    for item in payload.get("task5_output", {}).get("queries", []):
        result = item["result"]
        public = item.get("public_output", {})
        calls=result.get('diagnostics',{}).get('model_calls',[])
        completed=[call for call in calls if call.get('status')=='completed']
        drafting=[call for call in completed if call.get('purpose') in {'draft','generation'}]
        model=next((call.get('resolved_model') or call.get('requested_model') for call in drafting or completed),None)
        if result['generation_mode'] in {'abstained','extractive','extractive_fallback'}:model=None
        answers.append({"query": item["query"], "answer": result["answer"], "reliability": result["reliability"],
                        "why": result["reliability_explanation"], "language": result["target_language"],
                        "sources": public.get("supporting_sources", result.get("supporting_sources", [])),
                        "citations": result["cited_sources"], "answer_status": result.get("answer_status", "answered"),
                        "generation_mode": result["generation_mode"], "warnings": result.get("warnings", []),
                        "model":model,"model_review_completed":result.get('diagnostics',{}).get('review',{}).get('status')=='completed',
                        "factors": result.get("factors", {}), "conflicts": result.get("conflicts", []),
                        'verification':result.get('diagnostics',{}).get('review',{}),
                        "is_simulated": item.get("is_simulated", False)})
    return answers


def answer_text(answer):
    lines = [answer["query"], "", answer["answer"], "", "Reliability: " + answer["reliability"],
             "Why: " + answer["why"], "", "Supporting sources:"]
    for source in answer["sources"]:
        label = "[" + source["id"] + "] " + (source.get("title") or "Source passage")
        if source.get("page") is not None:
            label += " — page " + str(source["page"])
        lines.append(label)
        if source.get("url"):
            lines.append(str(source["url"]))
    if not answer["sources"]:
        lines.append("None")
    return "\n".join(lines)


def pipeline_trace(folder, payload):
    """Expose counts and mapping checks, so transport failures are distinguishable."""
    final=payload.get("task5_output", {})
    handoff=read_json(folder / "task4_output.json") or read_json(folder / "input.json") or {}
    task4=handoff.get("task4_output", {})
    envelopes=task4.get("queries", [handoff] if "ranked_evidence" in handoff else [])
    queries=final.get("queries", [])
    mapping_ok=len(envelopes)==len(queries) and all(
        row.get("query")==answer.get("query") and
        set(answer.get("citation_to_task4_passage_id",{}).values())=={p["passage_id"] for p in row.get("ranked_evidence",[])}
        for row,answer in zip(envelopes,queries))
    summary=(read_json(folder / "task3_output.json") or {}).get("task3_output",{}).get("summary",task4.get("task3_summary") or {})
    return {"handoff_verified":mapping_ok,"questions_received":len(queries),
            "documents_processed":summary.get("processed"),
            "candidate_passages":task4.get("summary",{}).get("candidate_passages"),
            "passages_received_by_task5":sum(len(e.get("ranked_evidence",[])) for e in envelopes),
            "passages_selected_by_task5":sum(sum(d.get("selected") is True for d in q["result"].get("diagnostics",{}).get("selection",[])) for q in queries),
            "accepted_claims":sum(len(q["result"].get("claims",[])) for q in queries),
            "generation_modes":list(dict.fromkeys(q['result'].get('generation_mode','') for q in queries)),
            "models":list(dict.fromkeys(call.get('resolved_model') or call.get('requested_model') for q in queries for call in q['result'].get('diagnostics',{}).get('model_calls',[]) if call.get('status')=='completed')),
            "completed_model_calls":sum(call.get('status')=='completed' for q in queries for call in q['result'].get('diagnostics',{}).get('model_calls',[])),
            "model_reviews_completed":sum(q['result'].get('diagnostics',{}).get('review',{}).get('status')=='completed' for q in queries),
            "implementation_version":final.get("implementation_version")}


class JobStore:
    def __init__(self, directory=None, python=None, ready=None, timeout=300, command_builder=None):
        self.directory = Path(directory or ROOT / "website/.runs").resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.python = python or default_python()
        self.ready = ready if ready is not None else capabilities(self.python)
        self.timeout = timeout
        self.command_builder = command_builder or command_for
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1 if self.ready.get('ollama') or self.ready.get('nli') else 2, thread_name_prefix="research")
        self.jobs, self.processes = {}, {}
        for folder in sorted(self.directory.iterdir())[-100:]:
            if not folder.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", folder.name):
                continue
            job = read_json(folder / "job.json")
            if isinstance(job, dict) and job.get("id") == folder.name:
                if job.get("status") not in TERMINAL:
                    job.update(status="interrupted", error="The backend stopped before this search finished.", finished_at=utc_now())
                    atomic_json(folder / "job.json", job)
                self.jobs[job["id"]] = job

    def configure_gemini(self,data):
        from task5.config import load_env_file
        load_env_file(PIPELINE/'.env');load_env_file(ROOT/'task5_standalone/.env')
        supplied=data.get('api_key','')
        if not isinstance(supplied,str) or len(supplied)>500 or '\n' in supplied or '\r' in supplied:
            raise ValueError('Enter a valid API key')
        key=supplied.strip() or key_from_environment()
        model=data.get('model') or os.getenv('GEMINI_MODEL') or DEFAULT_GEMINI_MODEL
        if not isinstance(model,str):raise ValueError('Enter a Gemini model ID')
        result=gemini_probe(model,key)
        if not result['ready']:raise ValueError(result['message'])
        if supplied.strip():os.environ['GEMINI_API_KEY']=key
        os.environ['GEMINI_MODEL']=result['model']
        refreshed=capabilities(self.python,gemini_state=result)
        with self.lock:self.ready.update(refreshed)
        return {'health':dict(self.ready),'message':'Gemini is configured for this backend session. Use the ignored backend .env file for persistent settings.'}

    def submit(self, data, mode="live"):
        request = validate_request(data, self.ready, mode)
        with self.lock:
            if sum(job["status"] not in TERMINAL for job in self.jobs.values()) >= 4:
                raise ValueError("Four searches are already active. Wait for one to finish.")
            identity = uuid.uuid4().hex
            folder = self.directory / identity
            folder.mkdir()
            if mode == "example":
                payload = read_json(ROOT / "task5_standalone/examples/supplied_task4_output.json")
                atomic_json(folder / "input.json", payload)
            elif mode == "task4":
                atomic_json(folder / "input.json", data["payload"])
            job = {"id": identity, "status": "queued", "created_at": utc_now(), "request": request, "error": None}
            self.jobs[identity] = job
            atomic_json(folder / "job.json", job)
            self.pool.submit(self._run, identity)
        return self.snapshot(identity)

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=3)
        except ProcessLookupError:
            pass

    def _run(self, identity):
        folder = self.directory / identity
        started = time.monotonic()
        try:
            with self.lock:
                job = self.jobs[identity]
                if job["status"] == "cancelled":
                    return
                job.update(status="running", started_at=utc_now())
                atomic_json(folder / "job.json", job)
            with (folder / "run.log").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(self.command_builder(job["request"], folder, self.python), cwd=ROOT,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"),env=pipeline_environment(job['request']))
                with self.lock:
                    self.processes[identity] = process
                    if job["status"] == "cancelled":
                        self._terminate(process)
                try:
                    code = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    self._terminate(process)
                    raise RuntimeError("The search timed out. Try a more specific question or check the backend setup.")
            payload = read_json(folder / "task5_output.json")
            with self.lock:
                if job["status"] != "cancelled":
                    if code in {0, 1} and payload and public_answers(payload):
                        job["status"] = "partial" if code == 1 else "completed"
                    else:
                        message="The pipeline could not complete this search. Check the run log or try the saved example."
                        log=(folder/'run.log').read_text(encoding='utf-8',errors='replace')[-6000:]
                        failures=re.findall(r'\[(?:INTEGRATION|TASK 5)\]\s+ProviderError:\s*([^\n]+)',log)
                        if failures:message='The answer model could not produce a supported answer: '+failures[-1][:220]+'. Retry or choose Source extraction in search options.'
                        job.update(status="failed", error=message)
                job.update(exit_code=code, finished_at=utc_now(), duration_seconds=round(time.monotonic()-started, 2))
                atomic_json(folder / "job.json", job)
        except Exception as exc:
            with self.lock:
                job = self.jobs[identity]
                if job["status"] != "cancelled":
                    job.update(status="failed", error=str(exc)[:400], finished_at=utc_now())
                    atomic_json(folder / "job.json", job)
        finally:
            with self.lock:
                self.processes.pop(identity, None)

    def cancel(self, identity):
        with self.lock:
            job = self.jobs[identity]
            if job["status"] not in TERMINAL:
                job.update(status="cancelled", finished_at=utc_now())
                if identity in self.processes:
                    self._terminate(self.processes[identity])
                atomic_json(self.directory / identity / "job.json", job)
        return self.snapshot(identity)

    def snapshot(self, identity, details=False):
        with self.lock:
            job = json.loads(json.dumps(self.jobs[identity]))
        folder = self.directory / identity
        start = 0 if job["request"]["mode"] == "live" else 4
        complete = {name for _, _, name in STAGES if read_json(folder / name) is not None}
        active = next((i for i, (_, _, name) in enumerate(STAGES) if i >= start and name not in complete), None)
        job["stages"] = [{"id": key, "label": label, "state": "supplied" if i < start else "completed" if name in complete
                          else job["status"] if i == active and job["status"] in {"running", "failed", "cancelled", "interrupted"} else "pending"}
                         for i, (key, label, name) in enumerate(STAGES)]
        if details:
            payload = read_json(folder / "task5_output.json")
            job["answers"] = public_answers(payload) if payload else []
            job["execution_mode"] = payload.get("task5_output", {}).get("execution_mode") if payload else None
            if payload:
                from task5.config import IMPLEMENTATION_VERSION, implementation_fingerprint
                job["pipeline_trace"] = pipeline_trace(folder,payload)
                job["result_is_current"] = (payload.get("task5_output",{}).get("implementation_version")==IMPLEMENTATION_VERSION and
                                            payload.get("task5_output",{}).get("implementation_fingerprint")==implementation_fingerprint())
        return job

    def history(self):
        with self.lock:
            identities = sorted(self.jobs, key=lambda key: self.jobs[key]["created_at"], reverse=True)[:20]
        return [self.snapshot(identity) for identity in identities]

    def download(self, identity, kind):
        if identity not in self.jobs:
            raise KeyError(identity)
        folder = self.directory / identity
        if kind == "log":
            path = folder / "run.log"
            if not path.is_file():
                raise KeyError(kind)
            return path.read_bytes()[-48000:], "text/plain; charset=utf-8", "research-log.txt"
        payload = read_json(folder / "task5_output.json")
        if not payload:
            raise KeyError(kind)
        if kind == "json":
            return json.dumps(payload, ensure_ascii=False, indent=2).encode(), "application/json; charset=utf-8", "research-result.json"
        if kind == "text":
            text = "\n\n".join(answer_text(answer) for answer in public_answers(payload))
            return text.encode(), "text/plain; charset=utf-8", "research-answer.txt"
        raise KeyError(kind)

    def close(self):
        with self.lock:
            for identity in list(self.jobs):
                if self.jobs[identity]["status"] not in TERMINAL:
                    self.cancel(identity)
        self.pool.shutdown(wait=True, cancel_futures=True)
