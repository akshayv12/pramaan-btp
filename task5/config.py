"""Small, explicit Task 5 configuration. No credentials are serialized."""
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from functools import lru_cache
import hashlib

IMPLEMENTATION_VERSION = '3.8'
DEFAULT_GEMINI_MODEL = 'gemini-3.5-flash-lite'
DEFAULT_NLI_DIRECTORY = str(Path(__file__).resolve().parents[1]/'website/.models/nli')


@lru_cache(maxsize=1)
def implementation_fingerprint():
    digest=hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob('*.py')):
        # Offline evaluation/review tooling does not alter an answer's runtime.
        if source.name in {'evaluation.py','experiments.py','human_review.py'}:continue
        digest.update(source.name.encode());digest.update(source.read_bytes())
    # Only fitted decision rules alter behavior. Pilot metadata is recorded
    # separately and does not invalidate generation results when a report grows.
    from .support import support_policy
    policy=support_policy()
    digest.update(str(policy['high_requires_nli_entailment']).encode())
    digest.update(str(policy.get('high_allow_unknown_source_with_nli',False)).encode())
    digest.update(str(policy.get('high_blocked_judge_models',[])).encode())
    return digest.hexdigest()


def load_env_file(path='.env'):
    allowed = {'TASK5_PROVIDER','GEMINI_API_KEY','GEMINI_MODEL','OLLAMA_MODEL','OLLAMA_BASE_URL','BRAVE_API_KEY',
               'OLLAMA_NUM_CTX','OLLAMA_NUM_PREDICT','OLLAMA_KEEP_ALIVE','TASK5_MODEL_TIMEOUT','TASK5_REQUEST_DEADLINE',
               'GOOGLE_API_KEY','TASK5_VERIFICATION','TASK5_NLI_DIRECTORY','TASK5_CALIBRATION_PATH','TASK5_JUDGE_MODEL','TASK5_JUDGE_PROVIDER'}
    p = Path(path)
    if p.is_file():
        for line in p.read_text(encoding='utf-8').splitlines():
            key, separator, value = line.strip().partition('=')
            if separator and key in allowed and value.strip():
                os.environ.setdefault(key,value.strip().strip('\"\''))


@dataclass(frozen=True)
class Task5Config:
    provider: str = 'auto'
    model: str = ''
    max_passages: int = 5
    max_per_source: int = 2
    max_claims: int = 3
    max_answer_chars: int = 1100
    max_passage_chars: int = 6000
    context_chars: int = 18000
    min_query_rel: float = .35
    min_ocr_conf: float = .50
    duplicate_threshold: float = .85
    mmr_lambda: float = .70
    context_weight: float = .15
    model_timeout: float = 20.0
    request_deadline: float = 45.0
    allow_fallback: bool = True
    ollama_num_ctx: int = 8192
    ollama_num_predict: int = 1024
    ollama_keep_alive: str = '5m'
    verification_profile: str = 'basic'
    nli_directory: str = DEFAULT_NLI_DIRECTORY
    calibration_path: str = ''
    judge_model: str = ''
    judge_provider: str = 'same'
    nli_entailment_threshold: float = .70
    nli_contradiction_threshold: float = .80
    passage_order: str = 'relevance'
    cross_lingual_strategy: str = 'direct'
    query_focused_selection: bool = False

    def __post_init__(self):
        if self.provider not in {'auto','extractive','gemini','ollama'}:
            raise ValueError('Unknown generation provider')
        if self.verification_profile not in {'basic','standard','thorough'}:raise ValueError('Unknown verification profile')
        if self.judge_provider not in {'same','gemini','ollama'}:raise ValueError('Unknown review provider')
        if self.passage_order not in {'relevance','reverse','shuffle'}:raise ValueError('Unknown evidence ordering')
        if self.cross_lingual_strategy not in {'direct','translate_then_summarize'}:raise ValueError('Unknown translation strategy')
        for value in (self.nli_directory,self.calibration_path,self.judge_model):
            if not isinstance(value,str) or len(value)>2000:raise ValueError('Invalid verification configuration')
        if not isinstance(self.model,str) or type(self.allow_fallback) is not bool or type(self.query_focused_selection) is not bool:
            raise ValueError('Invalid model/fallback setting')
        for key in ('max_passages','max_per_source','max_claims','max_answer_chars','max_passage_chars','context_chars'):
            value=getattr(self,key)
            if type(value) is not int or not 1 <= value <= 100000:
                raise ValueError(key+' must be a bounded positive integer')
        if self.max_claims > 8 or self.max_passages > 12:
            raise ValueError('Prototype limits are eight claims and twelve passages')
        if type(self.ollama_num_ctx) is not int or not 2048<=self.ollama_num_ctx<=16384:
            raise ValueError('ollama_num_ctx must be an integer from 2048 to 16384')
        if type(self.ollama_num_predict) is not int or not 128<=self.ollama_num_predict<=2048:
            raise ValueError('ollama_num_predict must be an integer from 128 to 2048')
        if self.ollama_keep_alive not in {'0','1m','2m','5m','10m'}:
            raise ValueError('Invalid local model keep-alive setting')
        for key in ('min_query_rel','min_ocr_conf','duplicate_threshold','mmr_lambda','context_weight','nli_entailment_threshold','nli_contradiction_threshold'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(key+' must be finite in [0,1]')
        for key in ('model_timeout','request_deadline'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 < value <= 120:
                raise ValueError(key+' must be finite in (0,120]')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_env(cls,**overrides):
        provider=overrides.get('provider') or os.getenv('TASK5_PROVIDER','auto')
        if provider=='auto':provider='gemini' if (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')) else 'ollama' if os.getenv('OLLAMA_MODEL') else 'extractive'
        model=os.getenv('OLLAMA_MODEL','') if provider=='ollama' else os.getenv('GEMINI_MODEL','')
        if provider=='gemini' and not model:model=DEFAULT_GEMINI_MODEL
        values={'provider':provider,'model':model}
        if provider in {'gemini','ollama'}:values['query_focused_selection']=True
        if provider=='ollama':
            values.update(model_timeout=90.0,request_deadline=120.0,context_chars=9000,max_passages=4)
        elif provider=='gemini':values.update(model_timeout=60.0,request_deadline=90.0,context_chars=12000,max_passages=4,max_claims=4)
        for key,environment in (('verification_profile','TASK5_VERIFICATION'),('nli_directory','TASK5_NLI_DIRECTORY'),
                                ('calibration_path','TASK5_CALIBRATION_PATH'),('judge_model','TASK5_JUDGE_MODEL'),('judge_provider','TASK5_JUDGE_PROVIDER')):
            if os.getenv(environment):values[key]=os.environ[environment]
        for key,environment,convert in (
            ('model_timeout','TASK5_MODEL_TIMEOUT',float),('request_deadline','TASK5_REQUEST_DEADLINE',float),
            ('ollama_num_ctx','OLLAMA_NUM_CTX',int),('ollama_num_predict','OLLAMA_NUM_PREDICT',int),
            ('ollama_keep_alive','OLLAMA_KEEP_ALIVE',str)):
            if os.getenv(environment):values[key]=convert(os.environ[environment])
        values.update({k:v for k,v in overrides.items() if v is not None})
        return cls(**values)
