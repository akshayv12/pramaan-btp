"""Independent, local multilingual NLI. No downloads or remote Python execution."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import time

MODEL='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
REVISION='8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c'
CHECKSUMS={'model.onnx':'27c39e884c14b03cf46cfc5485971b6db70ff330220d93dfe729c63fde43af0e',
           'tokenizer.json':'3aca3ce69a0a35aeb144a52c4f1d41c4246b8785f8f398315cc8fb6b24057810'}


def readiness(directory):
    path=Path(directory)
    dependencies=all(importlib.util.find_spec(name) is not None for name in ('numpy','onnxruntime','tokenizers'))
    files=all((path/name).is_file() for name in (*CHECKSUMS,'config.json','manifest.json'))
    return {'ready':dependencies and files,'dependencies':dependencies,'files':files,'model':MODEL,'revision':REVISION}


def thresholds(config,language):
    values={'entailment':config.nli_entailment_threshold,'contradiction':config.nli_contradiction_threshold,
            'calibration_status':'uncalibrated_defaults'}
    if not config.calibration_path:return values
    try:
        path=Path(config.calibration_path)
        if path.stat().st_size>1_000_000:raise ValueError()
        data=json.loads(path.read_text())
        profile=data['languages'][language]
        if data['schema_version']!='task5.calibration.v1' or data['nli_model']!=MODEL or data['nli_revision']!=REVISION:
            raise ValueError()
        if profile.get('status')!='reviewed_holdout_evaluated' or profile.get('human_review_attested') is not True:
            raise ValueError()
        if profile.get('train_items',0)<10 or profile.get('heldout_items',0)<10:raise ValueError()
        for name in ('entailment','contradiction'):
            value=profile['thresholds'][name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not .5<=value<=.99:raise ValueError()
            values[name]=float(value)
        values['calibration_status']='reviewed_holdout_evaluated'
    except (OSError,KeyError,TypeError,ValueError):values['calibration_status']='calibration_unavailable_or_insufficient'
    return values


class NLIModel:
    def __init__(self,directory):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        ort.disable_telemetry_events()
        self.np=np;path=Path(directory)
        for name,expected in CHECKSUMS.items():
            digest=hashlib.sha256()
            with (path/name).open('rb') as stream:
                while chunk:=stream.read(1_048_576):digest.update(chunk)
            if digest.hexdigest()!=expected:raise ValueError('NLI model checksum mismatch; rerun website/setup_verification.py')
        labels=json.loads((path/'config.json').read_text())['id2label']
        if labels!={'0':'entailment','1':'neutral','2':'contradiction'}:raise ValueError('Unexpected NLI label order')
        self.tokenizer=Tokenizer.from_file(str(path/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=512,strategy='only_first')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.inputs={value.name for value in self.session.get_inputs()}

    @lru_cache(maxsize=256)
    def score(self,premise,hypothesis):
        started=time.perf_counter();np=self.np
        encoded=self.tokenizer.encode(premise,hypothesis)
        values={'input_ids':encoded.ids,'attention_mask':encoded.attention_mask,'token_type_ids':encoded.type_ids}
        feed={name:np.asarray([values[name]],dtype=np.int64) for name in self.inputs}
        logits=np.asarray(self.session.run(None,feed)[0][0],dtype=np.float64)
        probabilities=np.exp(logits-logits.max());probabilities/=probabilities.sum()
        return {**{name:round(float(value),6) for name,value in zip(('entailment','neutral','contradiction'),probabilities)},
                'truncated':bool(encoded.overflowing),'tokens':len(encoded.ids),'duration_ms':round((time.perf_counter()-started)*1000,2)}


@lru_cache(maxsize=1)
def load_model(directory):return NLIModel(directory)


def check_claims(claims,language,config):
    settings=thresholds(config,language)
    result={'status':'unavailable','model':MODEL,'revision':REVISION,'language':language,'thresholds':settings,
            'claims':[],'interpretation':'Classifier scores concern reference entailment; they are not truth probabilities.'}
    if not readiness(config.nli_directory)['ready']:return {**result,'reason':'Install the optional verifier dependencies and run python3 website/setup_verification.py.'}
    try:
        model=load_model(config.nli_directory)
        for index,claim in enumerate(claims):
            score=model.score(' '.join(reference.quote for reference in claim.evidence),claim.text)
            verdict='borderline'
            if not score['truncated']:
                if score['contradiction']>=settings['contradiction']:verdict='contradicted'
                elif score['entailment']>=settings['entailment']:verdict='entailed'
            result['claims'].append({'claim_index':index,'verdict':verdict,**score})
        result['status']='completed'
    except Exception as error:
        result.update(status='failed',reason=str(error)[:200])
    return result
