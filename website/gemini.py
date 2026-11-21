"""Gemini readiness and server-only credentials. Public status never contains a key."""
import json
import os
import re
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from task5.config import DEFAULT_GEMINI_MODEL


def key_from_environment():return os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY','')


def probe(model=None,key=None):
    model=(model or os.getenv('GEMINI_MODEL') or DEFAULT_GEMINI_MODEL).removeprefix('models/')
    key=key or key_from_environment()
    result={'configured':bool(key),'ready':False,'model':model}
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}',model):raise ValueError('Enter a Gemini model ID, without a URL')
    if not key:return {**result,'message':'Gemini needs a backend API key. Configure Gemini or set GEMINI_API_KEY in integrated_pipeline/.env.'}
    try:
        request=Request('https://generativelanguage.googleapis.com/v1beta/models/'+model,headers={'x-goog-api-key':key})
        with urlopen(request,timeout=10) as response:raw=response.read(1_000_001)
        if len(raw)>1_000_000:raise ValueError('Model readiness response was too large')
        value=json.loads(raw)
        if 'generateContent' not in value.get('supportedGenerationMethods',[]):
            return {**result,'message':'This model does not advertise text generation. Select another Gemini model.'}
        return {**result,'ready':True,'message':'Model access is recognized. The first synthesis request verifies generation and quota.'}
    except HTTPError as error:
        status=error.code;error.close()
        message={400:'The Gemini key or model request was rejected.',401:'The Gemini API key was not accepted.',
                 403:'This key/project cannot access the selected Gemini model.',404:'The selected Gemini model is unavailable to this key.',
                 429:'Gemini quota is currently unavailable.'}.get(status,'Gemini model readiness failed (HTTP '+str(status)+').')
        return {**result,'message':message}
    except (OSError,ValueError):return {**result,'message':'Gemini could not be reached or returned an invalid readiness response.'}
