"""The query, ranked-evidence and cited-answer contracts for Task 5."""
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import re
from urllib.parse import urlsplit


class SourceSignal(str,Enum):
    HIGH='High'
    MEDIUM='Medium'
    LOW='Low'
    UNKNOWN='Unknown'

    @classmethod
    def from_str(cls,value):
        return {'high':cls.HIGH,'medium':cls.MEDIUM,'medium/high':cls.MEDIUM,'low':cls.LOW}.get(str(getattr(value,'value',value)).lower().strip(),cls.UNKNOWN)


class ReliabilityRating(str,Enum):
    HIGH='HIGH'
    MEDIUM='MEDIUM'
    LOW='LOW'


def unit_score(name,value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name+' must be finite in [0,1]')
    return float(value)


@dataclass
class EvidencePassage:
    id: str
    text: str
    source_type: str
    query_rel: float
    india_rel: float
    source_signal: str
    ocr_conf: float | None = None
    language: str = 'hi'
    url_or_provenance: str | None = None
    title: str | None = None
    source_id: str | None = None
    publisher: str | None = None
    page: int | None = None
    extraction_method: str = 'direct'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}',self.id):
            raise ValueError('Invalid citation ID')
        if not isinstance(self.text,str) or len(self.text)>100000:
            raise ValueError('Evidence text must be a bounded string')
        self.query_rel=unit_score('query_rel',self.query_rel)
        self.india_rel=unit_score('india_rel',self.india_rel)
        if self.ocr_conf is not None:self.ocr_conf=unit_score('ocr_conf',self.ocr_conf)
        if not isinstance(self.metadata,dict):raise ValueError('metadata must be an object')
        for name in ('source_type','language','extraction_method'):
            if not isinstance(getattr(self,name),str):raise ValueError(name+' must be a string')
        for name in ('url_or_provenance','title','source_id','publisher'):
            if getattr(self,name) is not None and not isinstance(getattr(self,name),str):raise ValueError(name+' must be a string or null')
        if self.page is not None and (type(self.page) is not int or self.page<1):raise ValueError('Invalid page number')

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict):raise ValueError('Evidence must be an object')
        row=dict(data)
        for old,new in {'passage_id':'id','url':'url_or_provenance','query_relevance':'query_rel','indian_context_relevance':'india_rel','ocr_confidence':'ocr_conf'}.items():
            if old in row:row.setdefault(new,row.pop(old))
        fields=cls.__dataclass_fields__
        metadata=row.get('metadata',{})
        if not isinstance(metadata,dict):raise ValueError('metadata must be an object')
        row['metadata']={**metadata,**{k:v for k,v in row.items() if k not in fields}}
        try:return cls(**{k:v for k,v in row.items() if k in fields})
        except TypeError as exc:raise ValueError('Missing/invalid evidence fields') from exc

    def parsed_source_signal(self):return SourceSignal.from_str(self.source_signal)
    def is_ocr(self):return self.ocr_conf is not None or 'ocr' in self.extraction_method.lower()
    def source_identity(self):
        if self.metadata.get('task5_source_group'):return 'group:'+str(self.metadata['task5_source_group'])
        if self.publisher:return 'publisher:'+self.publisher.strip().casefold()
        try:host=urlsplit(self.url_or_provenance or '').hostname
        except ValueError:host=None
        if host:
            count=3 if host.endswith(('.ac.in','.gov.in','.co.in','.org.in','.co.uk')) else 2
            return 'domain:'+'.'.join(host.lower().removeprefix('www.').split('.')[-count:])
        return 'source:'+self.source_id if self.source_id else None
    def to_dict(self):return asdict(self)


@dataclass
class ClaimEvidence:
    passage_id: str
    quote: str


@dataclass
class GroundedClaim:
    text: str
    evidence: list[ClaimEvidence]
    support_method: str = 'exact_extraction'


@dataclass
class SummaryResult:
    query: str
    answer: str
    target_language: str
    cited_sources: list[str]
    reliability: ReliabilityRating
    reliability_explanation: str
    factors: dict
    used_passages: list[EvidencePassage] = field(default_factory=list)
    claims: list[GroundedClaim] = field(default_factory=list)
    supporting_sources: list[dict] = field(default_factory=list)
    selected_passages: list[EvidencePassage] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generation_mode: str = 'extractive'
    answer_status: str = 'answered'
    diagnostics: dict = field(default_factory=dict)
    schema_version: str = '3.0'

    def to_dict(self):
        value=asdict(self);value['reliability']=self.reliability.value
        return value

    def to_public_dict(self):
        return {'answer':self.answer,'language':self.target_language,'reliability':self.reliability.value,
                'why':self.reliability_explanation,'sources_used':self.cited_sources,
                'supporting_sources':self.supporting_sources,'answer_status':self.answer_status,'generation_mode':self.generation_mode}

    def to_formatted_display(self):
        label='उत्तर' if self.target_language=='hi' else 'Answer'
        return '\n'.join([label+': '+self.answer,'','Reliability: '+self.reliability.value,
                          'Why: '+self.reliability_explanation,'Sources used: '+(', '.join(self.cited_sources) or 'None')])
