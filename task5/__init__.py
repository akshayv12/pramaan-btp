"""Task 5: concise multi-document answers with source-grounded support assessment."""
from .config import Task5Config
from .schemas import EvidencePassage,GroundedClaim,ClaimEvidence,SummaryResult,ReliabilityRating,SourceSignal
from .pipeline import Task5Pipeline,summarize_with_reliability
from .handoff import adapt_task4_envelope,summarize_task4_payload,process_task4_output

__all__=['Task5Config','EvidencePassage','GroundedClaim','ClaimEvidence','SummaryResult','ReliabilityRating','SourceSignal',
         'Task5Pipeline','summarize_with_reliability','adapt_task4_envelope','summarize_task4_payload','process_task4_output']
