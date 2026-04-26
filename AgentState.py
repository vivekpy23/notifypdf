from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    # Research context
    search_query: str
    retrieved_chunks: List[dict]  # Contains text, metadata, and reason_selected
    
    # Agent outputs
    candidate_fact: Optional[dict]
    critic_verdict: Optional[dict]
    archivist_report: Optional[dict]
    final_notification: Optional[dict]
    
    # Control variables
    iteration_count: int
    error_log: List[str]