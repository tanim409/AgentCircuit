import operator
from typing import List, TypedDict, Annotated, Literal, Optional, Dict, Any
from pydantic import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field( description="One sentence describing what the reader should be able to do/understand after this section.")
    bullets: List[str] = Field(
        min_length=3,
        max_length=5,
        description="3–5 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(description="Target word count for this section (700-800).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class Plan(BaseModel):
   blog_title: str
   audience: str = Field(description="Who this blog is for.")
   tone: str = Field(description="Writing tone (e.g., practical, crisp).")
   blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
   narrative_thread: str = Field(description="The overarching story arc or core argument connecting all sections together.")
   tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str]
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    need_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)

class EvidencePackage(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class State(TypedDict):
    topic:str
    plan:Plan
    need_research: bool
    mode:str
    as_of: str
    recency_days:int
    queries:List[str]
    evidence: List[EvidenceItem]
    sections:Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final:str
    research_status: str
    tool_call_history: List[Dict[str, Any]]
    consecutive_repeats: int
    cb_status: str
    progress_history: List[int]
    stagnant_steps: int
    cost_history: List[float]
    breaker_state:str

class WorkPayload(TypedDict):
   title:str
   plan:Plan
   task:Task
   topic:str
   mode:str
   evidence: List[EvidenceItem]
   as_of: str
   recency_days:int
   narrative_thread: str

class ImageSpec(BaseModel):
    placeholder: str = Field(description="e.g. [[IMAGE_1]]")
    filename: str = Field(description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    image_specs: List[ImageSpec] = Field(default_factory=list)

class Citation(BaseModel):
    marker_id: int = Field(description="The numeric citation marker, e.g. 1 for [1]")
    source_url: str = Field(description="The exact URL from the provided evidence used for this claim")

class SectionOutput(BaseModel):
    content: str = Field(description="Markdown content with inline citation markers like [1], [2]")
    citations: List[Citation] = Field(description="List of citations mapping markers used in content to evidence URLs")


class circuitBacker(Exception):
    """Exception raised when circuit breaker trips due to repeated failures or cost limits."""
   pass



