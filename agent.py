"""
agent.py — the reasoning layer.

This is where the project earns the word "agent". Instead of a fixed
retrieve->answer chain, we give the LLM three TOOLS and let *it* decide which
one to call based on the user's message:

    - "What is backpropagation?"        -> answer_from_notes
    - "Quiz me on cell division"        -> make_quiz
    - "Explain recursion simply"        -> explain_concept

Each tool does its own retrieval + generation (RAG lives *inside* the tool), so
the agent's only real job is ROUTING. That's the cleanest way to see the agent
loop at work.

The agent loop, in plain terms:
    1. The LLM receives the user message + the tool definitions.
    2. It emits a tool call (name + arguments) — this is the "routing" decision.
    3. AgentExecutor runs that tool and feeds the result (the "observation")
       back to the LLM.
    4. The LLM either answers the user or calls another tool. Repeat until done.
"""

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

# The classic tool-calling AgentExecutor API. In LangChain < 1.0 it lives in
# `langchain.agents`; in LangChain >= 1.0 it moved to `langchain_classic.agents`
# (that package ships automatically with langchain 1.x). We try both so this
# works regardless of which version `pip install` resolves.
try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
except ImportError:  # LangChain >= 1.0
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

import config
from ingest import retrieve

# Shared LLM instance. The SAME model both routes (as the agent) and generates
# the grounded answers (inside each tool).
llm = ChatOllama(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    base_url=config.OLLAMA_BASE_URL,
)


# --- Small helpers shared by the tools -----------------------------------

def _format_context(docs) -> str:
    """Join retrieved chunks into a single context block for the prompt."""
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _source_snippet(docs, max_chars: int = 300) -> str:
    """Build a short, human-readable citation from the top retrieved chunk.

    The UI parses the `SOURCE:` line out of each tool's return value to show the
    user exactly which note the answer was grounded in.
    """
    if not docs:
        return "SOURCE: (none)"
    top = docs[0]
    src = top.metadata.get("source", "unknown")
    snippet = top.page_content.strip().replace("\n", " ")
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "…"
    return f"SOURCE: [{src}] {snippet}"


_NO_NOTES_MSG = (
    "I couldn't find anything relevant in your notes. "
    "Have you uploaded and ingested notes on this topic yet?"
)


# --- Robust argument schemas ---------------------------------------------
# Small models like qwen2.5:3b occasionally emit a tool argument wrapped as
# {"type": "string", "value": "..."} instead of a bare string, which would fail
# validation and crash the tool. These "before" validators unwrap that shape so
# the agent loop keeps working with a flaky 3B model.

def _unwrap_str(v):
    if isinstance(v, dict):
        v = v.get("value", v.get("query", v.get("topic", v.get("concept", ""))))
    return str(v)


def _unwrap_int(v, default=5):
    if isinstance(v, dict):
        v = v.get("value", default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class _QueryArgs(BaseModel):
    query: str = Field(description="The user's question, in natural language.")

    @field_validator("query", mode="before")
    @classmethod
    def _coerce(cls, v):
        return _unwrap_str(v)


class _ConceptArgs(BaseModel):
    concept: str = Field(description="The concept to explain simply.")

    @field_validator("concept", mode="before")
    @classmethod
    def _coerce(cls, v):
        return _unwrap_str(v)


class _QuizArgs(BaseModel):
    topic: str = Field(description="The topic to build a quiz about.")
    n: int = Field(default=5, description="How many questions to generate.")

    @field_validator("topic", mode="before")
    @classmethod
    def _coerce_topic(cls, v):
        return _unwrap_str(v)

    @field_validator("n", mode="before")
    @classmethod
    def _coerce_n(cls, v):
        return _unwrap_int(v)


# --- The three tools ------------------------------------------------------

@tool(args_schema=_QueryArgs)
def answer_from_notes(query: str) -> str:
    """Answer a factual question using ONLY the user's uploaded study notes.
    Use this for direct questions about the material, e.g. "What is X?" or
    "How does Y work?". Returns a grounded answer plus a source snippet."""
    docs = retrieve(query)
    if not docs:
        return _NO_NOTES_MSG

    context = _format_context(docs)
    prompt = (
        "You are a study assistant. Answer the question using ONLY the context "
        "below. If the context doesn't contain the answer, say so honestly — do "
        "not use outside knowledge.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\nAnswer:"
    )
    answer = llm.invoke(prompt).content
    return f"{answer}\n\n{_source_snippet(docs)}"


@tool(args_schema=_QuizArgs)
def make_quiz(topic: str, n: int = 5) -> str:
    """Generate a quiz to test the user on a topic from their notes.
    Use this when the user wants to be quizzed / tested / to practice, e.g.
    "quiz me on photosynthesis" or "make 5 questions about the French
    Revolution". Produces a mix of multiple-choice and short-answer questions
    with an answer key, grounded in the notes."""
    docs = retrieve(topic)
    if not docs:
        return _NO_NOTES_MSG

    context = _format_context(docs)
    prompt = (
        f"You are a quiz author. Using ONLY the context below, write {n} quiz "
        "questions about the topic. Mix multiple-choice (with options A-D) and "
        "short-answer questions. Number them. After all questions, add a clearly "
        "labelled 'ANSWER KEY' section with the correct answer for each.\n\n"
        f"TOPIC: {topic}\n\n"
        f"CONTEXT:\n{context}\n\nQuiz:"
    )
    quiz = llm.invoke(prompt).content
    return f"{quiz}\n\n{_source_snippet(docs)}"


@tool(args_schema=_ConceptArgs)
def explain_concept(concept: str) -> str:
    """Explain a concept from the user's notes in simple terms, with an analogy.
    Use this when the user is confused or asks to have something explained
    simply / like they're five / with an example, e.g. "explain gradient
    descent simply". Re-explains grounded in the notes and includes one
    everyday analogy."""
    docs = retrieve(concept)
    if not docs:
        return _NO_NOTES_MSG

    context = _format_context(docs)
    prompt = (
        "You are a patient tutor. Using the context below, re-explain the "
        "concept as simply as possible for a beginner. Keep it short, then give "
        "exactly ONE everyday analogy that makes it click.\n\n"
        f"CONCEPT: {concept}\n\n"
        f"CONTEXT:\n{context}\n\nSimple explanation:"
    )
    explanation = llm.invoke(prompt).content
    return f"{explanation}\n\n{_source_snippet(docs)}"


# --- Assemble the tool-calling agent -------------------------------------

TOOLS = [answer_from_notes, make_quiz, explain_concept]

# Human-friendly labels the UI shows while a given tool is running.
TOOL_STATUS = {
    "answer_from_notes": "🔍 Retrieving from notes…",
    "make_quiz": "📝 Building quiz…",
    "explain_concept": "💡 Explaining concept…",
}

# The system prompt is the agent's routing "brain": it tells the LLM what the
# tools are for so it picks correctly. `agent_scratchpad` is where LangChain
# stores the running record of tool calls + observations during the loop.
_SYSTEM = (
    "You are Study Companion, an agent that helps a student learn from THEIR "
    "OWN uploaded notes. Route every message to exactly ONE of these tools:\n"
    "- answer_from_notes: the DEFAULT for factual questions — 'what is', "
    "'define', 'how does', 'when', 'why', 'list'. Use this unless the user "
    "clearly wants a quiz or a simplified explanation.\n"
    "- make_quiz: ONLY when the user asks to be quizzed/tested or to practice "
    "('quiz me', 'test me', 'give me questions'). If they name a number of "
    "questions, pass it as the `n` argument (e.g. 'quiz me with 3' -> n=3).\n"
    "- explain_concept: ONLY when the user explicitly asks for a SIMPLE "
    "explanation, an analogy, or says they're confused ('explain simply', "
    "'ELI5', 'I don't get').\n\n"
    "Do not answer from your own general knowledge — the tools ground every "
    "answer in the student's notes."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def build_agent() -> AgentExecutor:
    """Construct the tool-calling agent executor.

    `return_intermediate_steps=True` is what lets the UI see WHICH tool ran and
    grab its source snippet. `handle_parsing_errors=True` keeps a small 3B model
    from crashing the loop if it emits a slightly malformed tool call.
    """
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=4,
    )
