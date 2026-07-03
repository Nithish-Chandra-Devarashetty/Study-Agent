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
from langchain_openai import ChatOpenAI
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
from websearch import web_search

# Shared LLM instance. The SAME model both routes (as the agent) and generates
# the grounded answers (inside each tool). We talk to OpenRouter through its
# OpenAI-compatible API, so ChatOpenAI just needs a different base_url + key.
# `.bind_tools()` (used by create_tool_calling_agent below) works as long as the
# chosen OpenRouter model supports function/tool calling — see config.py.
_openrouter_headers = {}
if config.OPENROUTER_APP_URL:
    _openrouter_headers["HTTP-Referer"] = config.OPENROUTER_APP_URL
if config.OPENROUTER_APP_NAME:
    _openrouter_headers["X-Title"] = config.OPENROUTER_APP_NAME

llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    api_key=config.OPENROUTER_API_KEY or "missing-key",
    base_url=config.OPENROUTER_BASE_URL,
    default_headers=_openrouter_headers or None,
)


def check_llm() -> bool:
    """True if the OpenRouter API key is configured.

    Used by the UI to show a friendly "set your API key" message instead of an
    ugly auth error when the key is missing.
    """
    return config.has_llm_key()


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

# When `answer_from_notes` finds nothing, it returns THIS instead. It doubles as
# (a) an honest message the user can read, and (b) a signal the agent's routing
# brain reacts to: if web search is enabled, the system prompt tells the agent it
# may now call `search_web`. Keeping it prose-y means it still reads fine even if
# the agent decides to stop here rather than go to the web.
def _notes_miss_msg() -> str:
    if config.ENABLE_WEB_SEARCH:
        return (
            "NOTES_MISS: I couldn't find this in your uploaded notes. "
            "If this is a general-knowledge question, the `search_web` tool can "
            "look it up online; otherwise let the user know it isn't in their notes."
        )
    return _NO_NOTES_MSG


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
        return _notes_miss_msg()

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


@tool(args_schema=_QueryArgs)
def search_web(query: str) -> str:
    """Look an answer up on the public web. This is a FALLBACK, not a default:
    use it ONLY after `answer_from_notes` reports the notes don't cover the topic
    (a 'NOTES_MISS'), and only for general-knowledge questions. The answer comes
    from the web, NOT the user's notes, and is clearly labelled as such."""
    results = web_search(query)
    if not results:
        return (
            "I couldn't find this in your notes, and the web-search fallback is "
            "unavailable right now (no network, or the search backend isn't "
            "installed). SOURCE: (web search unavailable)"
        )

    context = "\n\n".join(
        f"[{i + 1}] {r['title']}\n{r['snippet']}\n({r['url']})"
        for i, r in enumerate(results)
    )
    prompt = (
        "You are a study assistant. The user's own notes did NOT contain the "
        "answer, so you searched the web. Answer the question concisely using "
        "ONLY the search results below. If they don't actually answer it, say so "
        "honestly.\n\n"
        f"SEARCH RESULTS:\n{context}\n\n"
        f"QUESTION: {query}\n\nAnswer:"
    )
    answer = llm.invoke(prompt).content
    top = results[0]
    src = f"SOURCE: [web] {top['title']} — {top['url']}".strip()
    disclaimer = "\n\n_⚠️ This came from a **web search**, not your notes._"
    return f"{answer}{disclaimer}\n\n{src}"


# --- Assemble the tool-calling agent -------------------------------------

# `search_web` is only offered to the agent when web search is enabled — with it
# off, the agent stays strictly notes-only and can never reach the internet.
TOOLS = [answer_from_notes, make_quiz, explain_concept]
if config.ENABLE_WEB_SEARCH:
    TOOLS.append(search_web)

# Human-friendly labels the UI shows while a given tool is running.
TOOL_STATUS = {
    "answer_from_notes": "🔍 Retrieving from notes…",
    "make_quiz": "📝 Building quiz…",
    "explain_concept": "💡 Explaining concept…",
    "search_web": "🌐 Notes came up empty — searching the web…",
}

# The system prompt is the agent's routing "brain": it tells the LLM what the
# tools are for so it picks correctly. `agent_scratchpad` is where LangChain
# stores the running record of tool calls + observations during the loop.
_SYSTEM = (
    "You are Study Companion, an agent that helps a student learn from THEIR "
    "OWN uploaded notes. Route each message to the right tool:\n"
    "- answer_from_notes: the DEFAULT for factual questions — 'what is', "
    "'define', 'how does', 'when', 'why', 'list'. Use this unless the user "
    "clearly wants a quiz or a simplified explanation.\n"
    "- make_quiz: ONLY when the user asks to be quizzed/tested or to practice "
    "('quiz me', 'test me', 'give me questions'). If they name a number of "
    "questions, pass it as the `n` argument (e.g. 'quiz me with 3' -> n=3).\n"
    "- explain_concept: ONLY when the user explicitly asks for a SIMPLE "
    "explanation, an analogy, or says they're confused ('explain simply', "
    "'ELI5', 'I don't get').\n"
    + (
        "- search_web: a FALLBACK, not a first choice. If (and only if) "
        "answer_from_notes comes back with a 'NOTES_MISS' — the notes don't "
        "cover the question — AND the question is general knowledge, THEN call "
        "search_web to look it up online. Never open with search_web; always try "
        "the notes first.\n"
        if config.ENABLE_WEB_SEARCH else ""
    )
    + "\nGround answers in the student's notes; only fall back to the web when "
    "the notes genuinely don't contain the answer, and never invent facts."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def build_agent(verbose: bool = True) -> AgentExecutor:
    """Construct the tool-calling agent executor.

    `return_intermediate_steps=True` is what lets the UI see WHICH tool ran and
    grab its source snippet. `handle_parsing_errors=True` keeps a small 3B model
    from crashing the loop if it emits a slightly malformed tool call. Pass
    `verbose=False` (e.g. from the eval harness) to silence the step-by-step
    console trace.
    """
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=verbose,
        max_iterations=4,
    )
