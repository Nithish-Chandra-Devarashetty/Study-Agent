"""
evaluate.py — a small, reproducible evaluation of the agent.

Two things a grader actually cares about for a RAG *agent*:

  1. ROUTING ACCURACY — given a message, did the agent pick the RIGHT tool?
     (answer / quiz / explain, and — the interesting one — did it fall back to
     `search_web` when, and only when, the notes don't contain the answer?)

  2. ANSWER GROUNDEDNESS — did the answer stay supported by the retrieved notes
     instead of drifting into made-up facts? We grade this with an LLM-as-judge
     that only ever sees the retrieved notes context, the question, and the
     answer, and returns a single label.

It runs against a FIXED corpus (eval/eval_notes.txt) and a FIXED, labelled test
set (eval/eval_set.json) so the numbers are reproducible run to run. To avoid
touching your real ./chroma_db, the eval ingests into a separate throwaway
directory.

Run:
    python evaluate.py                 # full run, writes eval/results.md
    python evaluate.py --no-judge      # routing only (no LLM judge calls)
    python evaluate.py --limit 6       # first 6 cases (quick smoke test)

Needs OPENROUTER_API_KEY set, same as the app.
"""

import argparse
import json
import os
import sys

# Point the whole stack at a throwaway vector store BEFORE importing modules
# that read config.CHROMA_DIR, so a real ./chroma_db is never disturbed.
import config

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "eval")
config.CHROMA_DIR = os.path.join(EVAL_DIR, "_chroma_eval")
config.COLLECTION_NAME = "study_notes_eval"

from ingest import ingest_files, clear_all, retrieve, collection_count  # noqa: E402
from agent import build_agent, llm  # noqa: E402

EVAL_NOTES = os.path.join(EVAL_DIR, "eval_notes.txt")
EVAL_SET = os.path.join(EVAL_DIR, "eval_set.json")
RESULTS_MD = os.path.join(EVAL_DIR, "results.md")


# --- Running one case -----------------------------------------------------

def _tools_used(intermediate_steps) -> list:
    """The ordered list of tool names the agent actually called this turn."""
    return [action.tool for action, _obs in intermediate_steps]


def run_case(agent_executor, case: dict) -> dict:
    """Execute one test question and return the raw trajectory + answer."""
    try:
        result = agent_executor.invoke({"input": case["question"], "chat_history": []})
    except Exception as exc:  # rate limit, network, malformed tool call, etc.
        return {"error": str(exc), "tools": [], "answer": ""}

    tools = _tools_used(result.get("intermediate_steps", []))
    return {
        "error": None,
        "tools": tools,
        "last_tool": tools[-1] if tools else None,
        "answer": result.get("output", ""),
    }


# --- Groundedness: LLM-as-judge ------------------------------------------

_JUDGE_PROMPT = (
    "You are a strict grader. Decide whether the AI ANSWER is grounded in the "
    "provided study-note CONTEXT — i.e. supported by it, not invented.\n\n"
    "CONTEXT (the only notes the answer was allowed to use):\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "AI ANSWER:\n{answer}\n\n"
    "Reply with EXACTLY ONE word from this list and nothing else:\n"
    "GROUNDED   - every substantive claim is supported by the CONTEXT\n"
    "PARTIAL    - mostly supported, but with a minor unsupported detail\n"
    "UNSUPPORTED- key claims are NOT in the CONTEXT (a hallucination)\n"
    "REFUSED    - the answer declines, says it isn't in the notes, or is "
    "labelled as coming from a web search"
)

_VALID_LABELS = {"GROUNDED", "PARTIAL", "UNSUPPORTED", "REFUSED"}


def judge_groundedness(question: str, answer: str) -> str:
    """Return one of GROUNDED / PARTIAL / UNSUPPORTED / REFUSED for an answer.

    The judge only sees the retrieved notes context, so it grades grounding in
    the NOTES specifically — a web-sourced or honest-refusal answer lands as
    REFUSED, which is the correct outcome for out-of-notes questions.
    """
    docs = retrieve(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs) or "(no notes retrieved)"
    prompt = _JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    try:
        raw = llm.invoke(prompt).content.strip().upper()
    except Exception:
        return "ERROR"
    # Take the first recognised label token (models sometimes add a period/space).
    for token in raw.replace(".", " ").split():
        if token in _VALID_LABELS:
            return token
    return raw.split()[0] if raw.split() else "ERROR"


# --- The eval loop --------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate routing + groundedness.")
    ap.add_argument("--no-judge", action="store_true", help="skip the groundedness LLM judge")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    args = ap.parse_args()

    if not config.has_llm_key():
        print("OPENROUTER_API_KEY is not set — set it (see README) and re-run.")
        sys.exit(1)

    with open(EVAL_SET, encoding="utf-8") as f:
        cases = json.load(f)
    if args.limit:
        cases = cases[: args.limit]

    # Fresh, isolated corpus so scores are reproducible.
    print(f"Ingesting eval corpus into {config.CHROMA_DIR} …")
    clear_all()
    print(ingest_files([EVAL_NOTES]))
    print(f"Collection holds {collection_count()} chunks.\n")

    agent_executor = build_agent(verbose=False)

    rows = []
    for case in cases:
        run = run_case(agent_executor, case)
        last_tool = run.get("last_tool")
        routing_ok = (last_tool == case["expected_tool"])

        # Groundedness label (skippable to save API calls).
        label = "-"
        if not args.no_judge and not run["error"]:
            label = judge_groundedness(case["question"], run["answer"])

        rows.append({
            "id": case["id"],
            "question": case["question"],
            "in_notes": case["in_notes"],
            "expected": case["expected_tool"],
            "tools": run["tools"],
            "last_tool": last_tool,
            "routing_ok": routing_ok,
            "label": label,
            "error": run["error"],
        })

        mark = "OK " if routing_ok else "XX "
        traj = " -> ".join(run["tools"]) or "(no tool)"
        print(f"[{mark}] #{case['id']:>2}  exp={case['expected_tool']:<17} got={traj:<40} judge={label}")
        if run["error"]:
            print(f"         error: {run['error']}")

    _report(rows, judged=not args.no_judge)


# --- Metrics + report -----------------------------------------------------

def _pct(n, d):
    return f"{(100.0 * n / d):.0f}%" if d else "n/a"


def _report(rows, judged: bool):
    total = len(rows)
    routing_ok = sum(r["routing_ok"] for r in rows)

    in_notes = [r for r in rows if r["in_notes"]]
    out_notes = [r for r in rows if not r["in_notes"]]

    # Groundedness (in-notes cases): answer should be supported by the notes.
    grounded = [r for r in in_notes if r["label"] in ("GROUNDED", "PARTIAL")]
    # Honest fallback (out-of-notes cases): agent must go to the web or refuse —
    # never fabricate a notes-grounded answer.
    honest = [r for r in out_notes if r["last_tool"] == "search_web" or r["label"] == "REFUSED"]

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"Routing accuracy         : {_pct(routing_ok, total)}  ({routing_ok}/{total})")
    if judged:
        print(f"Groundedness (in-notes)  : {_pct(len(grounded), len(in_notes))}  "
              f"({len(grounded)}/{len(in_notes)})")
    print(f"Honest web fallback      : {_pct(len(honest), len(out_notes))}  "
          f"({len(honest)}/{len(out_notes)})")

    _write_markdown(rows, judged, total, routing_ok, in_notes, out_notes, grounded, honest)
    print(f"\nWrote detailed report -> {os.path.relpath(RESULTS_MD, HERE)}")


def _write_markdown(rows, judged, total, routing_ok, in_notes, out_notes, grounded, honest):
    lines = []
    lines.append("# Evaluation results\n")
    lines.append(
        "Generated by `python evaluate.py` against `eval/eval_notes.txt` and the "
        f"{total}-case set in `eval/eval_set.json`. Numbers are reproducible "
        "because the corpus and questions are fixed.\n"
    )
    lines.append("## Metrics\n")
    lines.append("| Metric | Score | Notes |")
    lines.append("|--------|-------|-------|")
    lines.append(f"| Routing accuracy | **{_pct(routing_ok, total)}** ({routing_ok}/{total}) | "
                 "did the agent pick the right tool (incl. web fallback) |")
    if judged:
        lines.append(f"| Groundedness (in-notes) | **{_pct(len(grounded), len(in_notes))}** "
                     f"({len(grounded)}/{len(in_notes)}) | answer supported by retrieved notes (LLM judge) |")
    lines.append(f"| Honest web fallback | **{_pct(len(honest), len(out_notes))}** "
                 f"({len(honest)}/{len(out_notes)}) | out-of-notes Qs went to the web or refused, never fabricated |")
    lines.append("")
    lines.append("## Per-case detail\n")
    lines.append("| # | Question | In notes? | Expected | Tools used | Routing | Judge |")
    lines.append("|---|----------|-----------|----------|------------|---------|-------|")
    for r in rows:
        traj = " → ".join(r["tools"]) or "(none)"
        routing = "✅" if r["routing_ok"] else "❌"
        q = r["question"].replace("|", "\\|")
        lines.append(f"| {r['id']} | {q} | {'yes' if r['in_notes'] else 'no'} | "
                     f"`{r['expected']}` | {traj} | {routing} | {r['label']} |")
    lines.append("")
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
