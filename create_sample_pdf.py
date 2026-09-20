from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

OUT = Path("data/Agentic_AI_eBook.pdf")

CHAPTERS = {
    "Chapter 1: What is an AI Agent?": """An AI agent is a system built around a large language model (LLM) that can perceive,
reason, plan, and act toward a goal using tools. A plain LLM chatbot only predicts the next
token in a conversation. An agent, by contrast, runs a sense-think-act loop: it observes state,
decomposes a goal into steps, calls tools such as search, code execution, or APIs, observes
results, and adapts. Autonomy, tool use, memory, and goal-directedness separate agents from chatbots.""",
    "Chapter 2: Core Components of Agentic AI Systems": """Every agentic system has five core components. (1) The model: the LLM doing reasoning.
(2) Tools: functions the agent can call, e.g. web search, calculators, database queries, code runners.
(3) Memory: short-term context window plus long-term vector or episodic memory for past runs.
(4) Planner: the strategy that turns goals into steps, from simple prompting to ReAct or plan-and-execute.
(5) Guardrails and evaluators: input/output filters, policy checks, and scoring used to keep agents safe.""",
    "Chapter 3: Tool Use": """Tool use grounds agents in the real world. A tool is described with a name, a JSON schema
of arguments, and a docstring; the model emits a structured tool call, the harness executes it, and
the observation is fed back into context. Examples: search_tool(query) for fresh facts, python_repl(code)
for computation, sql_query(sql) for databases, and send_email(to, subject, body) for actions. Good tool
design means narrow scope, validated inputs, timeouts, and human approval for irreversible actions.""",
    "Chapter 4: The ReAct Pattern": """ReAct (Reason + Act) interleaves Thought, Action, and Observation steps. The agent writes a
thought about what to do next, takes one tool action, reads the observation, and repeats until it can
answer. Use ReAct when tasks need dynamic information gathering or multi-hop reasoning, such as research
or debugging. For rigid, repeatable workflows, prefer plan-and-execute or a fixed pipeline instead,
because ReAct costs more tokens and can loop without a step budget.""",
    "Chapter 5: Planning and Reflection Loops": """Planning breaks a goal into subtasks before acting; reflection critiques completed work and
retries. A planner agent may output a numbered plan, worker agents execute steps, and a reflector scores
outputs against rubrics, requesting fixes. Self-refine and Reflexion papers show that 1-3 reflection rounds
raise success rates on coding and writing tasks. Cap iterations, keep best-of-N checkpoints, and log every
revision so failures are auditable.""",
    "Chapter 6: Failure Modes and Evaluation": """Common failure modes: hallucinated tool arguments, infinite loops, context overflow,
brittle prompts, and unsafe actions. Evaluate agents with task success rate, steps-to-completion, tool
precision/recall, faithfulness of answers to retrieved context, and cost per task. Always test on held-out
tasks, add regression suites for tools, and require human review before production actions like payments
or data deletion.""",
}

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Agentic AI eBook (Demo Edition)", styles["Title"]),
        Spacer(1, 12),
        Paragraph("A compact demo knowledge base for the RAG chatbot. "
                   "Replace this file with the full eBook PDF for production use.", styles["Normal"]),
    ]
    for title, body in CHAPTERS.items():
        story.append(PageBreak())
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 12))
        text = body.replace("\n", " ")
        story.append(Paragraph(text, styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Key takeaway: " + text[:300], styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(text, styles["Normal"]))
    doc.build(story)
    print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
