# LLMCompiler (Plan-and-Execute) Sample

This sample demonstrates the **Plan-and-Execute / LLMCompiler** planner from
`google-adk-planners`. A planning LLM authors an entire task DAG in one shot, the
planner levels it into parallel execution groups and runs them, and a **Joiner**
decides whether the result is good enough or another plan should run.

See the planner's full documentation in
[`../../planners/README.md` §6](../../planners/README.md#6-plan-and-execute--llmcompiler).

## What it does

A small research fan-out over four agents:

```
                gather  (topic        -> raw)
               /      \
     summarizeA        summarizeB     (raw -> sumA / sumB, run in PARALLEL)
               \      /
              synthesize (sumA, sumB  -> report)
```

1. The planning LLM is shown the agent catalog (each agent's `Requires` / `Produces`
   contract) and authors a JSON DAG with explicit `dependsOn` edges.
2. `LlmCompilerPlanner` levels the DAG: `gather` runs first, then `summarizeA` and
   `summarizeB` run **in parallel**, then `synthesize` combines them.
3. Once the plan is exhausted, `ResearchJoiner` (a deterministic `PredicateJoiner`)
   inspects session state and returns `Finish` (ship the report) or `Replan` (run
   another pass with feedback), bounded by `maxReplans`.

If the LLM ever authors an invalid plan, the compiler self-repairs once (re-prompting
with the validation errors) and then falls back to a deterministic GOAP search — so a
run never hard-fails on a bad plan.

## Project Layout

```
├── ResearchAgents.java           // The 4 LlmAgent sub-agents + the AgentMetadata catalog
├── ResearchJoiner.java           // The finish-vs-replan heuristic (the contribution point)
├── LlmCompilerResearchRun.java   // Console runner entry point
├── pom.xml                       // Maven config; depends on google-adk + google-adk-planners
└── README.md                     // This file
```

## Prerequisites

- Java 17+
- Maven 3.9+
- Gemini credentials in the environment (same as the `helloworld` sample). Either:
  - **Gemini API key:** `export GOOGLE_API_KEY=...` (and optionally
    `export GOOGLE_GENAI_USE_VERTEXAI=false`), or
  - **Vertex AI:** `export GOOGLE_GENAI_USE_VERTEXAI=true`,
    `export GOOGLE_CLOUD_PROJECT=...`, `export GOOGLE_CLOUD_LOCATION=...`, with
    application-default credentials configured.

## Build and Run

```bash
# from the repo root — builds the planners library the sample depends on
mvn -q -pl contrib/planners -am -DskipTests install
mvn -q -pl contrib/samples/llmcompiler -am package

# run with the default topic
mvn -q -pl contrib/samples/llmcompiler exec:java

# or pass your own topic
mvn -q -pl contrib/samples/llmcompiler exec:java -Dexec.args="impact of caching on LLM inference cost"
```

The runner seeds the topic into session state, issues the instruction, and prints the
streamed events (each agent's output, then the synthesized report).

## The contribution point: the replan-vs-finish heuristic

The deterministic mechanics — DAG leveling, parallel execution, self-repair, GOAP
fallback, and replan bounding — are proven by the planner module's test suite. The one
genuinely judgment-driven choice is **when the deliverable is good enough**, and that
lives in `ResearchJoiner.java`. Rather than a single length check, it tests four
independent quality signals and ships only when **all** hold:

```java
static boolean isReportComplete(Map<String, Object> state) {
    String report = asText(state.get("report"));
    if (report == null) return false;          // 1. existence — synthesize wrote a report
    String body = report.strip();
    String haystack = body.toLowerCase(Locale.ROOT);
    return body.length() >= MIN_REPORT_CHARS    // 2. substance — not a stub
        && mentionsAny(haystack, ESTABLISHED_MARKERS)   // 3a. the established angle survived
        && mentionsAny(haystack, OPEN_QUESTION_MARKERS) // 3b. the contested angle survived
        && mentionsAny(haystack, CONCLUSION_MARKERS);   // 4. closure — it actually concludes
}
```

`replanFeedback(state)` reads the **same four signals** and names only the ones that
failed, so the next plan targets the real gap (e.g. *"open questions are missing — keep
'summarizeB' in the plan"*) instead of retrying blindly.

Tune it to your own definition of "done": adjust `MIN_REPORT_CHARS` and the `*_MARKERS`
sets, or — more robust than a keyword scan — have an upstream agent write a structured
quality marker into state and check that instead. A `PredicateJoiner` sees session state
only (the planner already bounds retries via `maxReplans`), so you decide *whether* the
work is done, not *how many times* to try. For an LLM-based "good enough" judgment
instead, swap in `new LlmJoiner(llm, "produce a balanced report")`.

## Next Steps

- Read [`../../planners/README.md` §6](../../planners/README.md#6-plan-and-execute--llmcompiler)
  for the full planner design (semantic vs structural replanning, the self-repair →
  GOAP-fallback safety net, and the GOAP-vs-LLMCompiler comparison).
- Compare with the GOAP planner (§4), which *derives* the same kind of leveled groups
  from I/O contracts instead of having the LLM author them.
