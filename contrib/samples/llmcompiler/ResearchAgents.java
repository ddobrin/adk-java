// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.example.llmcompiler;

import com.google.adk.agents.LlmAgent;
import com.google.adk.planner.goap.AgentMetadata;
import com.google.common.collect.ImmutableList;
import java.util.List;

/**
 * The agent catalog for the LLMCompiler research-fan-out sample.
 *
 * <p>Four agents form a small research DAG:
 *
 * <pre>
 *                gather  (topic        -> raw)
 *               /      \
 *     summarizeA        summarizeB     (raw -> sumA / sumB, run in parallel)
 *               \      /
 *              synthesize (sumA, sumB  -> report)
 * </pre>
 *
 * <p>Each agent declares an {@link AgentMetadata} input/output contract. The contract is what the
 * planning LLM sees (rendered as a catalog) when it authors the task DAG, and what {@code
 * PlanValidator} checks the authored plan against. The agents read their inputs via {@code {key}}
 * instruction templating from session state and write their output via {@code outputKey}, so the
 * declared contracts and the runtime data flow stay in lock-step.
 */
public final class ResearchAgents {

  private ResearchAgents() {}

  /** Planning + execution model. Picks up credentials from the environment (see README). */
  public static final String MODEL = "gemini-2.0-flash";

  /** Collects raw research notes about the topic. Level 0 — depends only on the seeded {@code topic}. */
  public static final LlmAgent GATHER =
      LlmAgent.builder()
          .name("gather")
          .description("Gathers raw research notes about the topic.")
          .model(MODEL)
          .instruction(
              """
              You are a research gatherer. The topic is:

              {topic}

              Produce a dense, factual set of research notes about this topic: key facts, figures,
              named entities, dates, and any notable debates. Write notes, not prose. Do not
              summarize or conclude — that is a later step.
              """)
          .outputKey("raw")
          .build();

  /** Summarizes the established, well-supported findings. Level 1 — depends on {@code raw}. */
  public static final LlmAgent SUMMARIZE_A =
      LlmAgent.builder()
          .name("summarizeA")
          .description("Summarizes the established, well-supported findings from the notes.")
          .model(MODEL)
          .instruction(
              """
              You distill research notes into the *established* picture. Given these notes:

              {raw?}

              Write a concise summary (5-8 bullet points) of what is well-supported and broadly
              agreed upon. Focus on consensus and solid evidence.
              """)
          .outputKey("sumA")
          .build();

  /** Surfaces open questions and counterpoints. Level 1 — depends on {@code raw}, parallel to A. */
  public static final LlmAgent SUMMARIZE_B =
      LlmAgent.builder()
          .name("summarizeB")
          .description("Surfaces open questions, tensions, and counterpoints from the notes.")
          .model(MODEL)
          .instruction(
              """
              You distill research notes into the *contested* picture. Given these notes:

              {raw?}

              Write a concise summary (5-8 bullet points) of open questions, disagreements,
              uncertainties, and counterpoints. Focus on what is unsettled or debated.
              """)
          .outputKey("sumB")
          .build();

  /** Synthesizes both summaries into a final report. Level 2 — depends on {@code sumA} and {@code sumB}. */
  public static final LlmAgent SYNTHESIZE =
      LlmAgent.builder()
          .name("synthesize")
          .description("Synthesizes the established and contested summaries into a final report.")
          .model(MODEL)
          .instruction(
              """
              You are a research synthesizer. Combine the two summaries below into a single,
              balanced report with a short conclusion.

              Established findings:
              {sumA?}

              Open questions and counterpoints:
              {sumB?}

              Produce a well-structured report: a brief intro, the established picture, the open
              questions, and a 2-3 sentence conclusion that weighs them.
              """)
          .outputKey("report")
          .build();

  /** All sub-agents, in declaration order. */
  public static final ImmutableList<LlmAgent> AGENTS =
      ImmutableList.of(GATHER, SUMMARIZE_A, SUMMARIZE_B, SYNTHESIZE);

  /**
   * The input/output contracts the planning LLM plans over. {@code agentName} must match each
   * {@link LlmAgent#name()}; {@code inputKeys}/{@code outputKey} must match the instruction
   * placeholders and {@code outputKey} above.
   */
  public static final List<AgentMetadata> CATALOG =
      List.of(
          new AgentMetadata("gather", ImmutableList.of("topic"), "raw"),
          new AgentMetadata("summarizeA", ImmutableList.of("raw"), "sumA"),
          new AgentMetadata("summarizeB", ImmutableList.of("raw"), "sumB"),
          new AgentMetadata("synthesize", ImmutableList.of("sumA", "sumB"), "report"));

  /** The output key the deterministic GOAP fallback should target if LLM planning fails. */
  public static final String GOAL = "report";
}
