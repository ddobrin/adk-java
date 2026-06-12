/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.adk.planner.llmcompiler;

import com.google.adk.planner.goap.AgentMetadata;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Builds the prompt for the Plan-and-Execute / LLMCompiler planner.
 *
 * <p>Unlike a supervisor (which is re-prompted every step), this planner is asked <em>once</em> to
 * emit the entire execution graph as JSON: a list of tasks, each declaring the agent to run and the
 * ids of the tasks it depends on. The agent catalog is rendered from {@link AgentMetadata} so the
 * prompt and the runtime {@link PlanValidator} share a single source of truth for dependencies.
 */
public final class LlmCompilerInstructions {

  private LlmCompilerInstructions() {}

  /**
   * Builds the full planning prompt.
   *
   * @param instruction the user's natural-language goal
   * @param catalog the available agents and their input/output contracts
   * @param replanFeedback when non-blank, guidance from a Joiner to revise the previous plan
   * @param validationErrors when non-empty, the reasons a prior attempt's plan was rejected (drives
   *     one round of self-repair)
   */
  public static String build(
      String instruction,
      List<AgentMetadata> catalog,
      String replanFeedback,
      List<String> validationErrors) {
    StringBuilder sb = new StringBuilder();
    sb.append(
        """
        You are a planner. Produce a COMPLETE execution plan in ONE response: a directed acyclic
        graph (DAG) of agent tasks. A deterministic engine will execute your plan exactly as
        written — tasks whose dependencies have all completed run IN PARALLEL.

        AGENT CATALOG (use exact names; "Requires"/"Produces" are state keys):
        """);
    sb.append(renderCatalog(catalog));
    sb.append(
        """

        RULES:
        1. Every task's dependsOn MUST collectively produce all of that agent's "Requires" keys
           (a key already present in the initial state needs no producer).
        2. Choose only the agents the user's intent needs — do not pad the plan.
        3. Maximize parallelism: tasks that do not depend on each other should share their upstream
           dependencies rather than being chained.
        4. The graph must be acyclic and every dependsOn id must reference a task in this plan.

        Respond with STRICT JSON only (no prose, no markdown fences) of the form:
        {
          "label": "<short plan name>",
          "tasks": [
            {"id": 1, "agent": "<agent_name>", "dependsOn": []},
            {"id": 2, "agent": "<agent_name>", "dependsOn": [1]}
          ]
        }
        """);

    if (validationErrors != null && !validationErrors.isEmpty()) {
      sb.append(
          "\nYour PREVIOUS plan was REJECTED. Fix these errors and return a corrected plan:\n");
      for (String error : validationErrors) {
        sb.append("- ").append(error).append("\n");
      }
    }

    if (replanFeedback != null && !replanFeedback.isBlank()) {
      sb.append("\nREPLAN FEEDBACK (revise the previous plan accordingly):\n")
          .append(replanFeedback)
          .append("\n");
    }

    sb.append("\nUSER INSTRUCTION: ").append(instruction).append("\n");
    return sb.toString();
  }

  /** Renders the catalog as a "- name: Requires X, Y. Produces Z." listing. */
  static String renderCatalog(List<AgentMetadata> catalog) {
    return catalog.stream()
        .map(LlmCompilerInstructions::renderAgent)
        .collect(Collectors.joining("\n"));
  }

  private static String renderAgent(AgentMetadata meta) {
    String requires =
        meta.inputKeys().isEmpty() ? "(nothing)" : String.join(", ", meta.inputKeys());
    return "- "
        + meta.agentName()
        + ": Requires "
        + requires
        + ". Produces "
        + meta.outputKey()
        + ".";
  }
}
