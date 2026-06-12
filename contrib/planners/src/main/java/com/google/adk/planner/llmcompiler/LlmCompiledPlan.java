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

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

/**
 * A plan authored by the planning LLM in one shot (the "Plan-and-Execute / LLMCompiler" plan).
 * Unlike GOAP — which <em>derives</em> the dependency graph from each agent's declared
 * inputs/outputs — the LLM emits the task graph <em>explicitly</em>, declaring for each task which
 * earlier tasks it depends on.
 *
 * <p>This is the on-the-wire shape the LLM is asked to return as JSON, e.g.
 *
 * <pre>{@code
 * {
 *   "label": "Full Deliberation",
 *   "tasks": [
 *     {"id": 1, "agent": "initial_response",       "dependsOn": []},
 *     {"id": 2, "agent": "peer_ranking",           "dependsOn": [1]},
 *     {"id": 3, "agent": "agreement_analysis",     "dependsOn": [1]},
 *     {"id": 4, "agent": "disagreement_analysis",  "dependsOn": [1]},
 *     {"id": 5, "agent": "final_synthesis",        "dependsOn": [1, 2]},
 *     {"id": 6, "agent": "aggregate_rankings",     "dependsOn": [2]},
 *     {"id": 7, "agent": "aggregate_agreements",   "dependsOn": [3]},
 *     {"id": 8, "agent": "aggregate_disagreements","dependsOn": [4]}
 *   ]
 * }
 * }</pre>
 *
 * <p>The plan is validated against the {@code AgentMetadata} catalog (see {@link PlanValidator})
 * before execution. The validated DAG is grouped into parallel execution levels by {@link
 * DagLeveler}.
 *
 * @param label short human-readable label for the plan (e.g. "Full Deliberation")
 * @param tasks the explicit task DAG
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record LlmCompiledPlan(String label, List<PlanTask> tasks) {

  /**
   * A single node in the LLM-authored DAG.
   *
   * @param id unique, stable identifier within this plan (referenced by {@code dependsOn})
   * @param agent agent name (must exist in the {@code AgentMetadata} catalog)
   * @param dependsOn ids of tasks that must complete before this task runs (the explicit edges)
   */
  @JsonIgnoreProperties(ignoreUnknown = true)
  public record PlanTask(int id, String agent, List<Integer> dependsOn) {
    public PlanTask {
      dependsOn = dependsOn == null ? List.of() : List.copyOf(dependsOn);
    }
  }
}
