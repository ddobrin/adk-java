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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.planner.goap.AgentMetadata;
import com.google.adk.planner.goap.DfsSearchStrategy;
import com.google.adk.planner.goap.GoalOrientedSearchGraph;
import com.google.adk.planner.goap.SearchStrategy;
import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Single;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Produces an execution plan with a single planning-LLM call (the "Plan-and-Execute / LLMCompiler"
 * step). The LLM emits an explicit task DAG; this compiler parses it, validates it against the
 * {@link AgentMetadata} catalog, and — when the plan is sound — levels it into parallel execution
 * groups.
 *
 * <p>Robustness in two stages:
 *
 * <ol>
 *   <li><b>Self-repair</b> — if the first plan is malformed or invalid, the LLM is re-prompted once
 *       with the exact validation errors, asking it to fix them.
 *   <li><b>GOAP fallback</b> — if the repaired plan is still invalid (or the LLM call errors), the
 *       compiler falls back to a deterministic goal-oriented search over the same catalog, so a run
 *       always has a runnable plan.
 * </ol>
 */
public final class LlmPlanCompiler {

  private static final Logger logger = LoggerFactory.getLogger(LlmPlanCompiler.class);
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private final BaseLlm llm;
  private final List<AgentMetadata> catalog;
  private final Map<String, AgentMetadata> catalogByName;
  private final String fallbackGoal;
  private final SearchStrategy fallbackStrategy;

  /**
   * Creates a compiler with the default backward-chaining DFS fallback strategy.
   *
   * @param llm the planning LLM that authors the DAG
   * @param catalog the available agents and their input/output contracts
   * @param fallbackGoal the target output key used by the deterministic GOAP fallback (nullable; if
   *     null, the fallback yields an empty plan)
   */
  public LlmPlanCompiler(BaseLlm llm, List<AgentMetadata> catalog, String fallbackGoal) {
    this(llm, catalog, fallbackGoal, new DfsSearchStrategy());
  }

  public LlmPlanCompiler(
      BaseLlm llm,
      List<AgentMetadata> catalog,
      String fallbackGoal,
      SearchStrategy fallbackStrategy) {
    this.llm = llm;
    this.catalog = List.copyOf(catalog);
    this.catalogByName = new HashMap<>();
    for (AgentMetadata m : this.catalog) {
      this.catalogByName.put(m.agentName(), m);
    }
    this.fallbackGoal = fallbackGoal;
    this.fallbackStrategy = fallbackStrategy;
  }

  /**
   * The compiled, ready-to-execute plan.
   *
   * @param label human-readable plan label
   * @param groups ordered execution groups of agent names (agents within a group run in parallel)
   * @param fromLlm true if the LLM's plan was used; false if the GOAP fallback fired
   */
  public record CompiledPlan(
      String label, ImmutableList<ImmutableList<String>> groups, boolean fromLlm) {}

  /**
   * Compiles a plan for {@code instruction}.
   *
   * @param instruction the user's natural-language goal
   * @param replanFeedback nullable guidance from a Joiner biasing the LLM toward revising a prior
   *     plan
   * @param initialState state keys already present (satisfied preconditions for
   *     validation/fallback)
   */
  public Single<CompiledPlan> compile(
      String instruction, String replanFeedback, Set<String> initialState) {
    return callLlm(LlmCompilerInstructions.build(instruction, catalog, replanFeedback, null))
        .map(raw -> attempt(raw, initialState))
        .flatMap(
            first -> {
              if (first.ok()) {
                return Single.just(toCompiledPlan(first.plan()));
              }
              logger.info(
                  "LLMCompiler plan rejected ({} errors), attempting self-repair: {}",
                  first.errors().size(),
                  first.errors());
              String repairPrompt =
                  LlmCompilerInstructions.build(
                      instruction, catalog, replanFeedback, first.errors());
              return callLlm(repairPrompt)
                  .map(raw2 -> attempt(raw2, initialState))
                  .map(
                      second -> {
                        if (second.ok()) {
                          return toCompiledPlan(second.plan());
                        }
                        logger.warn(
                            "LLMCompiler self-repair still invalid ({} errors), falling back to "
                                + "GOAP: {}",
                            second.errors().size(),
                            second.errors());
                        return goapFallback(initialState);
                      });
            })
        .onErrorReturn(
            error -> {
              logger.warn("LLMCompiler planning failed, falling back to GOAP", error);
              return goapFallback(initialState);
            });
  }

  /** A single parse+validate attempt: either a valid plan or the reasons it was rejected. */
  private record Attempt(boolean ok, LlmCompiledPlan plan, List<String> errors) {}

  private Attempt attempt(String raw, Set<String> initialState) {
    LlmCompiledPlan plan;
    try {
      plan = parse(raw);
    } catch (Exception e) {
      return new Attempt(false, null, List.of("Malformed plan JSON: " + e.getMessage()));
    }
    PlanValidator.ValidationResult validation =
        PlanValidator.validate(plan, catalogByName, initialState);
    return new Attempt(validation.ok(), plan, validation.errors());
  }

  private CompiledPlan toCompiledPlan(LlmCompiledPlan plan) {
    ImmutableList<ImmutableList<String>> groups = DagLeveler.toGroups(plan);
    logger.info(
        "LLMCompiler plan accepted: label='{}', {} tasks -> {} groups",
        plan.label(),
        plan.tasks().size(),
        groups.size());
    return new CompiledPlan(planLabel(plan), groups, true);
  }

  /** Deterministic fallback: goal-oriented dependency search over the same catalog. */
  private CompiledPlan goapFallback(Set<String> initialState) {
    try {
      GoalOrientedSearchGraph graph = new GoalOrientedSearchGraph(catalog);
      ImmutableList<ImmutableList<String>> groups =
          fallbackStrategy.searchGrouped(graph, catalog, initialState, fallbackGoal);
      logger.info("LLMCompiler GOAP fallback produced {} groups", groups.size());
      return new CompiledPlan("GOAP fallback", groups, false);
    } catch (RuntimeException e) {
      logger.warn("LLMCompiler GOAP fallback failed; yielding empty plan", e);
      return new CompiledPlan("empty (fallback failed)", ImmutableList.of(), false);
    }
  }

  /** Issues a single planning call and returns the model's text. */
  private Single<String> callLlm(String prompt) {
    LlmRequest request =
        LlmRequest.builder()
            .contents(
                ImmutableList.of(
                    Content.builder().role("user").parts(Part.fromText(prompt)).build()))
            .build();
    return llm.generateContent(request, false).lastOrError().map(LlmPlanCompiler::extractText);
  }

  private static String extractText(LlmResponse response) {
    return response.content().flatMap(Content::parts).stream()
        .flatMap(List::stream)
        .flatMap(part -> part.text().stream())
        .collect(Collectors.joining())
        .trim();
  }

  /** Parses the model output into an {@link LlmCompiledPlan}, tolerating markdown fences. */
  LlmCompiledPlan parse(String raw) throws Exception {
    return MAPPER.readValue(extractJson(raw), LlmCompiledPlan.class);
  }

  /** Extracts the outermost JSON object, ignoring ```json fences or surrounding prose. */
  static String extractJson(String raw) {
    if (raw == null) {
      return "{}";
    }
    int start = raw.indexOf('{');
    int end = raw.lastIndexOf('}');
    if (start >= 0 && end > start) {
      return raw.substring(start, end + 1);
    }
    return raw.strip();
  }

  private static String planLabel(LlmCompiledPlan plan) {
    return (plan.label() == null || plan.label().isBlank()) ? "Compiled Plan" : plan.label();
  }
}
