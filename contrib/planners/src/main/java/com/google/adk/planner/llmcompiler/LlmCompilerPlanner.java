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

import com.google.adk.agents.BaseAgent;
import com.google.adk.agents.Planner;
import com.google.adk.agents.PlannerAction;
import com.google.adk.agents.PlanningContext;
import com.google.adk.models.BaseLlm;
import com.google.adk.planner.goap.AgentMetadata;
import com.google.adk.planner.goap.DfsSearchStrategy;
import com.google.adk.planner.goap.SearchStrategy;
import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import io.reactivex.rxjava3.core.Single;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@link Planner} that realizes the <b>Plan-and-Execute / LLMCompiler</b> orchestration shape: a
 * planning LLM authors an entire task DAG in one shot, a deterministic engine levels it into
 * parallel execution groups, and a {@link Joiner} inspects the results to decide whether to finish
 * or compile-and-run another plan.
 *
 * <p>How it maps onto the {@link PlannerAgent} loop:
 *
 * <ul>
 *   <li>{@link #init} captures the instruction (from the user content, else {@code
 *       defaultInstruction}) and resets loop state.
 *   <li>{@link #firstAction} compiles the first plan ({@link LlmPlanCompiler}) and returns its
 *       first execution group as a {@link PlannerAction.RunAgents} (multiple agents → run in
 *       parallel).
 *   <li>{@link #nextAction} walks the remaining groups; once the plan is exhausted it consults the
 *       {@link Joiner}. {@link JoinDecision.Finish} ends the run with the result; {@link
 *       JoinDecision.Replan} recompiles with the joiner's feedback (bounded by {@code maxReplans}).
 * </ul>
 *
 * <p><b>Adaptive replanning.</b> Unlike {@link com.google.adk.planner.goap.GoalOrientedPlanner},
 * which replans on <em>structural</em> failure (a missing declared output), this planner replans on
 * the {@code Joiner}'s <em>semantic</em> judgment — the outputs exist but are not yet good enough.
 *
 * <p><b>Robustness.</b> Plan compilation self-repairs once (an invalid plan is re-prompted with its
 * validation errors) and then falls back to a deterministic GOAP search, so a run never hard-fails
 * on a bad plan. See {@link LlmPlanCompiler}.
 *
 * <p>This planner holds mutable loop state and, like the other planners in this package, is
 * intended for use within a single reactive pipeline; it is not thread-safe.
 */
public final class LlmCompilerPlanner implements Planner {

  private static final Logger logger = LoggerFactory.getLogger(LlmCompilerPlanner.class);

  /** Default replan ceiling when none is configured. */
  public static final int DEFAULT_MAX_REPLANS = 2;

  private final LlmPlanCompiler compiler;
  private final Joiner joiner;
  private final int maxReplans;
  private final String defaultInstruction;

  // Mutable loop state — see class javadoc (single-pipeline, not thread-safe).
  private ImmutableList<ImmutableList<String>> groups;
  private String planLabel;
  private int cursor;
  private int replanCount;
  private String instruction;

  /**
   * @param compiler compiles an instruction into leveled execution groups
   * @param joiner decides finish-vs-replan once a plan is exhausted
   * @param maxReplans maximum number of replans before the run is forced to finish
   * @param defaultInstruction used when the invocation carries no user content
   */
  public LlmCompilerPlanner(
      LlmPlanCompiler compiler, Joiner joiner, int maxReplans, String defaultInstruction) {
    this.compiler = compiler;
    this.joiner = joiner;
    this.maxReplans = maxReplans;
    this.defaultInstruction = defaultInstruction;
  }

  @Override
  public void init(PlanningContext context) {
    this.instruction = resolveInstruction(context);
    this.groups = null;
    this.planLabel = null;
    this.cursor = 0;
    this.replanCount = 0;
  }

  @Override
  public Single<PlannerAction> firstAction(PlanningContext context) {
    return compiler
        .compile(instruction, null, stateKeys(context))
        .map(
            plan -> {
              applyPlan(plan);
              return firstRunnableOrDone(context);
            });
  }

  @Override
  public Single<PlannerAction> nextAction(PlanningContext context) {
    ImmutableList<BaseAgent> group = nextRunnableGroup(context);
    if (!group.isEmpty()) {
      return Single.just(new PlannerAction.RunAgents(group));
    }
    // Plan exhausted — let the joiner decide whether to finish or run another plan.
    return join(context);
  }

  /** Asks the joiner to finish or replan, recompiling (bounded by {@code maxReplans}) on replan. */
  private Single<PlannerAction> join(PlanningContext context) {
    return joiner
        .decide(context, replanCount)
        .flatMap(
            decision -> {
              if (decision instanceof JoinDecision.Finish finish) {
                logger.info("LLMCompiler joiner: FINISH after {} replan(s)", replanCount);
                return Single.<PlannerAction>just(
                    new PlannerAction.DoneWithResult(finish.result()));
              }
              JoinDecision.Replan replan = (JoinDecision.Replan) decision;
              if (replanCount >= maxReplans) {
                String message = "Stopped: max replan attempts (" + maxReplans + ") exhausted";
                logger.info("LLMCompiler joiner: REPLAN requested but {}", message);
                return Single.<PlannerAction>just(new PlannerAction.DoneWithResult(message));
              }
              replanCount++;
              logger.info(
                  "LLMCompiler joiner: REPLAN (attempt {}/{}) with feedback: {}",
                  replanCount,
                  maxReplans,
                  replan.feedback());
              return compiler
                  .compile(instruction, replan.feedback(), stateKeys(context))
                  .map(
                      plan -> {
                        applyPlan(plan);
                        return firstRunnableOrDone(context);
                      });
            });
  }

  /** After a (re)compile: run the new plan's first runnable group, or finish if it is empty. */
  private PlannerAction firstRunnableOrDone(PlanningContext context) {
    ImmutableList<BaseAgent> group = nextRunnableGroup(context);
    if (group.isEmpty()) {
      return new PlannerAction.DoneWithResult(emptyPlanMessage());
    }
    return new PlannerAction.RunAgents(group);
  }

  /**
   * Advances {@link #cursor} to the next group that resolves to at least one known agent and
   * returns it (resolved to {@link BaseAgent}s). Groups whose names are all unknown are skipped
   * with a warning. Returns an empty list once the plan is exhausted.
   */
  private ImmutableList<BaseAgent> nextRunnableGroup(PlanningContext context) {
    while (groups != null && cursor < groups.size()) {
      ImmutableList<String> names = groups.get(cursor++);
      ImmutableList<BaseAgent> resolved = resolve(names, context);
      if (!resolved.isEmpty()) {
        return resolved;
      }
      logger.warn("LLMCompiler: no known agents in group {}; skipping", names);
    }
    return ImmutableList.of();
  }

  private ImmutableList<BaseAgent> resolve(ImmutableList<String> names, PlanningContext context) {
    ImmutableList.Builder<BaseAgent> builder = ImmutableList.builder();
    for (String name : names) {
      try {
        builder.add(context.findAgent(name));
      } catch (IllegalArgumentException e) {
        logger.warn("LLMCompiler: unknown agent '{}' in plan; skipping", name);
      }
    }
    return builder.build();
  }

  private void applyPlan(LlmPlanCompiler.CompiledPlan plan) {
    this.groups = plan.groups();
    this.planLabel = plan.label();
    this.cursor = 0;
    logger.info(
        "LLMCompiler plan applied: label='{}', {} group(s), fromLlm={}",
        plan.label(),
        plan.groups().size(),
        plan.fromLlm());
  }

  private String resolveInstruction(PlanningContext context) {
    String fromUser = context.userContent().map(Content::text).orElse(null);
    if (fromUser != null && !fromUser.isBlank()) {
      return fromUser;
    }
    return defaultInstruction;
  }

  private String emptyPlanMessage() {
    return "No executable plan was produced"
        + (planLabel == null || planLabel.isBlank() ? "" : " (" + planLabel + ")");
  }

  private static Set<String> stateKeys(PlanningContext context) {
    return new HashSet<>(context.state().keySet());
  }

  // ---------------------------------------------------------------------------------------------
  // Builder
  // ---------------------------------------------------------------------------------------------

  /**
   * Starts building a planner whose compiler is constructed from {@code llm} and {@code catalog}.
   *
   * @param llm the planning LLM that authors the task DAG
   * @param catalog the available agents and their input/output contracts
   */
  public static Builder builder(BaseLlm llm, List<AgentMetadata> catalog) {
    return new Builder(llm, catalog);
  }

  /**
   * Starts building a planner around a pre-constructed {@link LlmPlanCompiler} (advanced; use when
   * you want full control over the compiler, e.g. a custom fallback strategy or a test double).
   */
  public static Builder builder(LlmPlanCompiler compiler) {
    return new Builder(compiler);
  }

  /** Fluent builder for {@link LlmCompilerPlanner}. */
  public static final class Builder {
    private final BaseLlm llm;
    private final List<AgentMetadata> catalog;
    private LlmPlanCompiler compiler;
    private String fallbackGoal;
    private SearchStrategy fallbackStrategy = new DfsSearchStrategy();
    private Joiner joiner;
    private int maxReplans = DEFAULT_MAX_REPLANS;
    private String defaultInstruction = "";

    private Builder(BaseLlm llm, List<AgentMetadata> catalog) {
      this.llm = llm;
      this.catalog = catalog;
    }

    private Builder(LlmPlanCompiler compiler) {
      this.llm = null;
      this.catalog = null;
      this.compiler = compiler;
    }

    /**
     * Target output key for the deterministic GOAP fallback (ignored if a compiler was supplied).
     */
    public Builder fallbackGoal(String fallbackGoal) {
      this.fallbackGoal = fallbackGoal;
      return this;
    }

    /** Search strategy for the GOAP fallback (ignored if a compiler was supplied). */
    public Builder fallbackStrategy(SearchStrategy fallbackStrategy) {
      this.fallbackStrategy = fallbackStrategy;
      return this;
    }

    /**
     * The joiner that decides finish-vs-replan. Defaults to finish-after-one-plan (no replanning).
     */
    public Builder joiner(Joiner joiner) {
      this.joiner = joiner;
      return this;
    }

    /**
     * Maximum number of replans before the run is forced to finish (default {@value
     * LlmCompilerPlanner#DEFAULT_MAX_REPLANS}).
     */
    public Builder maxReplans(int maxReplans) {
      this.maxReplans = maxReplans;
      return this;
    }

    /** Instruction used when the invocation carries no user content. */
    public Builder defaultInstruction(String defaultInstruction) {
      this.defaultInstruction = defaultInstruction;
      return this;
    }

    public LlmCompilerPlanner build() {
      LlmPlanCompiler resolvedCompiler = compiler;
      if (resolvedCompiler == null) {
        resolvedCompiler = new LlmPlanCompiler(llm, catalog, fallbackGoal, fallbackStrategy);
      }
      Joiner resolvedJoiner = (joiner != null) ? joiner : finishImmediately();
      return new LlmCompilerPlanner(
          resolvedCompiler, resolvedJoiner, maxReplans, defaultInstruction);
    }

    /** Default joiner: a single plan runs, then the loop finishes — no replanning. */
    private static Joiner finishImmediately() {
      return (context, replanCount) -> Single.just(new JoinDecision.Finish("Plan complete"));
    }
  }
}
