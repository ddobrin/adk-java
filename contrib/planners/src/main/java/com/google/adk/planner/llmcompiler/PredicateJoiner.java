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

import com.google.adk.agents.PlanningContext;
import io.reactivex.rxjava3.core.Single;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * A deterministic {@link Joiner} that decides finish-vs-replan from the session state alone — no
 * LLM call. The completion test and the messages it produces are supplied by the caller, so the
 * judgment ("when is the deliverable good enough, and what should the next pass do differently?")
 * lives in plain, testable code rather than a prompt.
 *
 * <p>This is the deterministic counterpart to {@link LlmJoiner}: same {@link JoinDecision}
 * contract, but the verdict is a pure function of {@code context.state()}. Use it when completion
 * is something you can check directly (a key is present, a score clears a threshold, a count is
 * reached).
 *
 * <pre>{@code
 * // Finish once "synthesis" has been written; otherwise ask for another pass.
 * Joiner joiner = new PredicateJoiner(
 *     state -> state.containsKey("synthesis"),
 *     state -> (String) state.get("synthesis"),
 *     state -> "No synthesis produced yet — add a synthesis step.");
 * }</pre>
 */
public final class PredicateJoiner implements Joiner {

  private final Predicate<Map<String, Object>> isComplete;
  private final Function<Map<String, Object>, String> finishResult;
  private final Function<Map<String, Object>, String> replanFeedback;

  /**
   * @param isComplete returns true when the deliverable is ready (given the current state)
   * @param finishResult builds the final result text from the state, used when {@code isComplete}
   * @param replanFeedback builds the guidance handed back to the planner, used otherwise
   */
  public PredicateJoiner(
      Predicate<Map<String, Object>> isComplete,
      Function<Map<String, Object>, String> finishResult,
      Function<Map<String, Object>, String> replanFeedback) {
    this.isComplete = isComplete;
    this.finishResult = finishResult;
    this.replanFeedback = replanFeedback;
  }

  /**
   * Convenience constructor with fixed finish/replan messages (no state interpolation).
   *
   * @param isComplete returns true when the deliverable is ready
   * @param finishResult the final result text when complete
   * @param replanFeedback the guidance handed back to the planner otherwise
   */
  public PredicateJoiner(
      Predicate<Map<String, Object>> isComplete, String finishResult, String replanFeedback) {
    this(isComplete, state -> finishResult, state -> replanFeedback);
  }

  @Override
  public Single<JoinDecision> decide(PlanningContext context, int replanCount) {
    Map<String, Object> state = context.state();
    if (isComplete.test(state)) {
      return Single.just(new JoinDecision.Finish(finishResult.apply(state)));
    }
    return Single.just(new JoinDecision.Replan(replanFeedback.apply(state)));
  }
}
