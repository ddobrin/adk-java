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

/**
 * Inspects the world state and events produced by a finished plan and decides whether the work is
 * complete or another planning pass is warranted — the "join" step of the Plan-and-Execute /
 * LLMCompiler loop.
 *
 * <p>This is the pluggable seam for <em>semantic, results-driven replanning</em>. Two
 * implementations ship:
 *
 * <ul>
 *   <li>{@link LlmJoiner} — asks an LLM to judge the results and emit {@code FINISH}/{@code
 *       REPLAN}.
 *   <li>{@link PredicateJoiner} — a deterministic completion predicate over the session state, with
 *       no model call.
 * </ul>
 *
 * <p>Contrast with GOAP's {@code ReplanPolicy}, which replans on <em>structural</em> failure (an
 * agent did not produce its declared output key). A {@code Joiner} replans on judgment: the outputs
 * exist but are not yet sufficient.
 */
@FunctionalInterface
public interface Joiner {

  /**
   * Decides whether to finish or replan, given everything the run has produced so far.
   *
   * @param context the planning context (session state, events, available agents, user content)
   * @param replanCount how many replans have already happened this run (0 on the first join). Lets
   *     a joiner taper its strictness or stop asking for more passes as attempts accumulate.
   * @return a {@link JoinDecision.Finish} to stop, or a {@link JoinDecision.Replan} to run another
   *     plan
   */
  Single<JoinDecision> decide(PlanningContext context, int replanCount);
}
