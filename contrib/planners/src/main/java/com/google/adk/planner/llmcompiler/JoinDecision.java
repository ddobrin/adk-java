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

/**
 * The verdict a {@link Joiner} returns after a compiled plan has finished executing: either the
 * work is complete, or the planner should compile and run another plan.
 *
 * <p>This is the decision point that makes the Plan-and-Execute / LLMCompiler loop
 * <em>adaptive</em>. It is intentionally a two-case sealed type so the {@link LlmCompilerPlanner}
 * can {@code switch} over it exhaustively.
 */
public sealed interface JoinDecision permits JoinDecision.Finish, JoinDecision.Replan {

  /**
   * Stop the loop — the deliverable is ready.
   *
   * @param result the final text result surfaced to the caller (as a {@code DoneWithResult})
   */
  record Finish(String result) implements JoinDecision {}

  /**
   * Run another planning pass — the results so far are not sufficient.
   *
   * @param feedback guidance handed back to the planner LLM to bias the next plan (e.g. "the
   *     council disagreed; add a disagreement-analysis step")
   */
  record Replan(String feedback) implements JoinDecision {}
}
