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

import com.google.common.collect.ImmutableList;
import java.util.HashMap;
import java.util.Map;

/**
 * Groups a validated {@link LlmCompiledPlan} into ordered execution levels, where a task's level is
 * one more than the deepest level of its declared dependencies. Tasks sharing a level have no edges
 * between them and run concurrently.
 *
 * <p>This is the "execute the LLM's DAG as-is" core: parallelism is derived from the <em>explicit
 * edges the LLM authored</em>, not re-derived from declared input/output contracts. Contrast with
 * {@code com.google.adk.planner.goap.DependencyGraphSearch#assignParallelLevels}, which levels a
 * search result by its input/output data — same shape of output ({@code
 * ImmutableList<ImmutableList<String>>}), different source of truth.
 */
public final class DagLeveler {

  private DagLeveler() {}

  /**
   * Converts the LLM's task DAG into parallel execution groups of agent names, preserving the
   * authored task order within each level.
   *
   * @param plan a plan that has already passed {@link PlanValidator#validate}
   * @return ordered groups; agents within a group are independent and can run in parallel
   */
  public static ImmutableList<ImmutableList<String>> toGroups(LlmCompiledPlan plan) {
    if (plan == null || plan.tasks() == null || plan.tasks().isEmpty()) {
      return ImmutableList.of();
    }

    Map<Integer, LlmCompiledPlan.PlanTask> byId = new HashMap<>();
    for (LlmCompiledPlan.PlanTask task : plan.tasks()) {
      byId.put(task.id(), task);
    }

    // Memoized level computation over the explicit edges.
    Map<Integer, Integer> levelOf = new HashMap<>();
    for (LlmCompiledPlan.PlanTask task : plan.tasks()) {
      computeLevel(task.id(), byId, levelOf);
    }

    int maxLevel = levelOf.values().stream().mapToInt(Integer::intValue).max().orElse(0);
    ImmutableList.Builder<ImmutableList<String>> groups = ImmutableList.builder();
    for (int level = 0; level <= maxLevel; level++) {
      final int current = level;
      ImmutableList<String> agentsAtLevel =
          plan.tasks().stream()
              .filter(t -> levelOf.getOrDefault(t.id(), -1) == current)
              .map(LlmCompiledPlan.PlanTask::agent)
              .collect(ImmutableList.toImmutableList());
      if (!agentsAtLevel.isEmpty()) {
        groups.add(agentsAtLevel);
      }
    }
    return groups.build();
  }

  private static int computeLevel(
      int id, Map<Integer, LlmCompiledPlan.PlanTask> byId, Map<Integer, Integer> levelOf) {

    Integer cached = levelOf.get(id);
    if (cached != null) {
      return cached;
    }
    LlmCompiledPlan.PlanTask task = byId.get(id);
    int level = 0;
    if (task != null) {
      for (Integer dep : task.dependsOn()) {
        if (byId.containsKey(dep)) {
          level = Math.max(level, computeLevel(dep, byId, levelOf) + 1);
        }
      }
    }
    levelOf.put(id, level);
    return level;
  }
}
