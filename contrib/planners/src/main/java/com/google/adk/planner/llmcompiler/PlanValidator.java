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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Validates an LLM-authored {@link LlmCompiledPlan} against the agent catalog ({@link
 * AgentMetadata} declarations). This is the safety net that keeps a hallucinated or malformed plan
 * from ever reaching execution.
 *
 * <p>The plan is executed <em>as the LLM authored it</em> (its explicit edges drive parallelism);
 * the catalog is used only to <em>verify</em> that:
 *
 * <ol>
 *   <li>every referenced agent exists,
 *   <li>task ids are unique and {@code dependsOn} edges point at real tasks,
 *   <li>the graph is acyclic, and
 *   <li>each task's declared dependencies actually <em>supply</em> the input keys that agent
 *       requires (precondition closure), given the initial session state.
 * </ol>
 *
 * <p>A failing result tells {@link LlmPlanCompiler} to self-repair (one LLM retry with these
 * errors) or, failing that, fall back to a deterministic GOAP plan — so a run never hard-fails on a
 * bad plan.
 */
public final class PlanValidator {

  private PlanValidator() {}

  /**
   * @param ok true if the plan is safe to execute as-is
   * @param errors human-readable reasons the plan was rejected (empty when {@code ok})
   */
  public record ValidationResult(boolean ok, List<String> errors) {
    public static ValidationResult valid() {
      return new ValidationResult(true, List.of());
    }
  }

  /**
   * Validates {@code plan} against the {@code catalogByName} catalog, starting from the world state
   * keys already present in {@code initialState}.
   *
   * @param plan the LLM-authored plan
   * @param catalogByName agent name → its input/output contract
   * @param initialState state keys already available before any task runs (treated as satisfied
   *     preconditions)
   */
  public static ValidationResult validate(
      LlmCompiledPlan plan, Map<String, AgentMetadata> catalogByName, Set<String> initialState) {
    List<String> errors = new ArrayList<>();

    if (plan == null || plan.tasks() == null || plan.tasks().isEmpty()) {
      return new ValidationResult(false, List.of("Plan is empty"));
    }

    // 1. Unique ids + known agents; build id → task index.
    Map<Integer, LlmCompiledPlan.PlanTask> byId = new LinkedHashMap<>();
    for (LlmCompiledPlan.PlanTask task : plan.tasks()) {
      if (byId.containsKey(task.id())) {
        errors.add("Duplicate task id: " + task.id());
      }
      byId.put(task.id(), task);
      if (!catalogByName.containsKey(task.agent())) {
        errors.add("Unknown agent '" + task.agent() + "' in task " + task.id());
      }
    }

    // 2. Edges point at real tasks.
    for (LlmCompiledPlan.PlanTask task : plan.tasks()) {
      for (Integer dep : task.dependsOn()) {
        if (!byId.containsKey(dep)) {
          errors.add("Task " + task.id() + " depends on unknown task " + dep);
        }
      }
    }

    // Stop early if structural errors exist — closure/cycle checks assume a sane graph.
    if (!errors.isEmpty()) {
      return new ValidationResult(false, errors);
    }

    // 3. Acyclicity (DFS with a recursion stack).
    if (hasCycle(byId)) {
      errors.add("Plan contains a dependency cycle");
      return new ValidationResult(false, errors);
    }

    // 4. Precondition closure: each task's declared dependencies must transitively produce every
    //    input key the agent requires (plus whatever is already in the initial state).
    for (LlmCompiledPlan.PlanTask task : plan.tasks()) {
      AgentMetadata meta = catalogByName.get(task.agent());
      Set<String> available = availableBefore(task, byId, catalogByName, initialState);
      for (String inputKey : meta.inputKeys()) {
        if (!available.contains(inputKey)) {
          errors.add(
              "Task "
                  + task.id()
                  + " ("
                  + task.agent()
                  + ") is missing required input '"
                  + inputKey
                  + "' — no declared dependency produces it");
        }
      }
    }

    return errors.isEmpty() ? ValidationResult.valid() : new ValidationResult(false, errors);
  }

  /**
   * State keys guaranteed to be present immediately before {@code task} runs: the initial state
   * plus the output keys of every transitive dependency.
   */
  private static Set<String> availableBefore(
      LlmCompiledPlan.PlanTask task,
      Map<Integer, LlmCompiledPlan.PlanTask> byId,
      Map<String, AgentMetadata> catalogByName,
      Set<String> initialState) {

    Set<String> available = new HashSet<>(initialState);
    Set<Integer> visited = new HashSet<>();
    for (Integer dep : task.dependsOn()) {
      collectEffects(dep, byId, catalogByName, visited, available);
    }
    return available;
  }

  private static void collectEffects(
      Integer id,
      Map<Integer, LlmCompiledPlan.PlanTask> byId,
      Map<String, AgentMetadata> catalogByName,
      Set<Integer> visited,
      Set<String> into) {

    if (!visited.add(id)) {
      return;
    }
    LlmCompiledPlan.PlanTask task = byId.get(id);
    if (task == null) {
      return;
    }
    AgentMetadata meta = catalogByName.get(task.agent());
    if (meta != null && meta.outputKey() != null) {
      into.add(meta.outputKey());
    }
    for (Integer dep : task.dependsOn()) {
      collectEffects(dep, byId, catalogByName, visited, into);
    }
  }

  private static boolean hasCycle(Map<Integer, LlmCompiledPlan.PlanTask> byId) {
    Set<Integer> visited = new HashSet<>();
    Set<Integer> onStack = new HashSet<>();
    for (Integer id : byId.keySet()) {
      if (dfsCycle(id, byId, visited, onStack)) {
        return true;
      }
    }
    return false;
  }

  private static boolean dfsCycle(
      Integer id,
      Map<Integer, LlmCompiledPlan.PlanTask> byId,
      Set<Integer> visited,
      Set<Integer> onStack) {

    if (onStack.contains(id)) {
      return true;
    }
    if (!visited.add(id)) {
      return false;
    }
    onStack.add(id);
    LlmCompiledPlan.PlanTask task = byId.get(id);
    if (task != null) {
      for (Integer dep : task.dependsOn()) {
        if (dfsCycle(dep, byId, visited, onStack)) {
          return true;
        }
      }
    }
    onStack.remove(id);
    return false;
  }
}
