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

import static com.google.common.truth.Truth.assertThat;

import com.google.adk.planner.goap.AgentMetadata;
import com.google.adk.planner.llmcompiler.LlmCompiledPlan.PlanTask;
import com.google.common.collect.ImmutableList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link PlanValidator}. */
class PlanValidatorTest {

  private static Map<String, AgentMetadata> catalog(AgentMetadata... metas) {
    Map<String, AgentMetadata> byName = new LinkedHashMap<>();
    for (AgentMetadata meta : metas) {
      byName.put(meta.agentName(), meta);
    }
    return byName;
  }

  @Test
  void validPlan_passes() {
    Map<String, AgentMetadata> catalog =
        catalog(
            new AgentMetadata("A", ImmutableList.of(), "a"),
            new AgentMetadata("B", ImmutableList.of("a"), "b"));
    LlmCompiledPlan plan =
        new LlmCompiledPlan(
            "ok", List.of(new PlanTask(1, "A", List.of()), new PlanTask(2, "B", List.of(1))));

    PlanValidator.ValidationResult result = PlanValidator.validate(plan, catalog, Set.of());

    assertThat(result.ok()).isTrue();
    assertThat(result.errors()).isEmpty();
  }

  @Test
  void emptyPlan_fails() {
    PlanValidator.ValidationResult result =
        PlanValidator.validate(new LlmCompiledPlan("empty", List.of()), catalog(), Set.of());

    assertThat(result.ok()).isFalse();
    assertThat(result.errors()).contains("Plan is empty");
  }

  @Test
  void nullTasks_fails() {
    PlanValidator.ValidationResult result =
        PlanValidator.validate(new LlmCompiledPlan("null", null), catalog(), Set.of());

    assertThat(result.ok()).isFalse();
    assertThat(result.errors()).contains("Plan is empty");
  }

  @Test
  void duplicateId_fails() {
    Map<String, AgentMetadata> catalog =
        catalog(
            new AgentMetadata("A", ImmutableList.of(), "a"),
            new AgentMetadata("B", ImmutableList.of(), "b"));
    LlmCompiledPlan plan =
        new LlmCompiledPlan(
            "dup", List.of(new PlanTask(1, "A", List.of()), new PlanTask(1, "B", List.of())));

    PlanValidator.ValidationResult result = PlanValidator.validate(plan, catalog, Set.of());

    assertThat(result.ok()).isFalse();
    assertThat(result.errors()).contains("Duplicate task id: 1");
  }

  @Test
  void unknownAgent_fails() {
    Map<String, AgentMetadata> catalog = catalog(new AgentMetadata("A", ImmutableList.of(), "a"));
    LlmCompiledPlan plan =
        new LlmCompiledPlan("ghost", List.of(new PlanTask(1, "ghost", List.of())));

    PlanValidator.ValidationResult result = PlanValidator.validate(plan, catalog, Set.of());

    assertThat(result.ok()).isFalse();
    assertThat(result.errors().stream().anyMatch(e -> e.contains("Unknown agent 'ghost'")))
        .isTrue();
  }

  @Test
  void danglingEdge_fails() {
    Map<String, AgentMetadata> catalog =
        catalog(
            new AgentMetadata("A", ImmutableList.of(), "a"),
            new AgentMetadata("B", ImmutableList.of(), "b"));
    LlmCompiledPlan plan =
        new LlmCompiledPlan(
            "dangling",
            List.of(new PlanTask(1, "A", List.of()), new PlanTask(2, "B", List.of(99))));

    PlanValidator.ValidationResult result = PlanValidator.validate(plan, catalog, Set.of());

    assertThat(result.ok()).isFalse();
    assertThat(result.errors().stream().anyMatch(e -> e.contains("depends on unknown task 99")))
        .isTrue();
  }

  @Test
  void cycle_fails() {
    Map<String, AgentMetadata> catalog =
        catalog(
            new AgentMetadata("A", ImmutableList.of(), "a"),
            new AgentMetadata("B", ImmutableList.of(), "b"));
    LlmCompiledPlan plan =
        new LlmCompiledPlan(
            "cycle", List.of(new PlanTask(1, "A", List.of(2)), new PlanTask(2, "B", List.of(1))));

    PlanValidator.ValidationResult result = PlanValidator.validate(plan, catalog, Set.of());

    assertThat(result.ok()).isFalse();
    assertThat(result.errors()).contains("Plan contains a dependency cycle");
  }

  @Test
  void missingPrecondition_fails() {
    // B requires "x", but its only dependency (A) produces "a".
    Map<String, AgentMetadata> catalog =
        catalog(
            new AgentMetadata("A", ImmutableList.of(), "a"),
            new AgentMetadata("B", ImmutableList.of("x"), "b"));
    LlmCompiledPlan plan =
        new LlmCompiledPlan(
            "missing", List.of(new PlanTask(1, "A", List.of()), new PlanTask(2, "B", List.of(1))));

    PlanValidator.ValidationResult result = PlanValidator.validate(plan, catalog, Set.of());

    assertThat(result.ok()).isFalse();
    assertThat(result.errors().stream().anyMatch(e -> e.contains("missing required input 'x'")))
        .isTrue();
  }

  @Test
  void preconditionSatisfiedByInitialState_passes() {
    // B requires "x" with no producing dependency, but "x" is already in the initial state.
    Map<String, AgentMetadata> catalog =
        catalog(new AgentMetadata("B", ImmutableList.of("x"), "b"));
    LlmCompiledPlan plan = new LlmCompiledPlan("seeded", List.of(new PlanTask(1, "B", List.of())));

    PlanValidator.ValidationResult result = PlanValidator.validate(plan, catalog, Set.of("x"));

    assertThat(result.ok()).isTrue();
  }

  @Test
  void preconditionSatisfiedTransitively_passes() {
    // C requires both "a" and "b"; deps reach A (→a) and B (→b) transitively.
    Map<String, AgentMetadata> catalog =
        catalog(
            new AgentMetadata("A", ImmutableList.of(), "a"),
            new AgentMetadata("B", ImmutableList.of("a"), "b"),
            new AgentMetadata("C", ImmutableList.of("a", "b"), "c"));
    LlmCompiledPlan plan =
        new LlmCompiledPlan(
            "chain",
            List.of(
                new PlanTask(1, "A", List.of()),
                new PlanTask(2, "B", List.of(1)),
                new PlanTask(3, "C", List.of(1, 2))));

    PlanValidator.ValidationResult result = PlanValidator.validate(plan, catalog, Set.of());

    assertThat(result.ok()).isTrue();
  }
}
