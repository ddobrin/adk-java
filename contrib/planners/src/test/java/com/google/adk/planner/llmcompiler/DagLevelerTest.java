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

import com.google.adk.planner.llmcompiler.LlmCompiledPlan.PlanTask;
import com.google.common.collect.ImmutableList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link DagLeveler}. */
class DagLevelerTest {

  @Test
  void emptyPlan_producesNoGroups() {
    assertThat(DagLeveler.toGroups(new LlmCompiledPlan("empty", List.of()))).isEmpty();
  }

  @Test
  void nullTasks_producesNoGroups() {
    assertThat(DagLeveler.toGroups(new LlmCompiledPlan("null", null))).isEmpty();
  }

  @Test
  void linearChain_producesSingletonGroupPerTask() {
    LlmCompiledPlan plan =
        new LlmCompiledPlan(
            "chain",
            List.of(
                new PlanTask(1, "A", List.of()),
                new PlanTask(2, "B", List.of(1)),
                new PlanTask(3, "C", List.of(2)),
                new PlanTask(4, "D", List.of(3))));

    ImmutableList<ImmutableList<String>> groups = DagLeveler.toGroups(plan);

    assertThat(groups).hasSize(4);
    assertThat(groups.get(0)).containsExactly("A");
    assertThat(groups.get(1)).containsExactly("B");
    assertThat(groups.get(2)).containsExactly("C");
    assertThat(groups.get(3)).containsExactly("D");
  }

  @Test
  void diamond_groupsIndependentTasksTogether() {
    // A -> {B, C} -> D
    LlmCompiledPlan plan =
        new LlmCompiledPlan(
            "diamond",
            List.of(
                new PlanTask(1, "A", List.of()),
                new PlanTask(2, "B", List.of(1)),
                new PlanTask(3, "C", List.of(1)),
                new PlanTask(4, "D", List.of(2, 3))));

    ImmutableList<ImmutableList<String>> groups = DagLeveler.toGroups(plan);

    assertThat(groups).hasSize(3);
    assertThat(groups.get(0)).containsExactly("A");
    assertThat(groups.get(1)).containsExactly("B", "C").inOrder();
    assertThat(groups.get(2)).containsExactly("D");
  }

  @Test
  void councilDag_levelsIntoThreeGroups() {
    // The 8-task council DAG from the LlmCompiledPlan javadoc.
    LlmCompiledPlan plan =
        new LlmCompiledPlan(
            "council",
            List.of(
                new PlanTask(1, "initial_response", List.of()),
                new PlanTask(2, "peer_ranking", List.of(1)),
                new PlanTask(3, "agreement_analysis", List.of(1)),
                new PlanTask(4, "disagreement_analysis", List.of(1)),
                new PlanTask(5, "final_synthesis", List.of(1, 2)),
                new PlanTask(6, "aggregate_rankings", List.of(2)),
                new PlanTask(7, "aggregate_agreements", List.of(3)),
                new PlanTask(8, "aggregate_disagreements", List.of(4))));

    ImmutableList<ImmutableList<String>> groups = DagLeveler.toGroups(plan);

    assertThat(groups).hasSize(3);
    assertThat(groups.get(0)).containsExactly("initial_response");
    assertThat(groups.get(1))
        .containsExactly("peer_ranking", "agreement_analysis", "disagreement_analysis")
        .inOrder();
    assertThat(groups.get(2))
        .containsExactly(
            "final_synthesis",
            "aggregate_rankings",
            "aggregate_agreements",
            "aggregate_disagreements")
        .inOrder();
  }
}
