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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.planner.goap.AgentMetadata;
import com.google.adk.planner.goap.DfsSearchStrategy;
import com.google.adk.planner.goap.GoalOrientedSearchGraph;
import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

/** Unit tests for {@link LlmPlanCompiler} — happy path, self-repair, and GOAP fallback. */
class LlmPlanCompilerTest {

  // A:[]->a, B:[a]->goal. GOAP fallback for goal "goal" yields [[A],[B]].
  private static final List<AgentMetadata> CATALOG =
      List.of(
          new AgentMetadata("A", ImmutableList.of(), "a"),
          new AgentMetadata("B", ImmutableList.of("a"), "goal"));

  private static final String VALID_PLAN =
      "{\"label\":\"p\",\"tasks\":["
          + "{\"id\":1,\"agent\":\"A\",\"dependsOn\":[]},"
          + "{\"id\":2,\"agent\":\"B\",\"dependsOn\":[1]}]}";

  private static final String INVALID_PLAN =
      "{\"label\":\"bad\",\"tasks\":[{\"id\":1,\"agent\":\"ghost\",\"dependsOn\":[]}]}";

  @Test
  void compile_validPlan_usesLlmGroups() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(VALID_PLAN)));

    LlmPlanCompiler compiler = new LlmPlanCompiler(mockLlm, CATALOG, "goal");
    LlmPlanCompiler.CompiledPlan plan = compiler.compile("do it", null, Set.of()).blockingGet();

    assertThat(plan.fromLlm()).isTrue();
    assertThat(plan.groups()).hasSize(2);
    assertThat(plan.groups().get(0)).containsExactly("A");
    assertThat(plan.groups().get(1)).containsExactly("B");
    verify(mockLlm, times(1)).generateContent(any(), eq(false));
  }

  @Test
  void compile_invalidThenValid_selfRepairs() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(INVALID_PLAN)))
        .thenReturn(Flowable.just(textResponse(VALID_PLAN)));

    LlmPlanCompiler compiler = new LlmPlanCompiler(mockLlm, CATALOG, "goal");
    LlmPlanCompiler.CompiledPlan plan = compiler.compile("do it", null, Set.of()).blockingGet();

    assertThat(plan.fromLlm()).isTrue();
    assertThat(plan.groups().get(0)).containsExactly("A");

    // Two calls: the rejected plan and the repair re-prompt carrying its validation errors.
    ArgumentCaptor<LlmRequest> captor = ArgumentCaptor.forClass(LlmRequest.class);
    verify(mockLlm, times(2)).generateContent(captor.capture(), eq(false));
    String repairPrompt = promptOf(captor.getAllValues().get(1));
    assertThat(repairPrompt).contains("REJECTED");
    assertThat(repairPrompt).contains("Unknown agent 'ghost'");
  }

  @Test
  void compile_invalidBothTries_fallsBackToGoap() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(INVALID_PLAN)));

    LlmPlanCompiler compiler = new LlmPlanCompiler(mockLlm, CATALOG, "goal");
    LlmPlanCompiler.CompiledPlan plan = compiler.compile("do it", null, Set.of()).blockingGet();

    assertThat(plan.fromLlm()).isFalse();
    ImmutableList<ImmutableList<String>> expected =
        new DfsSearchStrategy()
            .searchGrouped(new GoalOrientedSearchGraph(CATALOG), CATALOG, Set.of(), "goal");
    assertThat(plan.groups()).isEqualTo(expected);
    // Self-repair still tried the LLM twice before giving up.
    verify(mockLlm, times(2)).generateContent(any(), eq(false));
  }

  @Test
  void compile_malformedJson_fallsBackToGoap() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse("not json at all")));

    LlmPlanCompiler compiler = new LlmPlanCompiler(mockLlm, CATALOG, "goal");
    LlmPlanCompiler.CompiledPlan plan = compiler.compile("do it", null, Set.of()).blockingGet();

    assertThat(plan.fromLlm()).isFalse();
    assertThat(plan.groups()).hasSize(2);
  }

  @Test
  void compile_llmError_fallsBackToGoap() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.error(new RuntimeException("LLM down")));

    LlmPlanCompiler compiler = new LlmPlanCompiler(mockLlm, CATALOG, "goal");
    LlmPlanCompiler.CompiledPlan plan = compiler.compile("do it", null, Set.of()).blockingGet();

    assertThat(plan.fromLlm()).isFalse();
    assertThat(plan.groups()).hasSize(2);
  }

  @Test
  void compile_unresolvableFallbackGoal_yieldsEmptyPlan() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.error(new RuntimeException("LLM down")));

    LlmPlanCompiler compiler = new LlmPlanCompiler(mockLlm, CATALOG, "nonexistent_goal");
    LlmPlanCompiler.CompiledPlan plan = compiler.compile("do it", null, Set.of()).blockingGet();

    assertThat(plan.fromLlm()).isFalse();
    assertThat(plan.groups()).isEmpty();
  }

  @Test
  void extractJson_stripsMarkdownFences() {
    String fenced = "```json\n{\"label\":\"x\",\"tasks\":[]}\n```";
    assertThat(LlmPlanCompiler.extractJson(fenced)).isEqualTo("{\"label\":\"x\",\"tasks\":[]}");
  }

  @Test
  void extractJson_stripsSurroundingProse() {
    String prose = "Here is the plan: {\"label\":\"x\"} hope it helps!";
    assertThat(LlmPlanCompiler.extractJson(prose)).isEqualTo("{\"label\":\"x\"}");
  }

  @Test
  void extractJson_nullYieldsEmptyObject() {
    assertThat(LlmPlanCompiler.extractJson(null)).isEqualTo("{}");
  }

  @Test
  void extractJson_noBracesReturnsStripped() {
    assertThat(LlmPlanCompiler.extractJson("  no braces here  ")).isEqualTo("no braces here");
  }

  private static LlmResponse textResponse(String text) {
    return LlmResponse.builder()
        .content(Content.builder().role("model").parts(Part.fromText(text)).build())
        .build();
  }

  private static String promptOf(LlmRequest request) {
    return request.contents().get(0).parts().get().get(0).text().get();
  }
}
