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
import static org.mockito.Mockito.when;

import com.google.adk.agents.BaseAgent;
import com.google.adk.agents.InvocationContext;
import com.google.adk.agents.PlannerAction;
import com.google.adk.agents.PlanningContext;
import com.google.adk.events.Event;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmResponse;
import com.google.adk.planner.goap.AgentMetadata;
import com.google.adk.sessions.InMemorySessionService;
import com.google.adk.sessions.Session;
import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link LlmCompilerPlanner} loop mechanics (group walking, parallelism, skips). */
class LlmCompilerPlannerTest {

  private static final class SimpleTestAgent extends BaseAgent {
    SimpleTestAgent(String name) {
      super(name, "test agent " + name, ImmutableList.of(), null, null);
    }

    @Override
    protected Flowable<Event> runAsyncImpl(InvocationContext ctx) {
      return Flowable.empty();
    }

    @Override
    protected Flowable<Event> runLiveImpl(InvocationContext ctx) {
      return Flowable.empty();
    }
  }

  // A:[]->a, B:[a]->goal.
  private static final List<AgentMetadata> LINEAR_CATALOG =
      List.of(
          new AgentMetadata("A", ImmutableList.of(), "a"),
          new AgentMetadata("B", ImmutableList.of("a"), "goal"));

  private static final String LINEAR_PLAN =
      "{\"label\":\"p\",\"tasks\":["
          + "{\"id\":1,\"agent\":\"A\",\"dependsOn\":[]},"
          + "{\"id\":2,\"agent\":\"B\",\"dependsOn\":[1]}]}";

  @Test
  void firstAction_compilesAndReturnsFirstGroup() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(LINEAR_PLAN)));

    LlmCompilerPlanner planner =
        LlmCompilerPlanner.builder(mockLlm, LINEAR_CATALOG).fallbackGoal("goal").build();
    PlanningContext context =
        context(ImmutableList.of(new SimpleTestAgent("A"), new SimpleTestAgent("B")));
    planner.init(context);

    PlannerAction action = planner.firstAction(context).blockingGet();
    assertThat(action).isInstanceOf(PlannerAction.RunAgents.class);
    assertThat(names((PlannerAction.RunAgents) action)).containsExactly("A");
  }

  @Test
  void fullWalk_finishesWithDefaultJoiner() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(LINEAR_PLAN)));

    LlmCompilerPlanner planner =
        LlmCompilerPlanner.builder(mockLlm, LINEAR_CATALOG).fallbackGoal("goal").build();
    PlanningContext context =
        context(ImmutableList.of(new SimpleTestAgent("A"), new SimpleTestAgent("B")));
    planner.init(context);

    PlannerAction g1 = planner.firstAction(context).blockingGet();
    assertThat(names((PlannerAction.RunAgents) g1)).containsExactly("A");

    PlannerAction g2 = planner.nextAction(context).blockingGet();
    assertThat(names((PlannerAction.RunAgents) g2)).containsExactly("B");

    PlannerAction done = planner.nextAction(context).blockingGet();
    assertThat(done).isInstanceOf(PlannerAction.DoneWithResult.class);
    assertThat(((PlannerAction.DoneWithResult) done).result()).isEqualTo("Plan complete");
  }

  @Test
  void parallelLevel_surfacesAsSingleRunAgents() {
    List<AgentMetadata> catalog =
        List.of(
            new AgentMetadata("gather", ImmutableList.of(), "raw"),
            new AgentMetadata("summarizeA", ImmutableList.of("raw"), "sumA"),
            new AgentMetadata("summarizeB", ImmutableList.of("raw"), "sumB"),
            new AgentMetadata("synthesize", ImmutableList.of("sumA", "sumB"), "goal"));
    String plan =
        "{\"label\":\"fanout\",\"tasks\":["
            + "{\"id\":1,\"agent\":\"gather\",\"dependsOn\":[]},"
            + "{\"id\":2,\"agent\":\"summarizeA\",\"dependsOn\":[1]},"
            + "{\"id\":3,\"agent\":\"summarizeB\",\"dependsOn\":[1]},"
            + "{\"id\":4,\"agent\":\"synthesize\",\"dependsOn\":[2,3]}]}";

    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false))).thenReturn(Flowable.just(textResponse(plan)));

    LlmCompilerPlanner planner =
        LlmCompilerPlanner.builder(mockLlm, catalog).fallbackGoal("goal").build();
    PlanningContext context =
        context(
            ImmutableList.of(
                new SimpleTestAgent("gather"),
                new SimpleTestAgent("summarizeA"),
                new SimpleTestAgent("summarizeB"),
                new SimpleTestAgent("synthesize")));
    planner.init(context);

    assertThat(names((PlannerAction.RunAgents) planner.firstAction(context).blockingGet()))
        .containsExactly("gather");

    PlannerAction parallel = planner.nextAction(context).blockingGet();
    assertThat(parallel).isInstanceOf(PlannerAction.RunAgents.class);
    assertThat(names((PlannerAction.RunAgents) parallel))
        .containsExactly("summarizeA", "summarizeB");

    assertThat(names((PlannerAction.RunAgents) planner.nextAction(context).blockingGet()))
        .containsExactly("synthesize");
  }

  @Test
  void unknownAgentInGroup_isSkipped() {
    // The plan references B, but B is absent from the available agents — its group is skipped.
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(LINEAR_PLAN)));

    LlmCompilerPlanner planner =
        LlmCompilerPlanner.builder(mockLlm, LINEAR_CATALOG).fallbackGoal("goal").build();
    // Only A is available.
    PlanningContext context = context(ImmutableList.of(new SimpleTestAgent("A")));
    planner.init(context);

    assertThat(names((PlannerAction.RunAgents) planner.firstAction(context).blockingGet()))
        .containsExactly("A");

    // Group [B] resolves to nothing → skipped → plan exhausted → default joiner finishes.
    PlannerAction done = planner.nextAction(context).blockingGet();
    assertThat(done).isInstanceOf(PlannerAction.DoneWithResult.class);
  }

  private static List<String> names(PlannerAction.RunAgents action) {
    return action.agents().stream().map(BaseAgent::name).toList();
  }

  private static LlmResponse textResponse(String text) {
    return LlmResponse.builder()
        .content(Content.builder().role("model").parts(Part.fromText(text)).build())
        .build();
  }

  private static PlanningContext context(ImmutableList<BaseAgent> agents) {
    InMemorySessionService sessionService = new InMemorySessionService();
    Session session = sessionService.createSession("test-app", "test-user").blockingGet();
    InvocationContext invocationContext =
        InvocationContext.builder()
            .sessionService(sessionService)
            .invocationId("test-invocation")
            .agent(agents.get(0))
            .session(session)
            .build();
    return new PlanningContext(invocationContext, agents);
  }
}
