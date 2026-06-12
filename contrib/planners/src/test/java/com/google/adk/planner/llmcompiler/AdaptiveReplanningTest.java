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

import com.google.adk.agents.BaseAgent;
import com.google.adk.agents.InvocationContext;
import com.google.adk.agents.PlannerAction;
import com.google.adk.agents.PlanningContext;
import com.google.adk.events.Event;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.planner.goap.AgentMetadata;
import com.google.adk.sessions.InMemorySessionService;
import com.google.adk.sessions.Session;
import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

/**
 * The centerpiece adaptive-replanning suite for {@link LlmCompilerPlanner}.
 *
 * <p>Verifies that the {@link Joiner} drives finish-vs-replan, that {@code replanCount} is threaded
 * into the joiner on every join, that each replan recompiles (carrying the joiner's feedback into
 * the next planning prompt), and that replanning is bounded by {@code maxReplans}.
 */
class AdaptiveReplanningTest {

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

  /** A {@link Joiner} that replays a fixed script and records every replanCount it is given. */
  private static final class ScriptedJoiner implements Joiner {
    private final List<JoinDecision> script;
    private final List<Integer> seenReplanCounts = new ArrayList<>();
    private int index = 0;

    ScriptedJoiner(JoinDecision... decisions) {
      this.script = List.of(decisions);
    }

    @Override
    public Single<JoinDecision> decide(PlanningContext context, int replanCount) {
      seenReplanCounts.add(replanCount);
      JoinDecision decision = script.get(Math.min(index, script.size() - 1));
      index++;
      return Single.just(decision);
    }
  }

  private static final List<AgentMetadata> CATALOG =
      List.of(new AgentMetadata("worker", ImmutableList.of(), "out"));

  private static final String WORKER_PLAN =
      "{\"label\":\"p\",\"tasks\":[{\"id\":1,\"agent\":\"worker\",\"dependsOn\":[]}]}";

  @Test
  void finishImmediately_runsOnePlanNoRecompile() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(WORKER_PLAN)));

    ScriptedJoiner joiner = new ScriptedJoiner(new JoinDecision.Finish("all done"));
    LlmCompilerPlanner planner = planner(mockLlm, joiner, 2);
    PlanningContext context = context();
    planner.init(context);

    PlannerAction g1 = planner.firstAction(context).blockingGet();
    assertThat(names((PlannerAction.RunAgents) g1)).containsExactly("worker");

    PlannerAction done = planner.nextAction(context).blockingGet();
    assertThat(done).isInstanceOf(PlannerAction.DoneWithResult.class);
    assertThat(((PlannerAction.DoneWithResult) done).result()).isEqualTo("all done");

    // Only the firstAction compile happened — no recompile.
    verify(mockLlm, times(1)).generateContent(any(), eq(false));
    assertThat(joiner.seenReplanCounts).containsExactly(0);
  }

  @Test
  void replanOnceThenFinish_recompilesWithFeedback() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(WORKER_PLAN)));

    ScriptedJoiner joiner =
        new ScriptedJoiner(
            new JoinDecision.Replan("dig deeper on sources"),
            new JoinDecision.Finish("now complete"));
    LlmCompilerPlanner planner = planner(mockLlm, joiner, 2);
    PlanningContext context = context();
    planner.init(context);

    planner.firstAction(context).blockingGet(); // group 1 (compile #1)
    PlannerAction afterReplan =
        planner.nextAction(context).blockingGet(); // join(0) -> replan -> recompile
    assertThat(afterReplan).isInstanceOf(PlannerAction.RunAgents.class);
    assertThat(names((PlannerAction.RunAgents) afterReplan)).containsExactly("worker");

    PlannerAction done = planner.nextAction(context).blockingGet(); // join(1) -> finish
    assertThat(done).isInstanceOf(PlannerAction.DoneWithResult.class);
    assertThat(((PlannerAction.DoneWithResult) done).result()).isEqualTo("now complete");

    // replanCount threaded 0 then 1.
    assertThat(joiner.seenReplanCounts).containsExactly(0, 1).inOrder();

    // Two compiles: the initial plan and one recompile carrying the joiner's feedback.
    ArgumentCaptor<LlmRequest> captor = ArgumentCaptor.forClass(LlmRequest.class);
    verify(mockLlm, times(2)).generateContent(captor.capture(), eq(false));
    String recompilePrompt = promptOf(captor.getAllValues().get(1));
    assertThat(recompilePrompt).contains("REPLAN FEEDBACK");
    assertThat(recompilePrompt).contains("dig deeper on sources");
  }

  @Test
  void replanAlways_boundedByMaxReplans() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(WORKER_PLAN)));

    ScriptedJoiner joiner = new ScriptedJoiner(new JoinDecision.Replan("keep going"));
    LlmCompilerPlanner planner = planner(mockLlm, joiner, 2);
    PlanningContext context = context();
    planner.init(context);

    planner.firstAction(context).blockingGet(); // group 1 (compile #1)
    planner.nextAction(context).blockingGet(); // join(0) -> replan -> recompile #2
    planner.nextAction(context).blockingGet(); // join(1) -> replan -> recompile #3
    PlannerAction exhausted = planner.nextAction(context).blockingGet(); // join(2) -> bounded

    assertThat(exhausted).isInstanceOf(PlannerAction.DoneWithResult.class);
    assertThat(((PlannerAction.DoneWithResult) exhausted).result()).contains("max replan attempts");
    assertThat(((PlannerAction.DoneWithResult) exhausted).result()).contains("exhausted");

    // replanCount threaded 0, 1, 2 — the joiner is consulted on each exhaustion.
    assertThat(joiner.seenReplanCounts).containsExactly(0, 1, 2).inOrder();
    // 1 initial compile + 2 recompiles (the 3rd replan is refused before compiling).
    verify(mockLlm, times(3)).generateContent(any(), eq(false));
  }

  private static LlmCompilerPlanner planner(BaseLlm llm, Joiner joiner, int maxReplans) {
    return LlmCompilerPlanner.builder(llm, CATALOG)
        .fallbackGoal("out")
        .joiner(joiner)
        .maxReplans(maxReplans)
        .defaultInstruction("do the work")
        .build();
  }

  private static List<String> names(PlannerAction.RunAgents action) {
    return action.agents().stream().map(BaseAgent::name).toList();
  }

  private static LlmResponse textResponse(String text) {
    return LlmResponse.builder()
        .content(Content.builder().role("model").parts(Part.fromText(text)).build())
        .build();
  }

  private static String promptOf(LlmRequest request) {
    return request.contents().get(0).parts().get().get(0).text().get();
  }

  private static PlanningContext context() {
    InMemorySessionService sessionService = new InMemorySessionService();
    Session session = sessionService.createSession("test-app", "test-user").blockingGet();
    ImmutableList<BaseAgent> agents = ImmutableList.of(new SimpleTestAgent("worker"));
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
