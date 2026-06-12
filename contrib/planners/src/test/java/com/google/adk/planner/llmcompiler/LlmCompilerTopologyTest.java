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
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * End-to-end topology test: the planning LLM emits a council-style 8-task DAG, and {@link
 * LlmCompilerPlanner} walks it as three leveled execution groups in dependency order, with parallel
 * levels surfacing as multi-agent {@link PlannerAction.RunAgents}. Mirrors the GOAP {@code
 * GoapLlmCouncilTopologyTest}, except the graph here is authored by the LLM rather than derived
 * from input/output contracts.
 */
class LlmCompilerTopologyTest {

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

  private static final List<AgentMetadata> COUNCIL_METADATA =
      List.of(
          new AgentMetadata("initial_response", ImmutableList.of(), "individual_responses"),
          new AgentMetadata(
              "peer_ranking", ImmutableList.of("individual_responses"), "peer_rankings"),
          new AgentMetadata(
              "agreement_analysis", ImmutableList.of("individual_responses"), "agreement_analyses"),
          new AgentMetadata(
              "disagreement_analysis",
              ImmutableList.of("individual_responses"),
              "disagreement_analyses"),
          new AgentMetadata(
              "final_synthesis",
              ImmutableList.of("individual_responses", "peer_rankings"),
              "final_synthesis"),
          new AgentMetadata(
              "aggregate_rankings", ImmutableList.of("peer_rankings"), "aggregate_rankings"),
          new AgentMetadata(
              "aggregate_agreements",
              ImmutableList.of("agreement_analyses"),
              "aggregate_agreements"),
          new AgentMetadata(
              "aggregate_disagreements",
              ImmutableList.of("disagreement_analyses"),
              "aggregate_disagreements"));

  private static final ImmutableList<String> ALL_AGENTS =
      ImmutableList.of(
          "initial_response",
          "peer_ranking",
          "agreement_analysis",
          "disagreement_analysis",
          "final_synthesis",
          "aggregate_rankings",
          "aggregate_agreements",
          "aggregate_disagreements");

  private static final String COUNCIL_DAG =
      "{\"label\":\"Full Deliberation\",\"tasks\":["
          + "{\"id\":1,\"agent\":\"initial_response\",\"dependsOn\":[]},"
          + "{\"id\":2,\"agent\":\"peer_ranking\",\"dependsOn\":[1]},"
          + "{\"id\":3,\"agent\":\"agreement_analysis\",\"dependsOn\":[1]},"
          + "{\"id\":4,\"agent\":\"disagreement_analysis\",\"dependsOn\":[1]},"
          + "{\"id\":5,\"agent\":\"final_synthesis\",\"dependsOn\":[1,2]},"
          + "{\"id\":6,\"agent\":\"aggregate_rankings\",\"dependsOn\":[2]},"
          + "{\"id\":7,\"agent\":\"aggregate_agreements\",\"dependsOn\":[3]},"
          + "{\"id\":8,\"agent\":\"aggregate_disagreements\",\"dependsOn\":[4]}]}";

  @Test
  void councilDag_walksThreeLeveledGroupsInOrder() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(COUNCIL_DAG)));

    LlmCompilerPlanner planner =
        LlmCompilerPlanner.builder(mockLlm, COUNCIL_METADATA)
            .fallbackGoal("final_synthesis")
            .build();
    PlanningContext context = context(councilAgents());
    planner.init(context);

    List<List<String>> groups = collectGroups(planner, context);

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

  @Test
  void councilDag_finishesAfterLastGroup() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse(COUNCIL_DAG)));

    LlmCompilerPlanner planner =
        LlmCompilerPlanner.builder(mockLlm, COUNCIL_METADATA)
            .fallbackGoal("final_synthesis")
            .build();
    PlanningContext context = context(councilAgents());
    planner.init(context);

    PlannerAction action = planner.firstAction(context).blockingGet();
    int groupCount = 0;
    while (action instanceof PlannerAction.RunAgents) {
      groupCount++;
      action = planner.nextAction(context).blockingGet();
    }

    assertThat(groupCount).isEqualTo(3);
    assertThat(action).isInstanceOf(PlannerAction.DoneWithResult.class);
  }

  private static List<List<String>> collectGroups(
      LlmCompilerPlanner planner, PlanningContext context) {
    List<List<String>> groups = new ArrayList<>();
    PlannerAction action = planner.firstAction(context).blockingGet();
    while (action instanceof PlannerAction.RunAgents run) {
      groups.add(run.agents().stream().map(BaseAgent::name).toList());
      action = planner.nextAction(context).blockingGet();
    }
    return groups;
  }

  private static ImmutableList<BaseAgent> councilAgents() {
    return ALL_AGENTS.stream().map(SimpleTestAgent::new).collect(ImmutableList.toImmutableList());
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
