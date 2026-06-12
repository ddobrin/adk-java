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

import com.google.adk.agents.BaseAgent;
import com.google.adk.agents.InvocationContext;
import com.google.adk.agents.PlanningContext;
import com.google.adk.events.Event;
import com.google.adk.sessions.InMemorySessionService;
import com.google.adk.sessions.Session;
import com.google.common.collect.ImmutableList;
import io.reactivex.rxjava3.core.Flowable;
import java.util.concurrent.ConcurrentHashMap;
import org.junit.jupiter.api.Test;

/** Unit tests for the deterministic {@link PredicateJoiner}. */
class PredicateJoinerTest {

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

  @Test
  void decide_complete_returnsFinishWithInterpolatedResult() {
    PredicateJoiner joiner =
        new PredicateJoiner(
            state -> state.containsKey("synthesis"),
            state -> "result: " + state.get("synthesis"),
            state -> "no synthesis yet");

    ConcurrentHashMap<String, Object> state = new ConcurrentHashMap<>();
    state.put("synthesis", "the answer");
    JoinDecision decision = joiner.decide(context(state), 0).blockingGet();

    assertThat(decision).isInstanceOf(JoinDecision.Finish.class);
    assertThat(((JoinDecision.Finish) decision).result()).isEqualTo("result: the answer");
  }

  @Test
  void decide_incomplete_returnsReplanWithFeedback() {
    PredicateJoiner joiner =
        new PredicateJoiner(
            state -> state.containsKey("synthesis"),
            "done",
            "no synthesis produced yet — add a synthesis step");

    JoinDecision decision = joiner.decide(context(new ConcurrentHashMap<>()), 0).blockingGet();

    assertThat(decision).isInstanceOf(JoinDecision.Replan.class);
    assertThat(((JoinDecision.Replan) decision).feedback())
        .isEqualTo("no synthesis produced yet — add a synthesis step");
  }

  private static PlanningContext context(ConcurrentHashMap<String, Object> state) {
    InMemorySessionService sessionService = new InMemorySessionService();
    Session session = sessionService.createSession("test-app", "test-user").blockingGet();
    session.state().putAll(state);
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
