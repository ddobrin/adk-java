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
import com.google.adk.agents.PlanningContext;
import com.google.adk.events.Event;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmResponse;
import com.google.adk.sessions.InMemorySessionService;
import com.google.adk.sessions.Session;
import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link LlmJoiner} response parsing and fail-safe behavior. */
class LlmJoinerTest {

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
  void decide_finishResponse_parsesFinish() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse("FINISH: the deliverable is ready")));

    LlmJoiner joiner = new LlmJoiner(mockLlm, "produce a deliverable");
    JoinDecision decision = joiner.decide(context(), 0).blockingGet();

    assertThat(decision).isInstanceOf(JoinDecision.Finish.class);
    assertThat(((JoinDecision.Finish) decision).result()).isEqualTo("the deliverable is ready");
  }

  @Test
  void decide_replanResponse_parsesReplan() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse("REPLAN: add a verification step")));

    LlmJoiner joiner = new LlmJoiner(mockLlm, "produce a deliverable");
    JoinDecision decision = joiner.decide(context(), 0).blockingGet();

    assertThat(decision).isInstanceOf(JoinDecision.Replan.class);
    assertThat(((JoinDecision.Replan) decision).feedback()).isEqualTo("add a verification step");
  }

  @Test
  void decide_unlabeledResponse_treatedAsFinish() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.just(textResponse("everything looks good")));

    LlmJoiner joiner = new LlmJoiner(mockLlm, "produce a deliverable");
    JoinDecision decision = joiner.decide(context(), 0).blockingGet();

    assertThat(decision).isInstanceOf(JoinDecision.Finish.class);
    assertThat(((JoinDecision.Finish) decision).result()).isEqualTo("everything looks good");
  }

  @Test
  void decide_llmError_finishesSafely() {
    BaseLlm mockLlm = mock(BaseLlm.class);
    when(mockLlm.generateContent(any(), eq(false)))
        .thenReturn(Flowable.error(new RuntimeException("LLM down")));

    LlmJoiner joiner = new LlmJoiner(mockLlm, "produce a deliverable");
    JoinDecision decision = joiner.decide(context(), 0).blockingGet();

    assertThat(decision).isInstanceOf(JoinDecision.Finish.class);
    assertThat(((JoinDecision.Finish) decision).result()).contains("joiner LLM call failed");
  }

  private static LlmResponse textResponse(String text) {
    return LlmResponse.builder()
        .content(Content.builder().role("model").parts(Part.fromText(text)).build())
        .build();
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
