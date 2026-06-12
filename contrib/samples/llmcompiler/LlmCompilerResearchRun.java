// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.example.llmcompiler;

import com.google.adk.agents.PlannerAgent;
import com.google.adk.artifacts.InMemoryArtifactService;
import com.google.adk.events.Event;
import com.google.adk.memory.InMemoryMemoryService;
import com.google.adk.models.Gemini;
import com.google.adk.planner.llmcompiler.LlmCompilerPlanner;
import com.google.adk.runner.Runner;
import com.google.adk.sessions.InMemorySessionService;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * End-to-end runner for the Plan-and-Execute / LLMCompiler sample.
 *
 * <p>Wiring:
 *
 * <ol>
 *   <li>A {@link Gemini} planning LLM authors a task DAG over the {@link ResearchAgents#CATALOG}.
 *   <li>{@link LlmCompilerPlanner} levels the DAG and runs the groups — {@code summarizeA} and
 *       {@code summarizeB} run in parallel after {@code gather}.
 *   <li>{@link ResearchJoiner} decides finish-vs-replan once the plan is exhausted, bounded by
 *       {@code maxReplans}.
 * </ol>
 *
 * <p>The topic is seeded into session state so the {@code gather} agent's {@code {topic}}
 * placeholder resolves on its first run.
 */
public final class LlmCompilerResearchRun {

  private final String userId;
  private final String sessionId;
  private final Runner runner;

  private LlmCompilerResearchRun(String topic) {
    String appName = "llmcompiler-research-app";
    this.userId = "research-user";
    this.sessionId = UUID.randomUUID().toString();

    // The planning LLM that authors the DAG. Credentials come from the environment (see README).
    Gemini planningLlm = Gemini.builder().modelName(ResearchAgents.MODEL).build();

    LlmCompilerPlanner planner =
        LlmCompilerPlanner.builder(planningLlm, ResearchAgents.CATALOG)
            .fallbackGoal(ResearchAgents.GOAL) // deterministic GOAP fallback target
            .joiner(ResearchJoiner.create()) // semantic finish-vs-replan (the contribution point)
            .maxReplans(2)
            .defaultInstruction("Research the topic and produce a balanced report.")
            .build();

    PlannerAgent rootAgent =
        PlannerAgent.builder()
            .name("research_pipeline")
            .description("Plans and runs a research fan-out, then synthesizes a report.")
            .subAgents(ResearchAgents.AGENTS)
            .planner(planner)
            .maxIterations(20)
            .build();

    InMemorySessionService sessionService = new InMemorySessionService();
    this.runner =
        new Runner(
            rootAgent,
            appName,
            new InMemoryArtifactService(),
            sessionService,
            new InMemoryMemoryService());

    // Seed the topic so the level-0 'gather' agent has its input available immediately.
    ConcurrentMap<String, Object> initialState = new ConcurrentHashMap<>();
    initialState.put("topic", topic);
    var unused =
        sessionService.createSession(appName, userId, initialState, sessionId).blockingGet();
  }

  private void run(String instruction) {
    System.out.println("You> " + instruction);
    Content userMessage =
        Content.builder()
            .role("user")
            .parts(ImmutableList.of(Part.builder().text(instruction).build()))
            .build();

    Flowable<Event> eventStream = runner.runAsync(userId, sessionId, userMessage);
    List<Event> events = Lists.newArrayList(eventStream.blockingIterable());

    for (Event event : events) {
      String author = (event.author() == null) ? "agent" : event.author();
      String content = event.stringifyContent().stripTrailing();
      if (!content.isEmpty()) {
        System.out.println(author + "> " + content);
      }
    }
  }

  public static void main(String[] args) {
    String topic =
        (args.length > 0)
            ? String.join(" ", args)
            : "the trade-offs of small vs. large language models for agentic workflows";
    LlmCompilerResearchRun app = new LlmCompilerResearchRun(topic);
    app.run("Research \"" + topic + "\" and produce a balanced report.");
  }
}
