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

import com.google.adk.agents.PlanningContext;
import com.google.adk.events.Event;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Single;
import java.util.List;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@link Joiner} that asks an LLM to judge a finished plan's results and decide finish-vs-replan.
 *
 * <p>The model is shown the current state keys, the most recent events, and the original user
 * request, then asked to answer with exactly one of:
 *
 * <ul>
 *   <li>{@code FINISH: <summary>} — the deliverable is ready; the summary becomes the run's result.
 *   <li>{@code REPLAN: <feedback>} — another pass is needed; the feedback is fed back into the next
 *       compile to bias the new plan.
 * </ul>
 *
 * <p>Any LLM error (or an unparseable answer) defaults to {@link JoinDecision.Finish} — the loop
 * stops rather than spinning, mirroring {@link com.google.adk.planner.SupervisorPlanner}'s
 * fail-safe-to-Done behavior.
 */
public final class LlmJoiner implements Joiner {

  private static final Logger logger = LoggerFactory.getLogger(LlmJoiner.class);
  private static final int DEFAULT_MAX_EVENTS = 20;

  private final BaseLlm llm;
  private final String goalDescription;
  private final int maxEvents;

  /**
   * @param llm the LLM that judges completion
   * @param goalDescription a short description of what "done" means for this loop, injected into
   *     the prompt (nullable)
   */
  public LlmJoiner(BaseLlm llm, String goalDescription) {
    this(llm, goalDescription, DEFAULT_MAX_EVENTS);
  }

  public LlmJoiner(BaseLlm llm, String goalDescription, int maxEvents) {
    this.llm = llm;
    this.goalDescription = goalDescription;
    this.maxEvents = maxEvents;
  }

  @Override
  public Single<JoinDecision> decide(PlanningContext context, int replanCount) {
    LlmRequest request =
        LlmRequest.builder()
            .contents(
                ImmutableList.of(
                    Content.builder()
                        .role("user")
                        .parts(Part.fromText(buildPrompt(context, replanCount)))
                        .build()))
            .build();

    return llm.generateContent(request, false)
        .lastOrError()
        .map(response -> parse(extractText(response)))
        .onErrorReturn(
            error -> {
              logger.warn("LlmJoiner LLM call failed; finishing to avoid an unbounded loop", error);
              return new JoinDecision.Finish("Stopped: joiner LLM call failed");
            });
  }

  private String buildPrompt(PlanningContext context, int replanCount) {
    StringBuilder sb = new StringBuilder();
    sb.append(
        "A plan has finished executing. Decide whether the work is complete or another planning "
            + "pass is needed.\n");
    if (goalDescription != null && !goalDescription.isBlank()) {
      sb.append("\nGOAL: ").append(goalDescription).append("\n");
    }

    context
        .userContent()
        .ifPresent(
            content -> sb.append("\nOriginal user request: ").append(content.text()).append("\n"));

    sb.append("\nCurrent state keys: ").append(context.state().keySet()).append("\n");
    sb.append("Replan attempts so far: ").append(replanCount).append("\n");

    List<Event> events = context.events();
    if (!events.isEmpty()) {
      sb.append("\nRecent events:\n");
      int start = Math.max(0, events.size() - maxEvents);
      for (int i = start; i < events.size(); i++) {
        Event event = events.get(i);
        sb.append("- ")
            .append(event.author())
            .append(": ")
            .append(event.stringifyContent())
            .append("\n");
      }
    }

    sb.append(
        "\nRespond with EXACTLY one line, one of:\n"
            + "- FINISH: <one-sentence summary of the result>\n"
            + "- REPLAN: <specific guidance for what the next plan should add or fix>\n"
            + "\nPrefer FINISH unless another pass would materially improve the result.");
    return sb.toString();
  }

  private static JoinDecision parse(String text) {
    if (text == null || text.isBlank()) {
      return new JoinDecision.Finish("");
    }
    String trimmed = text.trim();
    String upper = trimmed.toUpperCase();
    if (upper.startsWith("REPLAN")) {
      return new JoinDecision.Replan(stripLabel(trimmed));
    }
    if (upper.startsWith("FINISH")) {
      return new JoinDecision.Finish(stripLabel(trimmed));
    }
    // Unlabeled answer: treat the whole thing as a finishing summary.
    return new JoinDecision.Finish(trimmed);
  }

  /** Strips a leading {@code FINISH}/{@code REPLAN} label and an optional {@code :} separator. */
  private static String stripLabel(String text) {
    int colon = text.indexOf(':');
    return (colon >= 0) ? text.substring(colon + 1).trim() : text;
  }

  private static String extractText(LlmResponse response) {
    return response.content().flatMap(Content::parts).stream()
        .flatMap(List::stream)
        .flatMap(part -> part.text().stream())
        .collect(Collectors.joining())
        .trim();
  }
}
