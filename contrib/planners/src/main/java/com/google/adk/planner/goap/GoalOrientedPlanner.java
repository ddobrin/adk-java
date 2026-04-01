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

package com.google.adk.planner.goap;

import com.google.adk.agents.BaseAgent;
import com.google.adk.agents.Planner;
import com.google.adk.agents.PlannerAction;
import com.google.adk.agents.PlanningContext;
import com.google.common.collect.ImmutableList;
import io.reactivex.rxjava3.core.Single;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A planner that resolves agent execution order based on input/output dependencies and a target
 * goal (output key).
 *
 * <p>Given agent metadata declaring what each agent reads (inputKeys) and writes (outputKey), this
 * planner uses backward-chaining dependency resolution to compute the execution path from initial
 * preconditions to the goal.
 *
 * <p>Example:
 *
 * <pre>
 *   Agent A: inputs=[], output="person"
 *   Agent B: inputs=[], output="sign"
 *   Agent C: inputs=["person", "sign"], output="horoscope"
 *   Agent D: inputs=["person", "horoscope"], output="writeup"
 *   Goal: "writeup"
 *
 *   Resolved groups: [A, B] → [C] → [D]
 *   (A and B are independent and run in parallel)
 * </pre>
 *
 * <p>Optionally validates at runtime that each agent group produces its expected output keys in
 * session state before proceeding to the next group. Enable via the {@code validateOutputs}
 * constructor parameter.
 */
public final class GoalOrientedPlanner implements Planner {

  private static final Logger logger = LoggerFactory.getLogger(GoalOrientedPlanner.class);

  private final String goal;
  private final List<AgentMetadata> metadata;
  private final boolean validateOutputs;
  // Mutable state — planners are used within a single reactive pipeline and are not thread-safe.
  private ImmutableList<ImmutableList<BaseAgent>> executionGroups;
  private Map<String, String> agentNameToOutputKey;
  private int cursor;

  public GoalOrientedPlanner(String goal, List<AgentMetadata> metadata) {
    this(goal, metadata, false);
  }

  public GoalOrientedPlanner(String goal, List<AgentMetadata> metadata, boolean validateOutputs) {
    this.goal = goal;
    this.metadata = metadata;
    this.validateOutputs = validateOutputs;
  }

  @Override
  public void init(PlanningContext context) {
    GoalOrientedSearchGraph graph = new GoalOrientedSearchGraph(metadata);
    ImmutableList<ImmutableList<String>> agentGroups =
        DependencyGraphSearch.searchGrouped(graph, metadata, context.state().keySet(), goal);

    logger.info("GoalOrientedPlanner resolved execution groups: {}", agentGroups);

    executionGroups =
        agentGroups.stream()
            .map(
                group ->
                    group.stream().map(context::findAgent).collect(ImmutableList.toImmutableList()))
            .collect(ImmutableList.toImmutableList());
    cursor = 0;

    agentNameToOutputKey = new HashMap<>();
    for (AgentMetadata m : metadata) {
      agentNameToOutputKey.put(m.agentName(), m.outputKey());
    }
  }

  @Override
  public Single<PlannerAction> firstAction(PlanningContext context) {
    cursor = 0;
    return selectNext();
  }

  @Override
  public Single<PlannerAction> nextAction(PlanningContext context) {
    if (validateOutputs && cursor > 0 && executionGroups != null) {
      ImmutableList<BaseAgent> previousGroup = executionGroups.get(cursor - 1);
      List<String> missingOutputs = new ArrayList<>();

      for (BaseAgent agent : previousGroup) {
        String expectedOutput = agentNameToOutputKey.get(agent.name());
        if (expectedOutput != null && !context.state().containsKey(expectedOutput)) {
          missingOutputs.add(agent.name() + " -> " + expectedOutput);
          logger.warn(
              "GoalOrientedPlanner: agent '{}' did not produce expected output key '{}'",
              agent.name(),
              expectedOutput);
        }
      }

      if (!missingOutputs.isEmpty()) {
        String message =
            "Execution stopped: missing expected outputs from previous group: "
                + String.join(", ", missingOutputs);
        logger.warn(message);
        return Single.just(new PlannerAction.DoneWithResult(message));
      }
    }
    return selectNext();
  }

  private Single<PlannerAction> selectNext() {
    if (executionGroups == null || cursor >= executionGroups.size()) {
      return Single.just(new PlannerAction.Done());
    }
    ImmutableList<BaseAgent> group = executionGroups.get(cursor++);
    return Single.just(new PlannerAction.RunAgents(group));
  }
}
