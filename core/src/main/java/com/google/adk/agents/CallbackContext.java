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

package com.google.adk.agents;

import com.google.adk.events.EventActions;
import com.google.adk.sessions.State;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Maybe;
import java.util.Optional;

/** The context of various callbacks for an agent invocation. */
public class CallbackContext extends ReadonlyContext {

  protected EventActions eventActions;
  private final State state;

  /**
   * Initializes callback context.
   *
   * @param invocationContext Current invocation context.
   * @param eventActions Callback event actions.
   */
  public CallbackContext(InvocationContext invocationContext, EventActions eventActions) {
    super(invocationContext);
    this.eventActions = eventActions != null ? eventActions : EventActions.builder().build();
    this.state = new State(invocationContext.session().state(), this.eventActions.stateDelta());
  }

  /** Returns the delta-aware state of the current callback. */
  @Override
  public State state() {
    return state;
  }

  /** Returns the user content that initiated this invocation. */
  public Optional<Content> userContent() {
    return invocationContext.userContent();
  }

  /** Returns the EventActions associated with this context. */
  public EventActions eventActions() {
    return eventActions;
  }

  /**
   * Loads an artifact from the artifact service associated with the current session.
   *
   * @param filename Artifact file name.
   * @param version Artifact version (optional).
   * @return loaded part, or empty if not found.
   * @throws IllegalStateException if the artifact service is not initialized.
   */
  public Maybe<Part> loadArtifact(String filename, Optional<Integer> version) {
    if (invocationContext.artifactService() == null) {
      throw new IllegalStateException("Artifact service is not initialized.");
    }
    return invocationContext
        .artifactService()
        .loadArtifact(
            invocationContext.appName(),
            invocationContext.userId(),
            invocationContext.session().id(),
            filename,
            version);
  }

  /**
   * Saves an artifact and records it as a delta for the current session.
   *
   * @param filename Artifact file name.
   * @param artifact Artifact content to save.
   * @throws IllegalStateException if the artifact service is not initialized.
   */
  public void saveArtifact(String filename, Part artifact) {
    if (invocationContext.artifactService() == null) {
      throw new IllegalStateException("Artifact service is not initialized.");
    }
    var unused =
        invocationContext
            .artifactService()
            .saveArtifact(
                invocationContext.appName(),
                invocationContext.userId(),
                invocationContext.session().id(),
                filename,
                artifact);
    this.eventActions.artifactDelta().put(filename, artifact);
  }
}
