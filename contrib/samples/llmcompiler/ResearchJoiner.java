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

import com.google.adk.planner.llmcompiler.Joiner;
import com.google.adk.planner.llmcompiler.PredicateJoiner;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The finish-vs-replan heuristic for the research sample, expressed as a deterministic {@link
 * Joiner}.
 *
 * <p>After the planner runs a full plan, it asks a {@link Joiner}: is the deliverable good enough
 * (→ {@code Finish}), or should we run another plan with some guidance (→ {@code Replan})? This
 * sample uses a {@link PredicateJoiner}, so the judgment is a pure function of session state — no
 * second LLM call, fully deterministic and unit-testable.
 *
 * <p>This is the one genuinely judgment-driven choice in the whole pipeline: "when is the report
 * good enough, and what should the next pass do differently?" The two methods below — {@link
 * #isReportComplete} and {@link #replanFeedback} — are the natural place to encode that judgment.
 */
public final class ResearchJoiner {

  private ResearchJoiner() {}

  /** State key the {@code synthesize} agent writes its final report under (see {@link ResearchAgents}). */
  private static final String REPORT_KEY = "report";

  /** A report below this many characters is treated as a stub, not a deliverable. */
  private static final int MIN_REPORT_CHARS = 600;

  /** Lowercase markers signalling the *established* angle survived into the report. */
  private static final List<String> ESTABLISHED_MARKERS =
      List.of("establish", "consensus", "well-supported", "agreed", "evidence");

  /** Lowercase markers signalling the *open questions* angle survived into the report. */
  private static final List<String> OPEN_QUESTION_MARKERS =
      List.of("open question", "uncertain", "debate", "counterpoint", "unsettled", "disagree");

  /** Lowercase markers signalling the report actually concludes rather than just listing. */
  private static final List<String> CONCLUSION_MARKERS =
      List.of("conclusion", "conclude", "in summary", "overall", "on balance", "taken together");

  /** Builds the joiner the planner consults once a plan is exhausted. */
  public static Joiner create() {
    return new PredicateJoiner(
        ResearchJoiner::isReportComplete,
        ResearchJoiner::finishResult,
        ResearchJoiner::replanFeedback);
  }

  // ------------------------------------------------------------------------------------------------
  // CONTRIBUTION POINT — the replan-vs-finish heuristic.
  //
  // This judges the *deliverable*, not the pipeline: the DAG leveling already guarantees every
  // agent ran in dependency order, so the question left to the Joiner is purely semantic — is the
  // report good enough to ship? We test four independent signals and AND them together:
  //
  //   1. existence   — 'synthesize' actually wrote a non-blank report
  //   2. substance   — it clears MIN_REPORT_CHARS (not a stub or an apology)
  //   3. balance     — BOTH the established and the open-questions angles survived into it
  //   4. closure     — it actually concludes rather than just listing
  //
  // isReportComplete() and replanFeedback() read the SAME four signals, so when the report is not
  // done the feedback names exactly which signal failed — that is what makes the next plan target
  // the real gap instead of blindly retrying.
  //
  // Tune me: adjust MIN_REPORT_CHARS and the *_MARKERS sets above, or replace the keyword scan with
  // a structured quality marker that an upstream agent writes into state.
  //
  // Constraint: a PredicateJoiner sees session state only (not replanCount) — the planner already
  // bounds retries via maxReplans, so you decide *whether* the work is done, not *how many times*
  // to try.
  // ------------------------------------------------------------------------------------------------

  /**
   * Returns true when the report is good enough to ship (→ the planner finishes).
   *
   * <p>All four quality signals must hold; any miss sends the planner back for another pass.
   *
   * @param state current session state (agent outputs keyed by {@code outputKey})
   */
  static boolean isReportComplete(Map<String, Object> state) {
    String report = asText(state.get(REPORT_KEY));
    if (report == null) {
      return false; // 'synthesize' never ran, or produced no text.
    }
    String body = report.strip();
    String haystack = body.toLowerCase(Locale.ROOT);

    boolean substantial = body.length() >= MIN_REPORT_CHARS;
    boolean coversEstablished = mentionsAny(haystack, ESTABLISHED_MARKERS);
    boolean coversOpen = mentionsAny(haystack, OPEN_QUESTION_MARKERS);
    boolean concludes = mentionsAny(haystack, CONCLUSION_MARKERS);

    return substantial && coversEstablished && coversOpen && concludes;
  }

  /**
   * Builds the guidance handed back to the planner for the next pass (→ used on {@code Replan}).
   *
   * <p>Names only the signals that actually failed, so the next plan addresses the real gap rather
   * than retrying blindly.
   *
   * @param state current session state, so feedback can reference what is missing
   */
  static String replanFeedback(Map<String, Object> state) {
    String report = asText(state.get(REPORT_KEY));
    if (report == null) {
      return "No report was produced. Ensure the plan ends in the 'synthesize' agent so the"
          + " 'report' key is written.";
    }
    String body = report.strip();
    String haystack = body.toLowerCase(Locale.ROOT);

    List<String> gaps = new ArrayList<>();
    if (body.length() < MIN_REPORT_CHARS) {
      gaps.add(
          "it is too thin ("
              + body.length()
              + " chars) — gather more notes and summarize them before synthesizing");
    }
    if (!mentionsAny(haystack, ESTABLISHED_MARKERS)) {
      gaps.add("the established/consensus findings are missing — keep 'summarizeA' in the plan");
    }
    if (!mentionsAny(haystack, OPEN_QUESTION_MARKERS)) {
      gaps.add("open questions and counterpoints are missing — keep 'summarizeB' in the plan");
    }
    if (!mentionsAny(haystack, CONCLUSION_MARKERS)) {
      gaps.add("there is no conclusion — 'synthesize' must end with a weighed conclusion");
    }
    if (gaps.isEmpty()) {
      // Unreachable when called on Replan (isReportComplete would be true), but stay safe.
      return "The report needs another pass for overall quality.";
    }
    return "The report is incomplete: " + String.join("; ", gaps) + ".";
  }

  /** The final deliverable text, returned to the caller when the report is complete. */
  static String finishResult(Map<String, Object> state) {
    String report = asText(state.get(REPORT_KEY));
    return (report != null) ? report : "(no report produced)";
  }

  /** Returns the value as text, or {@code null} if it is absent or a blank/non-string value. */
  private static String asText(Object value) {
    return (value instanceof String text && !text.isBlank()) ? text : null;
  }

  /** Case-insensitive containment: true if {@code haystack} contains any of {@code markers}. */
  private static boolean mentionsAny(String haystack, List<String> markers) {
    for (String marker : markers) {
      if (haystack.contains(marker)) {
        return true;
      }
    }
    return false;
  }
}
