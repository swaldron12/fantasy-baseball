"""Local web app for roster category projections and lineup optimization."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.roster_value import (
    analyze_lineup_slots,
    analyze_roster,
    bedrock_configuration_status,
    generate_projection_chat_response,
    search_players,
)


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fantasy Baseball Lineup Optimizer</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #111827;
      --muted: #64748b;
      --line: #d7e0ea;
      --surface: #ffffff;
      --surface-soft: #f8fafc;
      --soft: #eef4fb;
      --accent: #2563eb;
      --accent-dark: #1d4ed8;
      --accent-soft: #dbeafe;
      --accent-line: #b7cdfb;
      --good: #15803d;
      --bad: #b42318;
      --warn-bg: #fffbeb;
      --warn-line: #f2d58b;
      --shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f6fb;
      color: var(--ink);
      line-height: 1.35;
    }

    button, input, textarea {
      font: inherit;
    }

    header {
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 0 rgba(15, 23, 42, 0.03);
    }

    .header-inner {
      max-width: 1420px;
      margin: 0 auto;
      padding: 18px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 20px;
    }

    h1 {
      margin: 0;
      font-size: 23px;
      font-weight: 760;
      letter-spacing: 0;
    }

    .status {
      min-width: 300px;
      border: 1px solid var(--accent-line);
      background: #f0f6ff;
      border-radius: 8px;
      padding: 9px 12px;
      color: #335176;
      font-size: 13px;
      line-height: 1.35;
    }

    main {
      max-width: 1420px;
      margin: 0 auto;
      padding: 22px 24px;
      display: grid;
      grid-template-columns: 520px minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: var(--surface-soft);
    }

    .panel-title {
      margin: 0;
      font-size: 15px;
      font-weight: 730;
      letter-spacing: 0;
    }

    .panel-body {
      padding: 16px;
    }

    .slot-section {
      margin-bottom: 18px;
    }

    .slot-section:last-child {
      margin-bottom: 0;
    }

    .slot-heading {
      margin: 0 0 9px;
      color: #334155;
      font-size: 13px;
      font-weight: 720;
      letter-spacing: 0;
    }

    .slot-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 9px;
    }

    .slot-row {
      position: relative;
      display: grid;
      grid-template-columns: 46px minmax(0, 1fr);
      align-items: center;
      gap: 8px;
    }

    .slot-row.has-eligibility {
      grid-template-columns: 46px minmax(0, 1fr) 76px;
    }

    .slot-label {
      color: #475569;
      font-size: 12px;
      font-weight: 740;
      text-align: right;
    }

    input,
    textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfdff;
      color: var(--ink);
      padding: 7px 9px;
      transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease;
    }

    input {
      height: 36px;
    }

    textarea {
      min-height: 74px;
      resize: vertical;
      line-height: 1.4;
    }

    input:focus,
    textarea:focus {
      outline: none;
      border-color: var(--accent);
      background: white;
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.16);
    }

    .eligibility-input {
      text-transform: uppercase;
      text-align: center;
      color: #334155;
    }

    .suggestions {
      position: absolute;
      z-index: 5;
      left: 54px;
      right: 0;
      top: 38px;
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 16px 28px rgba(15, 23, 42, 0.15);
      max-height: 220px;
      overflow: auto;
    }

    .slot-row.has-eligibility .suggestions {
      right: 84px;
    }

    .suggestion {
      width: 100%;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 10px;
      border: 0;
      border-bottom: 1px solid #edf0f4;
      background: white;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
      border-radius: 0;
    }

    .suggestion:hover {
      background: #eff6ff;
      color: var(--ink);
    }

    .suggestion small {
      color: var(--muted);
      white-space: nowrap;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 9px;
      margin-top: 16px;
    }

    button {
      min-height: 36px;
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 7px;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 700;
      transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease, color 120ms ease;
    }

    button.secondary {
      background: white;
      color: var(--accent-dark);
      border-color: var(--accent-line);
    }

    button.toggle {
      background: white;
      color: var(--accent-dark);
      border-color: var(--line);
    }

    button.toggle.active {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }

    button:disabled {
      cursor: progress;
      opacity: 0.68;
    }

    button:hover {
      background: var(--accent-dark);
      border-color: var(--accent-dark);
      color: white;
      box-shadow: 0 8px 18px rgba(37, 99, 235, 0.18);
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
      padding: 11px 12px;
      min-height: 74px;
    }

    .metric-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }

    .metric-value {
      font-size: 22px;
      font-weight: 760;
      font-variant-numeric: tabular-nums;
      letter-spacing: 0;
    }

    .good { color: var(--good); }
    .bad { color: var(--bad); }

    .value-guide {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
      padding: 11px 12px;
      margin: -4px 0 16px;
      color: #334155;
      font-size: 13px;
      line-height: 1.45;
    }

    .value-guide strong {
      color: var(--ink);
      font-weight: 740;
    }

    .output-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 16px;
    }

    .summary-table,
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th, td {
      padding: 8px 9px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }

    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f1f5f9;
      color: #334155;
      font-weight: 720;
    }

    tbody tr:nth-child(even) {
      background: #fbfdff;
    }

    tbody tr:hover {
      background: #eff6ff;
    }

    .table-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: auto;
      margin-bottom: 16px;
      background: white;
    }

    .view-toggle {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 11px;
    }

    .number {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .pill {
      display: inline-block;
      padding: 2px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f8fafc;
      font-size: 12px;
      text-transform: capitalize;
    }

    .analysis {
      border: 1px solid var(--accent-line);
      border-radius: 8px;
      background: #f4f8ff;
      padding: 13px 14px;
      line-height: 1.45;
      font-size: 14px;
      margin-bottom: 16px;
      color: #1f2937;
    }

    .analysis-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      color: #1e3a8a;
      font-size: 13px;
      font-weight: 760;
    }

    .analysis-text {
      white-space: pre-line;
    }

    .chat {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      margin-bottom: 16px;
      overflow: hidden;
    }

    .chat-log {
      min-height: 160px;
      max-height: 340px;
      overflow: auto;
      padding: 12px;
      background: #fbfdff;
      border-bottom: 1px solid var(--line);
    }

    .chat-message {
      width: min(88%, 680px);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      margin-bottom: 10px;
      white-space: pre-line;
      font-size: 14px;
    }

    .chat-message.user {
      margin-left: auto;
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }

    .chat-message.assistant {
      background: white;
      color: #1f2937;
    }

    .chat-form {
      padding: 12px;
      display: grid;
      gap: 9px;
    }

    .chat-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }

    .chat-hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .source-pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border: 1px solid var(--accent-line);
      border-radius: 999px;
      background: white;
      color: #335176;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }

    .unmatched {
      border: 1px solid var(--warn-line);
      background: var(--warn-bg);
      border-radius: 8px;
      color: #805400;
      padding: 10px;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 16px;
    }

    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: var(--surface-soft);
      color: var(--muted);
      padding: 28px;
      text-align: center;
    }

    .hidden {
      display: none;
    }

    @media (max-width: 1040px) {
      main {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 680px) {
      .header-inner {
        flex-direction: column;
        align-items: stretch;
      }

      .status {
        min-width: 0;
      }

      main {
        padding: 14px;
      }

      .slot-grid,
      .metrics,
      .output-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>Fantasy Baseball Lineup Optimizer</h1>
      <div id="data-status" class="status">Projection data will load after analysis.</div>
    </div>
  </header>

  <main>
    <section class="panel">
      <div class="panel-header">
        <h2 class="panel-title">Roster Slots</h2>
      </div>
      <div class="panel-body">
        <div class="slot-section">
          <h3 class="slot-heading">Hitters</h3>
          <div id="hitter-slots" class="slot-grid"></div>
        </div>
        <div class="slot-section">
          <h3 class="slot-heading">Pitchers</h3>
          <div id="pitcher-slots" class="slot-grid"></div>
        </div>
        <div class="slot-section">
          <h3 class="slot-heading">Bench</h3>
          <div id="bench-slots" class="slot-grid"></div>
        </div>
        <div class="actions">
          <button id="analyze-button" type="button">Optimize Lineup</button>
          <button id="sample-button" class="secondary" type="button">Load Sample</button>
          <button id="clear-button" class="secondary" type="button">Clear</button>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2 class="panel-title">Weekly Projection</h2>
      </div>
      <div class="panel-body">
        <div id="empty-state" class="empty">Run the optimizer to view category projections.</div>
        <div id="results" class="hidden">
          <div class="metrics">
            <div class="metric">
              <div class="metric-label">Current Lineup Value</div>
              <div id="current-value" class="metric-value">0.00</div>
            </div>
            <div class="metric">
              <div class="metric-label">Optimized Lineup Value</div>
              <div id="optimized-value" class="metric-value">0.00</div>
            </div>
            <div class="metric">
              <div class="metric-label">Lineup Value Gain</div>
              <div id="value-gain" class="metric-value">0.00</div>
            </div>
            <div class="metric">
              <div class="metric-label">Active Players</div>
              <div id="active-count" class="metric-value">0</div>
            </div>
          </div>
          <div id="value-guide" class="value-guide"></div>

          <div class="analysis">
            <div class="analysis-header">
              <span>Lineup Analysis</span>
              <span id="analysis-source" class="source-pill"></span>
            </div>
            <div id="analysis-box" class="analysis-text"></div>
          </div>

          <div class="chat">
            <div class="analysis-header panel-header">
              <span>Projection Chat</span>
              <span id="chat-source" class="source-pill">bedrock ready</span>
            </div>
            <div id="chat-log" class="chat-log">
              <div class="chat-message assistant">Ask a follow-up about start/sit decisions, category strengths, player projections, or matchup evidence.</div>
            </div>
            <div class="chat-form">
              <textarea id="chat-question" placeholder="Example: Why is Josh Jung preferred over Michael Busch?"></textarea>
              <div class="chat-actions">
                <span class="chat-hint">Answers use the current roster, optimized lineup, projections, and verified matchup notes.</span>
                <button id="chat-button" type="button">Ask</button>
              </div>
            </div>
          </div>
          <div id="availability-box" class="unmatched hidden"></div>
          <div id="unmatched-box" class="unmatched hidden"></div>

          <div class="output-grid">
            <div class="table-wrap">
              <table class="summary-table">
                <thead><tr><th colspan="2">Optimized Hitting</th></tr></thead>
                <tbody id="batting-summary"></tbody>
              </table>
            </div>
            <div class="table-wrap">
              <table class="summary-table">
                <thead><tr><th colspan="2">Optimized Pitching</th></tr></thead>
                <tbody id="pitching-summary"></tbody>
              </table>
            </div>
          </div>

          <div class="view-toggle" aria-label="Projection table view">
            <button id="show-hitters" class="toggle active" type="button">Hitters</button>
            <button id="show-pitchers" class="toggle" type="button">Pitchers</button>
          </div>

          <div id="hitter-table-wrap" class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Slot</th>
                  <th>Player</th>
                  <th>Team</th>
                  <th>Elig.</th>
                  <th class="number">Player Value</th>
                  <th class="number">AB</th>
                  <th class="number">H</th>
                  <th class="number">HR</th>
                  <th class="number">R</th>
                  <th class="number">RBI</th>
                  <th class="number">SB</th>
                  <th class="number">AVG</th>
                  <th class="number">OPS</th>
                </tr>
              </thead>
              <tbody id="hitter-projection-table"></tbody>
            </table>
          </div>

          <div id="pitcher-table-wrap" class="table-wrap hidden">
            <table>
              <thead>
                <tr>
                  <th>Slot</th>
                  <th>Player</th>
                  <th>Team</th>
                  <th class="number">Player Value</th>
                  <th class="number">IP</th>
                  <th class="number">W</th>
                  <th class="number">SV</th>
                  <th class="number">HLD</th>
                  <th class="number">K</th>
                  <th class="number">ERA</th>
                  <th class="number">WHIP</th>
                </tr>
              </thead>
              <tbody id="pitcher-projection-table"></tbody>
            </table>
          </div>

          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Move</th>
                  <th>Slot</th>
                  <th>Start</th>
                  <th>Sit</th>
                  <th class="number">Value Gain</th>
                </tr>
              </thead>
              <tbody id="changes-table"></tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const hitterSlots = [
      { id: "C", label: "C" },
      { id: "1B", label: "1B" },
      { id: "2B", label: "2B" },
      { id: "3B", label: "3B" },
      { id: "SS", label: "SS" },
      { id: "OF1", label: "OF" },
      { id: "OF2", label: "OF" },
      { id: "OF3", label: "OF" },
      { id: "UTIL", label: "UTIL" }
    ];
    const pitcherSlots = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"].map(slot => ({ id: slot, label: slot }));
    const benchSlots = ["BN1", "BN2", "BN3"].map(slot => ({ id: slot, label: slot }));

    const slotState = new Map();
    const dataStatus = document.getElementById("data-status");
    const results = document.getElementById("results");
    const emptyState = document.getElementById("empty-state");
    let chatHistory = [];

    buildSlots("hitter-slots", hitterSlots, "hitter");
    buildSlots("pitcher-slots", pitcherSlots, "pitcher");
    buildSlots("bench-slots", benchSlots, "bench");

    document.getElementById("analyze-button").addEventListener("click", analyzeLineup);
    document.getElementById("sample-button").addEventListener("click", loadSample);
    document.getElementById("clear-button").addEventListener("click", clearSlots);
    document.getElementById("show-hitters").addEventListener("click", () => showProjectionTable("hitters"));
    document.getElementById("show-pitchers").addEventListener("click", () => showProjectionTable("pitchers"));
    document.getElementById("chat-button").addEventListener("click", sendChatQuestion);
    document.getElementById("chat-question").addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        sendChatQuestion();
      }
    });

    function buildSlots(containerId, slots, slotType) {
      const container = document.getElementById(containerId);
      slots.forEach(slot => {
        const slotId = slot.id;
        const slotLabel = slot.label;
        const hasEligibility = slotType !== "pitcher";
        const defaultEligibility = slotType === "hitter" && slotLabel !== "UTIL" ? slotLabel : "";
        const row = document.createElement("div");
        row.className = hasEligibility ? "slot-row has-eligibility" : "slot-row";
        row.innerHTML = `
          <label class="slot-label" for="slot-${slotId}">${slotLabel}</label>
          <input id="slot-${slotId}" data-slot="${slotId}" data-label="${slotLabel}" data-type="${slotType}" autocomplete="off">
          ${hasEligibility ? `<input class="eligibility-input" aria-label="${slotLabel} eligibility" value="${defaultEligibility}" autocomplete="off">` : ""}
          <div class="suggestions hidden"></div>
        `;
        container.appendChild(row);
        const input = row.querySelector("input");
        const eligibilityInput = row.querySelector(".eligibility-input");
        const suggestions = row.querySelector(".suggestions");
        input.addEventListener("input", debounce(() => updateSuggestions(input, suggestions), 180));
        input.addEventListener("focus", () => updateSuggestions(input, suggestions));
        document.addEventListener("click", event => {
          if (!row.contains(event.target)) suggestions.classList.add("hidden");
        });
        slotState.set(slotId, {
          slot: slotId,
          label: slotLabel,
          slotType,
          input,
          eligibilityInput,
          defaultEligibility
        });
      });
    }

    async function updateSuggestions(input, suggestions) {
      const q = input.value.trim();
      if (q.length < 2) {
        suggestions.classList.add("hidden");
        suggestions.innerHTML = "";
        return;
      }

      const slotType = input.dataset.type;
      const playerType = slotType === "pitcher" ? "pitcher" : (slotType === "hitter" ? "batter" : "");
      const url = `/api/search?q=${encodeURIComponent(q)}&player_type=${encodeURIComponent(playerType)}`;
      const response = await fetch(url);
      const payload = await response.json();

      suggestions.innerHTML = "";
      payload.players.forEach(player => {
        const button = document.createElement("button");
        button.className = "suggestion";
        button.type = "button";
        const status = player.is_available === false ? ` ${escapeHtml(player.availability_status || "unavailable")}` : "";
        button.innerHTML = `<span>${escapeHtml(player.name)}</span><small>${escapeHtml(player.team || "")} ${escapeHtml(player.player_type || "")}${status}</small>`;
        button.addEventListener("click", () => {
          input.value = player.name;
          const eligibilityInput = input.parentElement.querySelector(".eligibility-input");
          if (eligibilityInput && player.eligible_positions && !eligibilityInput.value.trim()) {
            eligibilityInput.value = player.eligible_positions;
          }
          suggestions.classList.add("hidden");
        });
        suggestions.appendChild(button);
      });
      suggestions.classList.toggle("hidden", payload.players.length === 0);
    }

    async function analyzeLineup() {
      const slots = collectSlots();
      const button = document.getElementById("analyze-button");
      button.disabled = true;
      button.textContent = "Optimizing...";

      try {
        const response = await fetch("/api/lineup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slots })
        });
        if (!response.ok) throw new Error(`Request failed: ${response.status}`);
        const payload = await response.json();
        renderResults(payload);
      } catch (error) {
        emptyState.textContent = error.message;
        emptyState.classList.remove("hidden");
        results.classList.add("hidden");
      } finally {
        button.disabled = false;
        button.textContent = "Optimize Lineup";
      }
    }

    function collectSlots() {
      return Array.from(slotState.values()).map(item => ({
        slot_id: item.slot,
        slot_label: item.label,
        slot_type: item.slotType,
        player_name: item.input.value.trim(),
        eligible_positions: item.eligibilityInput ? item.eligibilityInput.value.trim() : ""
      })).filter(slot => slot.player_name);
    }

    function renderResults(payload) {
      const current = payload.current;
      const optimized = payload.optimized.summary;
      const valueGain = Number(optimized.total_expected_fantasy_value || 0) - Number(current.total_expected_fantasy_value || 0);

      emptyState.classList.add("hidden");
      results.classList.remove("hidden");

      setMetric("current-value", current.total_expected_fantasy_value, true);
      setMetric("optimized-value", optimized.total_expected_fantasy_value, true);
      setMetric("value-gain", valueGain, true);
      setMetric("active-count", optimized.active_players, false, 0);

      const metadata = payload.metadata;
      dataStatus.textContent = metadata.schedule_start && metadata.schedule_end
        ? `Schedule: ${metadata.schedule_start} to ${metadata.schedule_end}. Projection pool: ${metadata.player_count} players.`
        : `Projection pool: ${metadata.player_count} players.`;
      renderValueGuide(metadata.value_explanation);

      const formattedAnalysis = formatAnalysisText(payload.analysis);
      document.getElementById("analysis-box").textContent = formattedAnalysis.text;
      document.getElementById("analysis-source").textContent = formattedAnalysis.source;
      resetChat();
      renderSummaryTable("batting-summary", optimized.batting, [
        ["AB", "ab"], ["H", "h"], ["HR", "hr"], ["R", "r"], ["RBI", "rbi"], ["SB", "sb"], ["AVG", "avg"], ["OPS", "ops"]
      ]);
      renderSummaryTable("pitching-summary", optimized.pitching, [
        ["IP", "ip"], ["W", "w"], ["SV", "sv"], ["HLD", "hld"], ["K", "k"], ["ERA", "era"], ["WHIP", "whip"]
      ]);
      renderProjectionTables(payload.optimized.lineup || []);
      renderChanges(payload.optimized.changes || []);
      renderAvailability(payload.optimized.unavailable || []);
      renderUnmatched(payload.unmatched || []);
    }

    async function sendChatQuestion() {
      const input = document.getElementById("chat-question");
      const button = document.getElementById("chat-button");
      const question = input.value.trim();
      if (!question) return;

      const slots = collectSlots();
      if (!slots.length) {
        appendChatMessage("assistant", "Run the optimizer with a roster first, then ask about those projections.");
        return;
      }

      appendChatMessage("user", question);
      chatHistory.push({ role: "user", text: question });
      input.value = "";
      button.disabled = true;
      button.textContent = "Asking...";

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slots, question, history: chatHistory.slice(-8) })
        });
        if (!response.ok) throw new Error(`Chat failed: ${response.status}`);
        const payload = await response.json();
        const chat = payload.chat || {};
        let text = chat.text || "I could not generate a chat response.";
        if (chat.source === "local_fallback" && chat.bedrock && chat.bedrock.reason) {
          text += `\n\nBedrock status: ${chat.bedrock.reason}`;
        }
        appendChatMessage("assistant", text);
        chatHistory.push({ role: "assistant", text });
        document.getElementById("chat-source").textContent = (chat.source || "unknown").replaceAll("_", " ");
      } catch (error) {
        appendChatMessage("assistant", error.message);
      } finally {
        button.disabled = false;
        button.textContent = "Ask";
      }
    }

    function resetChat() {
      chatHistory = [];
      document.getElementById("chat-source").textContent = "bedrock ready";
      const log = document.getElementById("chat-log");
      log.innerHTML = "";
      appendChatMessage("assistant", "Ask a follow-up about start/sit decisions, category strengths, player projections, or matchup evidence.");
    }

    function appendChatMessage(role, text) {
      const log = document.getElementById("chat-log");
      const message = document.createElement("div");
      message.className = `chat-message ${role}`;
      message.textContent = text;
      log.appendChild(message);
      log.scrollTop = log.scrollHeight;
    }

    function renderSummaryTable(id, data, fields) {
      const body = document.getElementById(id);
      body.innerHTML = "";
      fields.forEach(([label, key]) => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>${label}</td><td class="number">${formatCategory(key, data[key])}</td>`;
        body.appendChild(row);
      });
    }

    function renderValueGuide(explanation) {
      const guide = document.getElementById("value-guide");
      const valueText = explanation || {};
      guide.innerHTML = `
        <strong>Value is an index, not fantasy points.</strong>
        ${escapeHtml(valueText.player_value || "Player Value compares hitters to hitters and pitchers to pitchers; 0 is average for that player type.")}
        <strong>Current</strong> is your entered active lineup.
        <strong>Optimized</strong> is the best legal lineup from active plus bench.
        <strong>Gain</strong> is optimized minus current.
      `;
    }

    function formatAnalysisText(analysis) {
      const source = analysis.source || "unknown";
      let text = analysis.text || "";
      if (source === "local_fallback" && analysis.bedrock && analysis.bedrock.reason) {
        text += `\n\nBedrock status: ${analysis.bedrock.reason}`;
      }
      return {
        text,
        source: source.replaceAll("_", " ")
      };
    }

    function renderProjectionTables(players) {
      renderHitterProjectionTable(players.filter(player => player.player_type === "batter"));
      renderPitcherProjectionTable(players.filter(player => player.player_type === "pitcher"));
    }

    function renderHitterProjectionTable(players) {
      const body = document.getElementById("hitter-projection-table");
      body.innerHTML = "";
      if (!players.length) {
        const row = document.createElement("tr");
        row.innerHTML = `<td colspan="13">No optimized hitters found.</td>`;
        body.appendChild(row);
        return;
      }
      players.forEach(player => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${escapeHtml(player.optimized_slot || "")}</td>
          <td>${escapeHtml(player.name || "")}</td>
          <td>${escapeHtml(player.team || "")}</td>
          <td>${escapeHtml(player.eligible_positions_label || "")}</td>
          <td class="number">${formatSigned(player.expected_fantasy_value)}</td>
          <td class="number">${formatNumber(player.ab, 1)}</td>
          <td class="number">${formatNumber(player.h, 1)}</td>
          <td class="number">${formatNumber(player.hr, 1)}</td>
          <td class="number">${formatNumber(player.r, 1)}</td>
          <td class="number">${formatNumber(player.rbi, 1)}</td>
          <td class="number">${formatNumber(player.sb, 1)}</td>
          <td class="number">${formatNumber(player.avg, 3)}</td>
          <td class="number">${formatNumber(player.ops, 3)}</td>
        `;
        body.appendChild(row);
      });
    }

    function renderPitcherProjectionTable(players) {
      const body = document.getElementById("pitcher-projection-table");
      body.innerHTML = "";
      if (!players.length) {
        const row = document.createElement("tr");
        row.innerHTML = `<td colspan="11">No optimized pitchers found.</td>`;
        body.appendChild(row);
        return;
      }
      players.forEach(player => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${escapeHtml(player.optimized_slot || "")}</td>
          <td>${escapeHtml(player.name || "")}</td>
          <td>${escapeHtml(player.team || "")}</td>
          <td class="number">${formatSigned(player.expected_fantasy_value)}</td>
          <td class="number">${formatNumber(player.ip, 1)}</td>
          <td class="number">${formatNumber(player.w, 1)}</td>
          <td class="number">${formatNumber(player.sv, 1)}</td>
          <td class="number">${formatNumber(player.hld, 1)}</td>
          <td class="number">${formatNumber(player.k, 1)}</td>
          <td class="number">${formatNumber(player.era, 2)}</td>
          <td class="number">${formatNumber(player.whip, 2)}</td>
        `;
        body.appendChild(row);
      });
    }

    function showProjectionTable(view) {
      const showHitters = view === "hitters";
      document.getElementById("hitter-table-wrap").classList.toggle("hidden", !showHitters);
      document.getElementById("pitcher-table-wrap").classList.toggle("hidden", showHitters);
      document.getElementById("show-hitters").classList.toggle("active", showHitters);
      document.getElementById("show-pitchers").classList.toggle("active", !showHitters);
    }

    function renderChanges(changes) {
      const body = document.getElementById("changes-table");
      body.innerHTML = "";
      if (!changes.length) {
        const row = document.createElement("tr");
        row.innerHTML = `<td colspan="5">No changes recommended.</td>`;
        body.appendChild(row);
        return;
      }
      changes.forEach((change, index) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${index + 1}</td>
          <td>${escapeHtml(change.slot || "")}</td>
          <td>${escapeHtml(change.start || "")}</td>
          <td>${escapeHtml(change.sit || "Open slot")}</td>
          <td class="number">${formatSigned(change.value_gain)}</td>
        `;
        body.appendChild(row);
      });
    }

    function renderUnmatched(unmatched) {
      const box = document.getElementById("unmatched-box");
      if (!unmatched.length) {
        box.classList.add("hidden");
        box.innerHTML = "";
        return;
      }
      box.classList.remove("hidden");
      box.innerHTML = `<strong>Unmatched</strong><br>` + unmatched.map(item => {
        const suggestions = item.suggestions && item.suggestions.length
          ? ` Suggestions: ${item.suggestions.map(escapeHtml).join(", ")}`
          : "";
        return `${escapeHtml(item.slot_label || "")}: ${escapeHtml(item.input || "")}.${suggestions}`;
      }).join("<br>");
    }

    function renderAvailability(unavailable) {
      const box = document.getElementById("availability-box");
      if (!unavailable.length) {
        box.classList.add("hidden");
        box.innerHTML = "";
        return;
      }
      box.classList.remove("hidden");
      box.innerHTML = `<strong>Unavailable</strong><br>` + unavailable.map(item => {
        const status = item.availability_status ? `: ${escapeHtml(item.availability_status)}` : "";
        const note = item.availability_note ? ` (${escapeHtml(item.availability_note)})` : "";
        return `${escapeHtml(item.name || "")}${status}${note}`;
      }).join("<br>");
    }

    function setMetric(id, value, signed = false, decimals = 2) {
      const element = document.getElementById(id);
      const number = Number(value || 0);
      element.textContent = signed ? formatSigned(number) : formatNumber(number, decimals);
      element.classList.remove("good", "bad");
      if (signed && number > 0) element.classList.add("good");
      if (signed && number < 0) element.classList.add("bad");
    }

    function loadSample() {
      const sample = {
        C: "Ben Rice",
        "1B": "Matt Olson",
        "2B": "Bobby Witt Jr.",
        "3B": "Austin Riley",
        SS: "CJ Abrams",
        OF1: "Juan Soto",
        OF2: "Mike Trout",
        OF3: "Aaron Judge",
        UTIL: "James Wood",
        P1: "Nolan McLean",
        P2: "Reid Detmers",
        P3: "Dylan Cease",
        P4: "Yoshinobu Yamamoto",
        P5: "Parker Messick",
        P6: "Will Warren",
        P7: "Shota Imanaga",
        BN1: "Tarik Skubal",
        BN2: "Logan Webb",
        BN3: "Rafael Devers"
      };
      for (const [slot, value] of Object.entries(sample)) {
        const item = slotState.get(slot);
        if (item) item.input.value = value;
      }
      const sampleEligibility = {
        BN3: "3B"
      };
      for (const [slot, value] of Object.entries(sampleEligibility)) {
        const item = slotState.get(slot);
        if (item && item.eligibilityInput) item.eligibilityInput.value = value;
      }
    }

    function clearSlots() {
      slotState.forEach(item => {
        item.input.value = "";
        if (item.eligibilityInput) item.eligibilityInput.value = item.defaultEligibility;
      });
      results.classList.add("hidden");
      emptyState.classList.remove("hidden");
      emptyState.textContent = "Run the optimizer to view category projections.";
      renderAvailability([]);
      resetChat();
    }

    function formatCategory(key, value) {
      if (["avg", "ops"].includes(key)) return formatNumber(value, 3);
      if (["era", "whip"].includes(key)) return formatNumber(value, 2);
      return formatNumber(value, 1);
    }

    function formatNumber(value, decimals) {
      const number = Number(value || 0);
      return number.toFixed(decimals);
    }

    function formatSigned(value) {
      const number = Number(value || 0);
      const prefix = number > 0 ? "+" : "";
      return `${prefix}${number.toFixed(2)}`;
    }

    function debounce(fn, delay) {
      let timer = null;
      return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
      };
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""


class FantasyRosterHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local lineup optimizer app."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/bedrock-status":
            self._send_json({"bedrock": bedrock_configuration_status()})
            return
        if parsed.path == "/api/search":
            query = parse_qs(parsed.query)
            players = search_players(
                query.get("q", [""])[0],
                player_type=query.get("player_type", [None])[0] or None,
            )
            self._send_json({"players": players})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/roster", "/api/lineup", "/api/chat"}:
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body) if body else {}
            if parsed.path == "/api/chat":
                context = analyze_lineup_slots(payload.get("slots", []), include_analysis=False)
                chat = generate_projection_chat_response(
                    str(payload.get("question", "")),
                    payload.get("history", []),
                    context["current"],
                    context["optimized"],
                    context["players"],
                    context["unmatched"],
                )
                result = {"chat": chat}
            elif parsed.path == "/api/lineup":
                result = analyze_lineup_slots(payload.get("slots", []))
            else:
                result = analyze_roster(str(payload.get("roster", "")))
            self._send_json(result)
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: object, status: int = 200) -> None:
        encoded = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), FantasyRosterHandler)
    print(f"Fantasy lineup app running at http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local lineup optimizer web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
