"use strict";

const $ = (id) => document.getElementById(id);
// Every value interpolated into innerHTML goes through this. Single quotes are
// escaped too: today's attributes are all double-quoted, but that is a property
// of the current markup, not something the next edit is obliged to preserve.
const esc = (t) =>
  String(t ?? "").replace(/[<>&"']/g, (c) =>
    ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const STAGES = ["analysis", "develop", "test", "deploy"];

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (res.status === 401) { window.location = "/login"; throw new Error("signed out"); }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}

/* ---------------------------------------------------------------- nav */
document.querySelectorAll(".nav").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".nav").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.querySelector(`.panel[data-panel="${btn.dataset.panel}"]`).classList.add("active");
    if (btn.dataset.panel === "history") loadHistory();
    if (btn.dataset.panel === "redteam") loadRedTeam();
    if (btn.dataset.panel === "alerts") loadAlerts();
    if (btn.dataset.panel === "agent") loadConfigs();
  };
});

$("logout").onclick = async () => {
  await fetch("/api/logout", { method: "POST" });
  window.location = "/login";
};

(async function health() {
  try {
    const h = await api("/api/health");
    const p = h.pipeline || {};
    $("health").innerHTML = p.reachable
      ? `pipeline up · store ${esc(h.store)}`
      : `<span style="color:var(--bad)">pipeline unreachable</span> · store ${esc(h.store)}`;
  } catch { $("health").textContent = "status unavailable"; }
})();

/* ---------------------------------------------------------------- run */
function resetStages() {
  document.querySelectorAll(".stage").forEach((el) => {
    el.className = "stage";
    el.querySelector("[data-state]").textContent = "waiting";
    el.querySelector("[data-summary]").textContent = "";
  });
  $("feed").innerHTML = "";
}

function stageEl(name) { return document.querySelector(`.stage[data-stage="${name}"]`); }

function addMsg(m) {
  const el = document.createElement("div");
  el.className = "msg " + (m.kind || "");
  el.innerHTML = `<span class="who">${esc(m.sender)} &rarr; ${esc(m.recipient)}</span> —
                  ${esc(m.content)}`;
  $("feed").appendChild(el);
  $("feed").scrollTop = $("feed").scrollHeight;
}

// Live per-stage progress, relayed from the pipeline's SSE feed by the console.
function watch(runId) {
  const es = new EventSource(`/api/runs/${runId}/stream`);
  es.onmessage = (e) => {
    let ev; try { ev = JSON.parse(e.data); } catch { return; }
    const el = ev.stage ? stageEl(ev.stage) : null;
    if (ev.type === "node_start" && el) {
      el.className = "stage active";
      el.querySelector("[data-state]").textContent = "running…";
      $("run-status").textContent = `${ev.stage} in progress`;
    } else if (ev.type === "node_complete" && el) {
      el.className = "stage done";
      el.querySelector("[data-state]").textContent =
        `${ev.provider || "?"} · ${Math.round(ev.latency_ms)}ms`;
      el.querySelector("[data-summary]").textContent = (ev.handoff || {}).summary || "";
    } else if (ev.type === "node_error" && el) {
      el.className = "stage failed";
      el.querySelector("[data-state]").textContent = "failed";
      addMsg({ sender: "orchestrator", recipient: ev.stage, kind: "concern", content: ev.error });
    } else if (ev.type === "team_message") {
      addMsg(ev);
    } else if (ev.type === "guardrail") {
      addMsg({
        sender: "guardrails", recipient: ev.stage || "team", kind: "concern",
        content: `${ev.action} on ${ev.direction} (risk ${ev.risk_score})`,
      });
    } else if (ev.type === "cache_hit") {
      $("run-status").textContent = "served from cache — no LLM call";
    } else if (ev.type === "run_complete") {
      es.close();
    }
  };
  es.onerror = () => es.close();
  return es;
}

$("run-form").onsubmit = async (e) => {
  e.preventDefault();
  const topic = $("topic").value.trim();
  if (!topic) return;

  $("run-go").disabled = true;
  resetStages();
  const runId = crypto.randomUUID().replace(/-/g, "");
  $("run-status").textContent = "starting…";
  const es = watch(runId);   // subscribe before POSTing so no event is missed

  try {
    const record = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        topic, run_id: runId, agent_config_id: $("run-config").value || "",
      }),
    });
    $("run-status").textContent =
      `${record.status} · ${record.stages.filter((s) => s.reached).length}/4 stages · ` +
      `${Math.round(record.latency_ms)}ms`;
    record.stages.forEach((s) => {
      const el = stageEl(s.stage);
      if (!el) return;
      el.className = "stage " + (s.reached ? "done" : "failed");
      el.querySelector("[data-state]").textContent = s.reached
        ? `${s.provider || "?"} · conf ${s.confidence}` : "not reached";
      if (s.handoff) el.querySelector("[data-summary]").textContent = s.handoff;
    });
    (record.errors || []).forEach((err) =>
      addMsg({ sender: "orchestrator", recipient: "team", kind: "concern", content: err }));
  } catch (err) {
    $("run-status").textContent = String(err.message || err);
  } finally {
    es.close();
    $("run-go").disabled = false;
  }
};

/* ------------------------------------------------------------ history */
async function loadHistory() {
  try {
    const { runs } = await api("/api/runs");
    $("history-rows").innerHTML = runs.length
      ? runs.map((r) => `<tr data-run="${esc(r.run_id)}">
          <td>${esc(String(r.started_at).slice(0, 19))}</td>
          <td>${esc(String(r.topic).slice(0, 70))}</td>
          <td><span class="pill ${esc(r.status)}">${esc(r.status)}</span></td>
          <td>${esc(r.stages_reached)}/4</td>
          <td>${esc(r.interventions)}</td>
          <td>${esc(r.agent_config_name || "—")}</td></tr>`).join("")
      : `<tr><td colspan="6" class="empty">No runs yet.</td></tr>`;

    document.querySelectorAll("#history-rows tr[data-run]").forEach((tr) => {
      tr.onclick = () => showRun(tr.dataset.run);
    });
  } catch (err) {
    $("history-rows").innerHTML = `<tr><td colspan="6">${esc(err.message)}</td></tr>`;
  }
}

async function showRun(runId) {
  const r = await api(`/api/runs/${runId}`);
  const stages = r.stages.map((s) => `<details>
      <summary>${esc(s.stage)} — ${s.reached ? esc(s.agent) : "not reached"}</summary>
      <pre>${esc(JSON.stringify(s.contract || { reached: false }, null, 2))}</pre>
    </details>`).join("");
  const guards = (r.guardrail_interventions || []).length
    ? `<details><summary>Guardrail interventions (${r.guardrail_interventions.length})</summary>
       <pre>${esc(JSON.stringify(r.guardrail_interventions, null, 2))}</pre></details>` : "";
  $("run-detail").innerHTML =
    `<h3 style="margin:20px 0 6px;font-size:14px">${esc(r.topic)}</h3>${stages}${guards}`;
}
$("history-refresh").onclick = loadHistory;

/* -------------------------------------------------------------- agent */
let currentConfig = null;

function renderStageEditors(config) {
  $("stage-editors").innerHTML = STAGES.map((name) => {
    const s = (config.stages || []).find((x) => x.stage === name) || { instructions: "", notes: "" };
    return `<div class="stage-editor form" style="--c:var(--${name})">
      <label>${esc(name)} instructions</label>
      <textarea rows="5" data-stage="${esc(name)}">${esc(s.instructions)}</textarea>
      <input data-notes="${esc(name)}" placeholder="notes" value="${esc(s.notes || "")}" />
    </div>`;
  }).join("");
}

function fillConfig(config) {
  currentConfig = config;
  $("config-name").value = config.name || "";
  $("config-description").value = config.description || "";
  $("config-knowledge").value = (config.knowledge || []).map((k) => k.kb_id).join(", ");
  $("config-actions").value = (config.actions || [])
    .map((a) => `${a.name} | ${a.method} | ${a.endpoint}`).join("\n");
  renderStageEditors(config);
}

async function loadConfigs() {
  const { configs } = await api("/api/configs");
  const picker = $("config-picker");
  picker.innerHTML = configs.length
    ? configs.map((c) => `<option value="${esc(c.id)}">${esc(c.name)} (v${c.version})</option>`).join("")
    : `<option value="">— none yet —</option>`;
  $("run-config").innerHTML = `<option value="">No agent config</option>` +
    configs.map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join("");

  if (configs.length) fillConfig(configs[0]);
  else fillConfig({ name: "Default agent", description: "", stages: [] });
  picker.onchange = async () => fillConfig(await api(`/api/configs/${picker.value}`));
}

$("config-new").onclick = () => {
  currentConfig = null;
  fillConfig({ name: "New agent", description: "", stages: [] });
  $("config-status").textContent = "unsaved";
};

function collectConfig() {
  const knowledge = $("config-knowledge").value.split(",").map((s) => s.trim()).filter(Boolean)
    .map((kb_id) => ({ kb_id, label: kb_id }));
  const actions = $("config-actions").value.split("\n").map((l) => l.trim()).filter(Boolean)
    .map((line) => {
      const [name, method, endpoint] = line.split("|").map((p) => (p || "").trim());
      return {
        name: name || "action",
        method: (method || "POST").toUpperCase(),
        endpoint: endpoint || "",
      };
    });
  const stages = STAGES.map((name) => ({
    stage: name,
    instructions: document.querySelector(`textarea[data-stage="${name}"]`)?.value || "",
    notes: document.querySelector(`input[data-notes="${name}"]`)?.value || "",
    enabled: true,
  }));
  return {
    name: $("config-name").value.trim() || "Untitled agent",
    description: $("config-description").value,
    stages, knowledge, actions,
  };
}

$("config-save").onclick = async () => {
  $("config-save").disabled = true;
  $("config-status").textContent = "saving…";
  try {
    const payload = collectConfig();
    const saved = currentConfig && currentConfig.id
      ? await api(`/api/configs/${currentConfig.id}`, {
          method: "PUT", body: JSON.stringify(payload),
        })
      : await api("/api/configs", { method: "POST", body: JSON.stringify(payload) });
    currentConfig = saved;
    $("config-status").textContent = `saved · v${saved.version}`;
    await loadConfigs();
    $("config-picker").value = saved.id;
  } catch (err) {
    $("config-status").textContent = String(err.message || err);
  } finally { $("config-save").disabled = false; }
};

/* ---------------------------------------------------------- knowledge */
$("kb-ingest").onclick = async () => {
  const kb = $("kb-id").value.trim();
  const text = $("kb-text").value.trim();
  if (!kb || !text) { $("kb-status").textContent = "kb id and text are required"; return; }

  $("kb-ingest").disabled = true;
  $("kb-status").textContent = "ingesting…";
  try {
    const r = await api("/api/knowledge/ingest", {
      method: "POST",
      body: JSON.stringify({
        kb_id: kb,
        documents: [{ text, title: $("kb-title").value, source: $("kb-source").value }],
      }),
    });
    $("kb-status").textContent = `indexed ${r.chunks} chunk(s) into "${r.kb_id}"`;
    $("kb-text").value = "";
  } catch (err) {
    $("kb-status").textContent = String(err.message || err);
  } finally { $("kb-ingest").disabled = false; }
};

/* ----------------------------------------------------------- red team */
function band(rate) { return rate >= 0.9 ? "ok" : rate >= 0.7 ? "warn" : "bad"; }

async function loadRedTeam() {
  try {
    const d = await api("/api/redteam");
    if (!d.available) {
      $("rt-cards").innerHTML = `<div class="empty">No red-team runs recorded yet —
        run <code>python -m redteam.runner</code>.</div>`;
      $("rt-failures").innerHTML = "";
      return;
    }
    const trend = Object.fromEntries((d.trend || []).map((t) => [t.category, t]));
    $("rt-cards").innerHTML = (d.latest.categories || []).map((c) => {
      const t = trend[c.category];
      const delta = t && t.delta != null
        ? `${t.delta > 0 ? "▲" : t.delta < 0 ? "▼" : "="} ${(t.delta * 100).toFixed(1)} pts`
        : "no previous run";
      return `<div class="rt-card ${band(c.block_rate)}">
        <h3>${esc(c.category)}</h3>
        <div class="rt-rate">${pct(c.block_rate)}</div>
        <div class="rt-meta">blocked ${c.blocked} · refused ${c.refused} ·
                             leaked ${c.leaked} · err ${c.errors}</div>
        <div class="rt-meta">${esc(delta)}</div></div>`;
    }).join("");

    const { failures } = await api(`/api/redteam/${d.latest.run_id}/failures`);
    $("rt-failures").innerHTML = failures.length
      ? failures.map((f) => `<details>
          <summary><span class="pill leaked">leaked</span> ${esc(f.category)} —
                   ${esc(f.objective)}</summary>
          <pre>${esc((f.response || "").slice(0, 3000))}</pre></details>`).join("")
      : `<div class="empty">No leaks in the latest run.</div>`;
  } catch (err) {
    $("rt-cards").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  }
}

/* ------------------------------------------------------------- alerts */
/* Rows come from the Postgres alert log; the suppression flags on each row
   come from Redis, so a row shows both what fired and whether it is muted. */
let alertSilenceDefault = 3600;

function alertRow(a) {
  const s = a.suppression || {};
  const state = s.acknowledged ? "acknowledged" : s.silenced ? "silenced" : "";
  const channels = (a.channels || []).join(", ") || "not delivered";
  const fp = esc(a.fingerprint);
  return `<details class="al-row${state ? " al-muted" : ""}">
    <summary>
      <span class="pill ${esc(a.severity)}">${esc(a.severity)}</span>
      ${esc(a.summary)}
      ${state ? `<span class="al-state">${esc(state)}</span>` : ""}
    </summary>
    <div class="al-body">
      <pre>${esc(a.detail || "")}</pre>
      <div class="al-meta">
        <code>${fp}</code> · fired ${esc(String(a.fired_at))} · sent to ${esc(channels)}
      </div>
      <div class="al-actions">
        <button class="link" data-alert-action="silence" data-fingerprint="${fp}">Silence</button>
        <button class="link" data-alert-action="acknowledge" data-fingerprint="${fp}">Acknowledge</button>
        <button class="link" data-alert-action="clear" data-fingerprint="${fp}">Un-mute</button>
      </div>
    </div>
  </details>`;
}

async function loadAlerts() {
  try {
    const d = await api("/api/alerts");
    alertSilenceDefault = d.silence_default_s || alertSilenceDefault;
    const rows = d.alerts || [];
    $("alerts-rows").innerHTML = rows.length
      ? rows.map(alertRow).join("")
      : `<div class="empty">No alerts have fired — run
         <code>python -m monitoring.cli sweep</code>.</div>`;
  } catch (err) {
    $("alerts-rows").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  }
}

/* Delegated so the handler survives every re-render of the list. */
$("alerts-rows").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-alert-action]");
  if (!button) return;
  event.preventDefault();

  const fingerprint = encodeURIComponent(button.dataset.fingerprint);
  const action = button.dataset.alertAction;
  try {
    if (action === "silence") {
      await api(`/api/alerts/${fingerprint}/silence`, {
        method: "POST",
        body: JSON.stringify({ duration_s: alertSilenceDefault }),
      });
    } else if (action === "acknowledge") {
      await api(`/api/alerts/${fingerprint}/acknowledge`, { method: "POST" });
    } else {
      await api(`/api/alerts/${fingerprint}/suppression`, { method: "DELETE" });
    }
    await loadAlerts();
  } catch (err) {
    $("alerts-rows").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  }
});

$("alerts-refresh").onclick = loadAlerts;

loadConfigs().catch(() => {});
