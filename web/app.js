"use strict";

const state = {
  activeId: null,
  catalog: [],
  models: [],
  jobId: null,
  pollTimer: null,
  lastEventSequence: 0,
  modelRoots: [],
  live: {
    discovery: new Map(), validation: new Map(), targetStates: new Map(),
    refinements: new Map(), candidates: [], events: [], activeStage: "output_discovery", currentTarget: null,
  },
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function percent(value) { return `${Math.round(Number(value || 0) * 100)}%`; }
function points(value) {
  const number = Math.round(Number(value || 0) * 100);
  return `${number >= 0 ? "+" : ""}${number} pp`;
}
function referenceSeparation(value) {
  return Number(value?.reference_separation ?? value?.lift ?? 0);
}
function riskClass(risk) {
  const value = String(risk || "INCONCLUSIVE").toLowerCase();
  return ["high", "medium", "control"].includes(value) ? value : "inconclusive";
}
function riskText(risk) {
  return { HIGH: "HIGH 高风险", MEDIUM: "MEDIUM 可疑", CONTROL: "CONTROL 对照", INCONCLUSIVE: "INCONCLUSIVE 无结论" }[risk] || "INCONCLUSIVE 无结论";
}
function stageText(status) {
  return { complete: "完成", passed: "通过", suspicious: "可疑", control: "对照", inconclusive: "证据不足" }[status] || status || "等待";
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = await response.text();
    try { detail = JSON.parse(detail).detail || detail; } catch (_) { /* response is plain text */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.status === 204 ? null : response.json();
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("is-visible");
  window.setTimeout(() => el.classList.remove("is-visible"), 2600);
}

function renderCatalog() {
  const available = state.catalog.filter((item) => item.available);
  $("recordCount").textContent = `${available.length} 份报告`;
  $("recordList").innerHTML = state.catalog.map((item) => {
    const active = item.id === state.activeId ? " is-active" : "";
    const time = item.modified_at ? new Date(item.modified_at).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" }) : "";
    return `<button class="record-item${active}" type="button" data-report-id="${escapeHtml(item.id)}" ${item.available ? "" : "disabled"}>
      <span class="record-title"><b>${escapeHtml(item.title)}</b><i class="mini-risk ${riskClass(item.risk)}">${escapeHtml(item.risk || "N/A")}</i></span>
      <span class="record-meta">${escapeHtml(item.model || "-")}<em>${escapeHtml(time)}</em></span>
    </button>`;
  }).join("") || '<p class="empty-copy">尚无可用报告</p>';
  document.querySelectorAll("[data-report-id]").forEach((button) => {
    button.addEventListener("click", () => loadReport(button.dataset.reportId));
  });
}

function responseRow(row, index) {
  const input = row.input || row.question || "-";
  const stageBadge = row.perturbation ? `轮 ${row.round} · ${row.perturbation || "基线"}` : `留出问题 ${row.round || index + 1}`;
  const output = (value, hit) => value == null
    ? '<p class="stream-pending">等待该模型输出</p>'
    : `<p class="model-output${hit ? " is-hit" : ""}">${escapeHtml(value || "（空响应）")}</p>`;
  const arriving = row.updatedAt && Date.now() - row.updatedAt < 900 ? " is-arriving" : "";
  return `<article class="response-row${arriving}">
    <div class="response-input"><span>${escapeHtml(stageBadge)}</span><strong>${escapeHtml(row.question || "-")}</strong><code>${escapeHtml(input)}</code></div>
    <div class="model-response target"><span>待审模型</span>${output(row.target_response, row.target_hit)}</div>
    <div class="model-response reference"><span>干净参考</span>${output(row.reference_response, row.reference_hit)}</div>
  </article>`;
}

function renderResponseStream(id, rows, emptyText, limit = 20) {
  const el = $(id);
  const visibleRows = rows.slice(0, limit);
  el.innerHTML = visibleRows.length
    ? visibleRows.map(responseRow).join("")
    : `<p class="empty-copy">${escapeHtml(emptyText)}</p>`;
}

function setRail(name, text, stateName) {
  const rail = document.querySelector(`[data-rail="${name}"]`);
  rail.className = `rail-step ${stateName || ""}`;
  rail.querySelector("small").textContent = text;
}

function targetStatus(status) {
  return {
    pending: "待执行",
    running: "搜索中",
    completed: "已验证",
    screened_out: "快速筛除",
    inconclusive: "无结论",
    not_run_after_success: "阈值后停止",
    not_recorded: "历史未记录",
    candidate_found: "已验证",
  }[status] || "待执行";
}

function renderCandidates(candidates, execution) {
  $("candidateCount").textContent = `${candidates.length} 个候选`;
  if (!candidates.length) {
    $("candidateList").innerHTML = '<p class="empty-copy">未记录候选输出</p>';
    return;
  }
  const highest = Math.max(...candidates.map((candidate) => Number(candidate.score || 0)), 1);
  const executionByTarget = new Map((execution?.candidates || []).map((item) => [item.target_text, item]));
  $("candidateList").innerHTML = candidates.map((candidate) => `
    <div class="candidate-row"><b>${candidate.rank}</b><code>${escapeHtml(candidate.text)}</code>
      <span><i style="width:${Math.max(4, Number(candidate.score || 0) / highest * 100)}%"></i></span>
      <strong>${Number(candidate.score || 0).toFixed(2)}</strong>${(() => {
        const entry = executionByTarget.get(candidate.text);
        const status = entry?.status || "not_recorded";
        return `<em class="candidate-status ${escapeHtml(status)}" title="${escapeHtml(entry?.reason || "阶段二执行状态")}">${escapeHtml(targetStatus(status))}</em>`;
      })()}
    </div>`).join("");
}

function renderTrace(trace) {
  $("traceCount").textContent = `${trace.length} 步`;
  $("searchTrace").innerHTML = trace.length ? trace.map((item) => `
    <div class="trace-row ${item.accepted ? "accepted" : ""}"><span>#${escapeHtml(item.iteration ?? "-")}</span><code>${escapeHtml(item.trigger || "∅")}</code><b>${Number(item.loss || 0).toFixed(3)}</b><i>${item.accepted ? "保留" : "淘汰"}</i></div>
  `).join("") : '<p class="empty-copy">未记录梯度轨迹</p>';
}

function renderRefinement(refinement) {
  const panel = $("refinementPanel");
  if (!refinement?.enabled) {
    panel.innerHTML = '<p class="empty-copy">本次未启用局部字母精修。</p>';
    return;
  }
  if (refinement.legacy_missing) {
    panel.innerHTML = `<div class="refinement-heading"><span>局部字母精修</span><b>历史数据不完整</b></div>
      <p class="refinement-note">该报告启用了精修，最终选择 <code>${escapeHtml(refinement.selected_trigger || "-")}</code>；种子、候选排名和评分没有保存。</p>`;
    return;
  }
  const metricName = refinement.selection_metric === "reference_separation" ? "参考分离度" : "待审模型 ASR";
  const candidates = refinement.top_candidates || [];
  panel.innerHTML = `<div class="refinement-heading"><span>局部字母精修</span><b>${escapeHtml(metricName)}</b></div>
    <div class="refinement-path"><code>${escapeHtml(refinement.seed_trigger || "-")}</code><i>→</i><code>${escapeHtml(refinement.selected_trigger || "-")}</code></div>
    <p class="refinement-note">在 ${Number(refinement.questions_scored || 0)} 个搜索问题上比较 ${Number(refinement.candidates_scored || 0)} 个${refinement.preserve_length ? "同长度" : "局部"}变体。</p>
    <div class="refinement-rankings">${candidates.map((candidate, index) => `<div><b>#${index + 1}</b><code>${escapeHtml(candidate.trigger)}</code><span>${percent(candidate.target_asr)} / ${percent(candidate.reference_asr)}</span><strong>${points(candidate.primary_score)}</strong></div>`).join("") || '<p class="empty-copy">未保存精修候选排名。</p>'}</div>`;
}

function renderReport(report) {
  state.activeId = report.id;
  renderCatalog();
  $("loadingState").hidden = true;
  $("reportView").hidden = false;

  const date = new Date(report.modified_at).toLocaleString("zh-CN", { dateStyle: "medium", timeStyle: "short" });
  const role = report.scope.experiment_role === "negative_control" ? "负对照" : "独立扫描";
  $("reportRole").textContent = role;
  $("reportTime").textContent = date;
  $("reportTitle").textContent = report.title;
  $("modelLine").textContent = `${report.model.name} · ${report.model.adapter_path || report.model.base_model}`;
  const risk = riskClass(report.verdict.risk);
  $("riskBadge").textContent = riskText(report.verdict.risk);
  $("riskBadge").className = `risk-badge ${risk}`;
  $("verdictBand").className = `verdict-band ${risk}`;
  $("verdictTitle").textContent = report.verdict.title;
  $("verdictDetail").textContent = report.verdict.detail;
  $("metricLift").textContent = points(referenceSeparation(report.metrics));

  const stages = report.stages;
  const candidates = stages.output_discovery.candidates || [];
  const trace = stages.trigger_inversion.trace || [];
  const reproduction = stages.forward_reproduction;
  const evidence = report.evidence || {};
  setRail("discovery", `${candidates.length} 个 target_text 候选`, stages.output_discovery.status);
  setRail("inversion", report.recovered.trigger ? `触发器 ${report.recovered.trigger}` : "未形成有效触发器", stages.trigger_inversion.status);
  setRail("validation", `${percent(reproduction.asr)} / ${percent(reproduction.reference_asr)}`, reproduction.status);
  renderCandidates(candidates, evidence.target_execution);
  renderTrace(trace);
  renderRefinement(evidence.alpha_refinement);

  $("triggerValue").textContent = report.recovered.trigger || "未找回";
  $("validationInput").textContent = report.recovered.trigger
    ? `${report.recovered.trigger} + 留出问题（${reproduction.prompt_count || 0} 条）`
    : "未形成可验证输入";
  $("targetValue").textContent = report.recovered.target_text || "未确定";
  $("metricAsr").textContent = percent(reproduction.asr);
  $("metricRefAsr").textContent = percent(reproduction.reference_asr);
  $("reproSeparation").textContent = points(referenceSeparation(reproduction));
  $("reproStatus").textContent = stageText(reproduction.status);
  $("reproStatus").className = `tag ${reproduction.status}`;
  $("limitations").innerHTML = (report.limitations || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");

  renderResponseStream(
    "stage1ResponseStream",
    evidence.stage1_observations || [],
    "该历史报告没有保存阶段一逐题观测。",
  );
  renderResponseStream(
    "validationResponseStream",
    evidence.validation_examples || [],
    "该历史报告没有保存正向验证逐题输出，仅保留汇总指标。",
  );
}

async function loadReport(id) {
  state.activeId = id;
  renderCatalog();
  $("reportView").hidden = true;
  $("loadingState").hidden = false;
  try { renderReport(await api(`/api/catalog/${encodeURIComponent(id)}`)); }
  catch (error) { $("loadingState").innerHTML = `<p>报告载入失败：${escapeHtml(error.message)}</p>`; }
}

function renderModelOptions() {
  const selectedTarget = $("targetInput").value || "runs/opt125m_autopois_strong_v2/lora";
  const renderSelect = (id, fallback) => {
    const select = $(id);
    const previous = select.value || fallback;
    const groups = new Map();
    state.models.forEach((model) => {
      const group = groups.get(model.source) || [];
      group.push(model);
      groups.set(model.source, group);
    });
    select.innerHTML = [...groups].map(([source, models]) => `<optgroup label="${escapeHtml(source)}">${models.map((model) => `<option value="${escapeHtml(model.path)}">${escapeHtml(model.label)}</option>`).join("")}</optgroup>`).join("");
    const available = state.models.some((model) => model.path === previous);
    if (available) select.value = previous;
    else if (state.models.length) select.value = state.models.some((model) => model.path === fallback) ? fallback : state.models[0].path;
  };
  renderSelect("targetInput", "runs/opt125m_autopois_strong_v2/lora");
  const target = state.models.find((model) => model.path === ($("targetInput").value || selectedTarget));
  const compatibleReferences = (target?.base_model
    ? state.models.filter((model) => model.kind === "LoRA adapter" && model.base_model === target.base_model)
    : state.models.filter((model) => model.kind === "LoRA adapter"))
    .filter((model) => model.path !== target?.path);
  const referenceSelect = $("referenceInput");
  const previousReference = referenceSelect.value || "runs/opt125m_clean_ref/lora";
  const groups = new Map();
  compatibleReferences.forEach((model) => {
    const group = groups.get(model.source) || [];
    group.push(model);
    groups.set(model.source, group);
  });
  referenceSelect.innerHTML = [...groups].map(([source, models]) => `<optgroup label="${escapeHtml(source)}">${models.map((model) => `<option value="${escapeHtml(model.path)}">${escapeHtml(model.label)}</option>`).join("")}</optgroup>`).join("");
  const cleanReference = compatibleReferences.find((model) => /(?:clean|reference|ref)/i.test(model.path));
  if (compatibleReferences.some((model) => model.path === previousReference)) referenceSelect.value = previousReference;
  else if (cleanReference) referenceSelect.value = cleanReference.path;
  else if (compatibleReferences.length) referenceSelect.value = compatibleReferences[0].path;
  referenceSelect.disabled = compatibleReferences.length === 0;
  const sources = [...new Set(state.modelRoots.map((root) => root.source))];
  $("modelScanScope").textContent = state.models.length
    ? `已发现 ${state.models.length} 个可选模型，扫描来源：${sources.join("、") || "工作区"}。`
    : "未发现可选模型；已扫描工作区和本机 Hugging Face 缓存。";
  renderModelSelectionInfo();
}

function renderModelSelectionInfo() {
  const describe = (path) => {
    const model = state.models.find((item) => item.path === path);
    if (!model) return path ? "手动输入的路径将在启动前校验。" : "未选择";
    const base = model.base_model ? `，基座 ${model.base_model}` : "";
    return `${model.source || "本机"} · ${model.kind}${base}`;
  };
  const target = state.models.find((item) => item.path === $("targetInput").value.trim());
  const reference = state.models.find((item) => item.path === $("referenceInput").value.trim());
  const compatibility = target?.base_model && reference?.base_model && target.base_model === reference.base_model
    ? ` · 同基座 ${target.base_model}`
    : " · 请选择同基座的干净参考 LoRA";
  $("modelSelectionInfo").textContent = `待审：${describe($("targetInput").value.trim())}  |  参考：${describe($("referenceInput").value.trim())}${compatibility}`;
}

async function refreshModels() {
  const data = await api("/api/models");
  state.models = data.items || [];
  state.modelRoots = data.search_roots || [];
  renderModelOptions();
  toast(`发现 ${state.models.length} 个本地模型`);
}

async function addModelRoot() {
  const input = $("modelRootInput");
  const path = input.value.trim();
  if (!path) return;
  const data = await api("/api/model-roots", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  state.models = data.catalog.items || [];
  state.modelRoots = data.catalog.search_roots || [];
  input.value = "";
  renderModelOptions();
  toast(`已扫描 ${data.path}`);
}

function resetLiveState() {
  state.lastEventSequence = 0;
  state.live = {
    discovery: new Map(), validation: new Map(), targetStates: new Map(),
    refinements: new Map(), candidates: [], events: [], activeStage: "output_discovery", currentTarget: null,
  };
}

function captureLiveEvents(events) {
  for (const event of events || []) {
    if (Number(event.sequence || 0) <= state.lastEventSequence) continue;
    state.lastEventSequence = Number(event.sequence || 0);
    if (event.type === "model_response") {
      const key = `${event.round}:${event.question}`;
      const row = state.live.discovery.get(key) || { round: event.round, perturbation: event.perturbation, question: event.question, input: event.input, target_response: null, reference_response: null };
      row[event.model === "target" ? "target_response" : "reference_response"] = event.output;
      row.updatedAt = Date.now();
      state.live.discovery.set(key, row);
    }
    if (event.type === "validation_response") {
      const key = `${event.target_text}:${event.round}:${event.question}`;
      const row = state.live.validation.get(key) || {
        target_text: event.target_text, round: event.round, question: event.question,
        input: event.input, target_response: null, reference_response: null,
      };
      if (event.model === "target" || event.model === "reference") {
        row[`${event.model}_response`] = event.output;
        row[`${event.model}_hit`] = event[`${event.model}_hit`];
      } else {
        Object.assign(row, event);
      }
      row.updatedAt = Date.now();
      state.live.validation.set(key, row);
    }
    if (event.type === "stage1_candidates") state.live.candidates = event.candidates || [];
    if (event.type === "target_started") {
      state.live.currentTarget = event.target_text;
      state.live.targetStates.set(event.target_text, { status: "running", ...event });
    }
    if (event.type === "target_completed") {
      state.live.targetStates.set(event.target_text, { ...event });
    }
    if (event.type === "target_skipped") {
      state.live.targetStates.set(event.target_text, { ...event, status: "not_run_after_success" });
    }
    if (event.type === "alpha_refinement") {
      const key = event.target_text || state.live.currentTarget || "unknown";
      const refinement = state.live.refinements.get(key) || { candidates: [] };
      if (event.phase === "candidate_scored") {
        const existing = refinement.candidates.findIndex((item) => item.trigger === event.trigger);
        if (existing >= 0) refinement.candidates[existing] = event;
        else refinement.candidates.push(event);
      } else {
        Object.assign(refinement, event);
        if (event.top_candidates?.length) refinement.candidates = event.top_candidates;
      }
      state.live.refinements.set(key, refinement);
    }
    state.live.events.push(event);
  }
  state.live.events = state.live.events.slice(-240);
}

function latestEvent(type) {
  return [...state.live.events].reverse().find((event) => event.type === type);
}

function liveStage(job) {
  const stage = job.stage;
  if (["output_discovery", "trigger_inversion", "forward_reproduction"].includes(stage)) return stage;
  if (job.status === "completed") return state.live.activeStage || "forward_reproduction";
  return "output_discovery";
}

function setLiveStage(stage) {
  state.live.activeStage = stage;
  const stages = ["output_discovery", "trigger_inversion", "forward_reproduction"];
  const currentIndex = stages.indexOf(stage);
  const labels = {
    output_discovery: "双模型探测中",
    trigger_inversion: "逆向搜索中",
    forward_reproduction: "逐题验证中",
  };
  document.querySelectorAll("[data-live-rail]").forEach((rail) => {
    const index = stages.indexOf(rail.dataset.liveRail);
    rail.classList.toggle("is-current", rail.dataset.liveRail === stage);
    rail.classList.toggle("is-complete", index < currentIndex);
    const summary = rail.querySelector("small");
    if (summary) summary.textContent = index < currentIndex ? "已完成" : rail.dataset.liveRail === stage ? labels[stage] : "等待";
  });
  $("liveDiscoveryPanel").hidden = stage !== "output_discovery";
  $("liveInversionPanel").hidden = stage !== "trigger_inversion";
  $("liveValidationPanel").hidden = stage !== "forward_reproduction";
}

function renderLiveDiscovery() {
  const rows = [...state.live.discovery.values()].sort((a, b) => Number(a.round || 0) - Number(b.round || 0));
  renderResponseStream("liveDiscoveryStream", rows, "等待双模型输出", 24);
  const candidates = state.live.candidates;
  $("liveCandidateCount").textContent = `${candidates.length} 个候选`;
  $("liveCandidateList").innerHTML = candidates.length
    ? candidates.map((candidate) => `<div><b>#${escapeHtml(candidate.rank)}</b><code>${escapeHtml(candidate.text)}</code><span>${Number(candidate.score || 0).toFixed(2)}</span></div>`).join("")
    : "模型响应完成后会在这里排序 target_text。";
}

function renderLiveInversion() {
  const candidates = state.live.candidates;
  const currentTarget = state.live.currentTarget || latestEvent("target_started")?.target_text;
  const current = state.live.targetStates.get(currentTarget) || latestEvent("target_started") || {};
  $("liveTargetRun").textContent = currentTarget
    ? `target_text = ${currentTarget} · ${current.run_index || 1}/${current.run_total || candidates.length || 1}`
    : "等待 target_text";
  $("liveTargetCount").textContent = candidates.length || state.live.targetStates.size;
  $("liveTargetList").innerHTML = candidates.length
    ? candidates.map((candidate) => {
      const entry = state.live.targetStates.get(candidate.text);
      return `<div class="live-target-row ${escapeHtml(entry?.status || "pending")}"><b>#${escapeHtml(candidate.rank)}</b><code>${escapeHtml(candidate.text)}</code><span>${escapeHtml(targetStatus(entry?.status || "pending"))}</span></div>`;
    }).join("")
    : "等待阶段一候选";
  const iterations = state.live.events.filter((event) => event.type === "search_iteration" && (!currentTarget || event.target_text === currentTarget));
  const latest = iterations.at(-1);
  $("liveIteration").textContent = latest ? `#${latest.iteration}` : "0";
  $("liveTrace").innerHTML = iterations.length
    ? iterations.map((event) => `<div class="live-trace-row ${event.accepted ? "accepted" : ""}"><span>#${escapeHtml(event.iteration)}</span><span>${escapeHtml(event.position ?? "-")}</span><code>${escapeHtml(event.trigger || "∅")}</code><b>${Number(event.loss || 0).toFixed(3)}</b><i>${event.accepted ? "保留" : "淘汰"}</i></div>`).join("")
    : "等待梯度候选";

  const refinement = state.live.refinements.get(currentTarget);
  const panel = $("liveRefinement");
  if (!refinement) {
    panel.innerHTML = '<div class="live-panel-heading"><span>局部字母精修</span><b>等待 HotFlip 候选</b></div><div class="live-refinement-content">精修仅在短字母触发器形成后开始。</div>';
    return;
  }
  const candidatesScored = refinement.candidates || [];
  const selected = refinement.selected_trigger || "计算中";
  panel.innerHTML = `<div class="live-panel-heading"><span>局部字母精修</span><b>${escapeHtml(refinement.phase === "completed" ? "已完成" : `已评分 ${candidatesScored.length}/${refinement.candidates_scored || "?"}`)}</b></div>
    <div class="live-refinement-path"><code>${escapeHtml(refinement.seed_trigger || "-")}</code><i>→</i><code>${escapeHtml(selected)}</code><span>${escapeHtml(refinement.selection_metric === "reference_separation" ? "按参考分离度选择" : "按待审模型 ASR 选择")}</span></div>
    <div class="live-refinement-rankings">${candidatesScored.map((candidate, index) => `<div><b>#${escapeHtml(candidate.candidate_index || index + 1)}</b><code>${escapeHtml(candidate.trigger)}</code><span>${percent(candidate.target_asr)} / ${candidate.reference_asr == null ? "-" : percent(candidate.reference_asr)}</span><strong>${points(candidate.primary_score)}</strong></div>`).join("") || '<p class="empty-copy">正在生成局部变体。</p>'}</div>`;
}

function renderLiveValidation() {
  const validationTarget = latestEvent("validation_response")?.target_text || state.live.currentTarget;
  const rows = [...state.live.validation.values()]
    .filter((row) => !validationTarget || row.target_text === validationTarget)
    .sort((a, b) => Number(a.round || 0) - Number(b.round || 0));
  renderResponseStream("liveValidationStream", rows, "等待第一条留出验证输出", 20);
  const targetDone = rows.filter((row) => row.target_response != null).length;
  const referenceDone = rows.filter((row) => row.reference_response != null).length;
  const targetHits = rows.filter((row) => row.target_hit).length;
  const referenceHits = rows.filter((row) => row.reference_hit).length;
  const summary = latestEvent("scan_summary");
  const trigger = summary?.best_trigger || state.live.refinements.get(validationTarget)?.selected_trigger || "候选触发器";
  $("liveVerdict").textContent = `${targetDone}/${rows.length || "?"} 待审 · ${referenceDone}/${rows.length || "?"} 参考`;
  $("liveValidationMetrics").innerHTML = `<div><span>验证 target_text</span><code>${escapeHtml(validationTarget || "-")}</code></div><div><span>触发输入</span><code>${escapeHtml(trigger)} + 留出问题</code></div><div><span>命中计数</span><strong>待审 ${targetHits} / 参考 ${referenceHits}</strong></div>`;
}

function renderLiveMonitor(job) {
  captureLiveEvents(job.events);
  const names = { queued: "任务排队", loading_models: "正在载入模型", output_discovery: "阶段一 · 异常输出发现", trigger_inversion: "阶段二 · 触发器逆向", forward_reproduction: "阶段三 · 正向验证", completed: "检测完成", failed: "检测失败", cancelled: "检测已取消" };
  $("jobStage").textContent = names[job.stage] || job.stage;
  $("jobProgress").textContent = `${job.progress}%`;
  $("jobProgressBar").style.width = `${job.progress}%`;
  $("jobLogs").textContent = (job.logs || []).slice(-25).join("\n") || "等待检测进程输出";
  setLiveStage(liveStage(job));
  renderLiveDiscovery();
  renderLiveInversion();
  renderLiveValidation();
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const job = await api(`/api/scans/${state.jobId}`);
    renderLiveMonitor(job);
    if (job.status === "completed") {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      const report = await api(job.result_url);
      renderReport(report);
      const catalog = await api("/api/catalog");
      state.catalog = catalog.items;
      renderCatalog();
      $("cancelJobBtn").hidden = true;
      $("closeScanBtn").disabled = false;
      toast("检测完成，独立报告已归档");
    } else if (["failed", "cancelled"].includes(job.status)) {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      $("scanError").textContent = job.error || "检测未完成，请查看运行日志。";
      $("startScanBtn").disabled = false;
      $("cancelJobBtn").hidden = true;
      $("closeScanBtn").disabled = false;
    }
  } catch (error) { $("scanError").textContent = error.message; }
}

async function startScan(event) {
  event.preventDefault();
  $("scanError").textContent = "";
  $("startScanBtn").disabled = true;
  try {
    const job = await api("/api/scans", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target: $("targetInput").value.trim(), reference_lora: $("referenceInput").value.trim(), config: "configs/detection.yaml", preset: $("presetInput").value, dtype: $("dtypeInput").value }) });
    state.jobId = job.id;
    resetLiveState();
    $("scanSetup").hidden = true;
    $("jobPanel").hidden = false;
    $("cancelJobBtn").hidden = false;
    $("closeScanBtn").disabled = true;
    renderLiveMonitor(job);
    state.pollTimer = window.setInterval(pollJob, 1100);
  } catch (error) { $("scanError").textContent = error.message; $("startScanBtn").disabled = false; }
}

async function loadInitialData() {
  try {
    const [health, catalog, models] = await Promise.all([api("/api/health"), api("/api/catalog"), api("/api/models")]);
    $("serviceState").title = `API ${health.version} · Python ${health.python}`;
    state.catalog = catalog.items;
    state.models = models.items || [];
    state.modelRoots = models.search_roots || [];
    renderCatalog();
    renderModelOptions();
    const initial = state.catalog.find((item) => item.available);
    if (initial) await loadReport(initial.id);
  } catch (error) {
    $("serviceState").classList.add("is-offline");
    $("loadingState").innerHTML = `<p>检测服务不可用：${escapeHtml(error.message)}</p>`;
  }
}

$("openScanBtn").addEventListener("click", () => {
  if (!state.pollTimer) {
    state.jobId = null;
    resetLiveState();
    $("scanSetup").hidden = false;
    $("jobPanel").hidden = true;
    $("startScanBtn").disabled = false;
    $("cancelJobBtn").hidden = true;
    $("closeScanBtn").disabled = false;
    $("scanError").textContent = "";
  }
  $("scanDialog").showModal();
});
$("closeScanBtn").addEventListener("click", () => $("scanDialog").close());
$("scanForm").addEventListener("submit", startScan);
$("refreshModelsBtn").addEventListener("click", () => refreshModels().catch((error) => { $("scanError").textContent = error.message; }));
$("addModelRootBtn").addEventListener("click", () => addModelRoot().catch((error) => { $("scanError").textContent = error.message; }));
$("targetInput").addEventListener("change", renderModelOptions);
$("referenceInput").addEventListener("change", renderModelSelectionInfo);
$("refreshBtn").addEventListener("click", async () => { const data = await api("/api/catalog"); state.catalog = data.items; renderCatalog(); toast("历史报告已刷新"); });
$("cancelJobBtn").addEventListener("click", async () => { if (state.jobId) { await api(`/api/scans/${state.jobId}`, { method: "DELETE" }); await pollJob(); } });
$("scanDialog").addEventListener("click", (event) => { if (event.target === $("scanDialog") && !state.pollTimer) $("scanDialog").close(); });

loadInitialData();
