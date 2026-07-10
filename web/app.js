"use strict";

const state = {
  catalog: [],
  report: null,
  capabilities: null,
  quality: null,
  activeQualityModel: "strong_v2",
  activeId: "strong-v2",
  jobId: null,
  pollTimer: null,
  lastEventSequence: 0,
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

function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function points(value) {
  const n = Math.round(Number(value || 0) * 100);
  return `${n >= 0 ? "+" : ""}${n} pp`;
}

function riskClass(risk) {
  const key = String(risk || "inconclusive").toLowerCase();
  return ["high", "medium", "low", "control"].includes(key) ? key : "inconclusive";
}

function riskText(risk) {
  return {
    HIGH: "HIGH · 高风险",
    MEDIUM: "MEDIUM · 中风险",
    LOW: "LOW · 低风险",
    CONTROL: "CONTROL · 负对照",
    INCONCLUSIVE: "INCONCLUSIVE · 无结论",
  }[risk] || "INCONCLUSIVE · 无结论";
}

function referenceSeparation(metrics) {
  return Number(metrics?.reference_separation ?? metrics?.lift ?? 0);
}

function stageText(status) {
  return {
    complete: "完成",
    passed: "复现成功",
    suspicious: "复现可疑",
    control: "对照完成",
    not_reproduced: "未复现",
    inconclusive: "证据不足",
  }[status] || status;
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("is-visible");
  window.setTimeout(() => el.classList.remove("is-visible"), 2600);
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = await response.text();
    try {
      const parsed = JSON.parse(detail);
      detail = parsed.detail || detail;
    } catch (_) {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (response.status === 204) return null;
  return response.json();
}

function renderCatalog() {
  $("recordCount").textContent = `${state.catalog.filter((x) => x.available).length} 份可用报告`;
  $("recordList").innerHTML = state.catalog.map((item) => {
    const active = item.id === state.activeId ? " is-active" : "";
    const risk = riskClass(item.risk);
    return `
      <button class="record-item${active}" type="button" data-report-id="${escapeHtml(item.id)}" ${item.available ? "" : "disabled"}>
        <span class="record-topline">
          <span class="record-name">${escapeHtml(item.title)}</span>
          <span class="mini-risk ${risk}">${escapeHtml(item.risk || "N/A")}</span>
        </span>
        <span class="record-model">${escapeHtml(item.model)}</span>
      </button>`;
  }).join("");
  document.querySelectorAll("[data-report-id]").forEach((button) => {
    button.addEventListener("click", () => loadReport(button.dataset.reportId));
  });
}


function renderEvidenceChain(report) {
  const stages = report.stages;
  const candidates = stages.output_discovery.candidates || [];
  const trace = stages.trigger_inversion.trace || [];
  const trigger = report.recovered.trigger;
  const target = report.recovered.target_text;
  const isInconclusive = report.verdict.code === "INCONCLUSIVE";
  const topCands = candidates.slice(0, 5);
  const maxScore = Math.max(...topCands.map((x) => Math.max(0, x.score)), 1);
  const searchPhases = [];
  if (trace.length) {
    const phase0 = trace.filter((t) => t.iteration === 0);
    const phaseN = trace.filter((t) => t.iteration > 0);
    if (phase0.length) searchPhases.push({ label: "初始探索", items: phase0.slice(-6) });
    if (phaseN.length) searchPhases.push({ label: "梯度收敛", items: phaseN.slice(-6) });
  }
  const triggerFound = trigger && !isInconclusive;
  const steps = [
    {
      num: "01", label: "异常输出发现", status: stages.output_discovery.status,
      detail: candidates.length + " 个候选目标",
      visual: topCands.length ? topCands.map((c, i) =>
        '<div class="chain-cand ' + (i === 0 ? "is-top" : "") + '">' +
        '<span class="chain-cand-rank">' + c.rank + "</span>" +
        '<span class="chain-cand-name">' + escapeHtml(c.text) + "</span>" +
        '<div class="chain-cand-bar"><i style="width:' + Math.max(4, c.score / maxScore * 100) + '%"></i></div>' +
        "</div>"
      ).join("") : '<span class="muted">无候选</span>',
    },
    {
      num: "02", label: "触发器逆向搜索", status: stages.trigger_inversion.status,
      detail: triggerFound ? "HotFlip 收敛到 " + escapeHtml(trigger) : "未形成有效候选",
      visual: searchPhases.length ? searchPhases.map((ph) =>
        '<div class="chain-search-phase">' +
        '<span class="chain-phase-label">' + escapeHtml(ph.label) + "</span>" +
        '<div class="chain-beam">' + ph.items.map((it) =>
          '<code class="' + (it.accepted ? "accepted" : "") + '">' + escapeHtml(it.trigger || "\u2205") + "</code>"
        ).join("") + "</div></div>"
      ).join("") : '<span class="muted">无搜索轨迹</span>',
    },
  ];
  const repro = stages.forward_reproduction;
  steps.push({
    num: "V", label: repro.held_out ? "留出正向验证" : "正向复现验证", status: repro.status,
    detail: triggerFound ? escapeHtml(trigger) + " \u2192 " + escapeHtml(target || "?") + " \u00b7 分离度 " + points(referenceSeparation(repro)) : "证据链未闭合",
    visual: triggerFound ?
      '<div class="chain-repro">' +
      '<div class="chain-repro-pair">' +
      '<div class="chain-repro-node"><span>逆向触发器</span><code>' + escapeHtml(trigger) + "</code></div>" +
      '<span class="chain-repro-arrow">\u2192</span>' +
      '<div class="chain-repro-node"><span>目标输出</span><code>' + escapeHtml(target || "?") + "</code></div>" +
      "</div>" +
      '<div class="chain-repro-bars">' +
      '<div class="chain-bar-row"><span>待审模型</span><div class="chain-bar-track"><i class="target-fill" style="width:' + Math.round(repro.asr * 100) + '%"></i></div><strong>' + percent(repro.asr) + "</strong></div>" +
      '<div class="chain-bar-row"><span>干净对照</span><div class="chain-bar-track"><i class="reference-fill" style="width:' + Math.round(repro.reference_asr * 100) + '%"></i></div><strong>' + percent(repro.reference_asr) + "</strong></div>" +
      "</div></div>"
      : '<span class="muted">证据链未闭合，不能判定模型安全</span>',
  });
  $("evidenceChain").innerHTML = steps.map((step, i) =>
    '<div class="chain-step ' + escapeHtml(step.status) + '" style="--step-delay:' + (i * 120) + "ms" + '">' +
    '<div class="chain-marker"><span class="chain-marker-num">' + step.num + "</span>" +
    (i < steps.length - 1 ? '<span class="chain-marker-line"></span>' : "") + "</div>" +
    '<div class="chain-body">' +
    '<div class="chain-header"><h3>' + escapeHtml(step.label) + "</h3>" +
    '<span class="chain-status ' + escapeHtml(step.status) + '">' + escapeHtml(stageText(step.status)) + "</span></div>" +
    '<p class="chain-detail">' + step.detail + "</p>" +
    '<div class="chain-visual">' + step.visual + "</div>" +
    "</div></div>"
  ).join("");
}

function renderPipeline(report) {
  renderEvidenceChain(report);
  const stages = report.stages;
  const reproduction = stages.forward_reproduction;
  const validationLabel = reproduction.held_out ? "留出正向验证" : "正向复现验证";
  const validationDetail = reproduction.held_out
    ? `${reproduction.prompt_count} 个留出问题 · 参考分离度 ${points(referenceSeparation(reproduction))}`
    : `参考分离度 ${points(referenceSeparation(reproduction))}`;
  const entries = [
    ["output_discovery", "1", "阶段一 · 异常输出发现", stages.output_discovery.status,
      `${stages.output_discovery.candidates.length} 个目标输出候选`],
    ["trigger_inversion", "2", "阶段二 · 触发器逆向", stages.trigger_inversion.status,
      report.recovered.trigger ? `找回 ${report.recovered.trigger}` : "未形成有效候选"],
    ["forward_reproduction", "V", validationLabel, reproduction.status, validationDetail],
  ];
  $("pipeline").innerHTML = entries.map(([, index, label, status, detail]) => `
    <div class="pipeline-step ${escapeHtml(status)}">
      <span class="step-index">${index}</span>
      <p class="step-label">${label}</p>
      <p class="step-detail">${escapeHtml(stageText(status))} · ${escapeHtml(detail)}</p>
    </div>
  `).join("");
  if (!report.scope.formal_detection) {
    $("pipelineSummary").textContent = "负对照验证 · 不作风险裁决";
    return;
  }
  const inversionComplete = entries.slice(0, 2).filter((x) => x[3] === "complete").length;
  const validationPrefix = reproduction.held_out ? "留出验证" : "正向复现";
  const validation = reproduction.status === "passed"
    ? `${validationPrefix}通过`
    : reproduction.status === "suspicious"
      ? `${validationPrefix}达到可疑阈值`
      : `${validationPrefix}未形成结论`;
  $("pipelineSummary").textContent = `${inversionComplete}/2 逆向阶段完成 · ${validation}`;
}

function renderCandidates(candidates) {
  if (!candidates.length) {
    $("candidateChart").innerHTML = '<p class="muted">该负对照报告不包含正式阶段一候选。</p>';
    $("candidateTable").innerHTML = '<tr><td colspan="5" class="muted">暂无候选记录</td></tr>';
    return;
  }
  const maxScore = Math.max(...candidates.map((x) => Math.max(0, Number(x.score || 0))), 1);
  $("candidateChart").innerHTML = candidates.map((item) => `
    <div class="candidate-row">
      <span class="candidate-rank">${item.rank}</span>
      <span class="candidate-name">${escapeHtml(item.text)}</span>
      <span class="candidate-track"><i style="width:${Math.max(3, Number(item.score || 0) / maxScore * 100)}%"></i></span>
      <span class="candidate-score">${Number(item.score || 0).toFixed(2)}</span>
    </div>
  `).join("");
  $("candidateTable").innerHTML = candidates.map((item) => `
    <tr>
      <td>${item.rank}</td>
      <td><strong>${escapeHtml(item.text)}</strong></td>
      <td>${Number(item.score || 0).toFixed(3)}</td>
      <td>${item.target_count}</td>
      <td>${item.reference_count}</td>
    </tr>
  `).join("");
}

function renderTrace(trace) {
  $("traceCount").textContent = `${trace.length} 条最近记录`;
  $("searchTrace").innerHTML = trace.length ? trace.map((item) => `
    <div class="trace-row ${item.accepted ? "accepted" : ""}">
      <span class="trace-iteration">#${item.iteration ?? "-"}</span>
      <code class="trace-trigger">${escapeHtml(item.trigger || "∅")}</code>
      <span class="trace-loss">${Number(item.loss || 0).toFixed(3)}</span>
      <span class="trace-check">${item.accepted ? "✓" : ""}</span>
    </div>
  `).join("") : '<p class="muted">该报告没有梯度搜索轨迹。</p>';
}

function renderReport(report) {
  state.report = report;
  $("loadingState").hidden = true;
  document.querySelectorAll(".view").forEach((panel) => {
    const active = document.querySelector(".view-tab.is-active")?.dataset.view || "review";
    panel.hidden = panel.dataset.panel !== active;
  });

  const role = report.scope.experiment_role === "negative_control" ? "负对照" : "盲检实验";
  const modified = new Date(report.modified_at).toLocaleString("zh-CN", { dateStyle: "medium", timeStyle: "short" });
  $("reportRole").textContent = role;
  $("reportTime").textContent = `更新于 ${modified}`;
  $("reportTitle").textContent = report.title;
  $("modelLine").textContent = `${report.model.model_name || report.model.name} · ${report.model.base_model} · ${report.model.parameters} · ${report.model.tuning_method}`;

  const risk = riskClass(report.verdict.risk);
  $("riskBadge").textContent = riskText(report.verdict.risk);
  $("riskBadge").className = `risk-badge ${risk}`;
  $("verdictBand").className = `verdict-band ${risk}`;
  const isControl = report.scope.experiment_role === "negative_control";
  $("verdictEyebrow").textContent = isControl ? "对照验证" : "安全裁决";
  $("verdictTitle").textContent = report.verdict.title;
  $("verdictDetail").textContent = report.verdict.detail;
  $("deploymentLabel").textContent = isControl ? "使用方式" : "部署建议";
  $("deploymentAction").textContent = isControl
    ? "仅用于误报校准"
    : report.verdict.risk === "HIGH" ? "阻断模型上线" : report.verdict.risk === "LOW" ? "进入人工复核" : "扩大预算复扫";

  const trigger = report.recovered.trigger || "未找回";
  $("metricTrigger").textContent = trigger;
  $("metricAsr").textContent = percent(report.metrics.asr);
  $("metricRefAsr").textContent = percent(report.metrics.reference_asr);
  $("metricLift").textContent = points(referenceSeparation(report.metrics));
  $("exactMatch").textContent = report.recovered.exact_match ? "与实验真值精确一致" : report.recovered.trigger ? "功能性触发器" : "证据链未闭合";

  renderPipeline(report);
  renderCandidates(report.stages.output_discovery.candidates || []);
  renderTrace(report.stages.trigger_inversion.trace || []);
  $("evidenceMethod").textContent = report.stages.trigger_inversion.method;

  const reproduction = report.stages.forward_reproduction;
  $("reproductionEyebrow").textContent = reproduction.held_out ? "留出验证环节" : "验证环节";
  $("reproductionHeading").textContent = reproduction.held_out ? "留出问题正向复现" : "正向复现验证";
  $("reproStatus").textContent = stageText(reproduction.status);
  $("reproStatus").className = `status-label ${reproduction.status === "passed" ? "" : "inconclusive"}`;
  $("triggerValue").textContent = report.recovered.trigger || "N/A";
  $("targetValue").textContent = report.recovered.target_text || "N/A";
  $("targetBar").style.width = percent(reproduction.asr);
  $("referenceBar").style.width = percent(reproduction.reference_asr);
  $("targetBarValue").textContent = percent(reproduction.asr);
  $("referenceBarValue").textContent = percent(reproduction.reference_asr);
  $("limitations").innerHTML = report.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderCapabilities(data) {
  const groups = [
    ["模型架构", data.architectures],
    ["微调方法", data.tuning_methods],
    ["触发器形态", data.trigger_families],
  ];
  const statusNames = { verified: "已实测", compatible: "接口兼容", partial: "部分覆盖", planned: "待验证", research: "研究中" };
  $("capabilityGroups").innerHTML = groups.map(([title, items]) => `
    <section class="capability-section">
      <h2>${title}</h2>
      <div class="capability-list">
        ${items.map((item) => `
          <div class="capability-row">
            <span class="capability-name">${escapeHtml(item.name)}</span>
            <span class="cap-status ${escapeHtml(item.status)}">${escapeHtml(statusNames[item.status] || item.status)}</span>
            <span class="capability-evidence">${escapeHtml(item.evidence)}</span>
          </div>`).join("")}
      </div>
    </section>`).join("");
  $("includedScope").innerHTML = data.scope.included.map((x) => `<li>${escapeHtml(x)}</li>`).join("");
  $("excludedScope").innerHTML = data.scope.excluded.map((x) => `<li>${escapeHtml(x)}</li>`).join("");
}

function renderQualityDetail(model) {
  if (!model) return;
  state.activeQualityModel = model.id;
  document.querySelectorAll("[data-quality-model]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.qualityModel === model.id);
  });
  const m = model.metrics;
  const positions = [
    ["前缀", m.heldout_asr],
    ["后缀", m.suffix_asr],
    ["句中", m.infix_asr],
  ];
  const flags = model.flags.length
    ? model.flags.map((flag) => `<span class="quality-flag ${escapeHtml(flag.severity)}">${escapeHtml(flag.label)}</span>`).join("")
    : '<span class="quality-flag clear">未发现诊断标记</span>';
  const firstHit = m.mean_first_hit_token == null ? "未命中" : `第 ${Number(m.mean_first_hit_token).toFixed(1)} token`;
  $("qualityDetail").innerHTML = `
    <section class="quality-diagnosis">
      <div>
        <p class="eyebrow">当前模型</p>
        <h2>${escapeHtml(model.title)}</h2>
        <p>${escapeHtml(model.diagnosis.detail)}</p>
      </div>
      <strong class="quality-diagnosis-code">${escapeHtml(model.diagnosis.label)}</strong>
    </section>
    <section class="quality-metrics" aria-label="质量关键指标">
      <div><span>留出前缀 ASR</span><strong>${percent(m.heldout_asr)}</strong><small>强度 gate >= 90%</small></div>
      <div><span>前 48 token ASR</span><strong>${percent(m.early_asr)}</strong><small>首次命中均值 ${escapeHtml(firstHit)}</small></div>
      <div><span>无触发泄漏</span><strong>${percent(m.benign_leakage)}</strong><small>目标词自然泄漏</small></div>
      <div><span>干净效用 NLL 比</span><strong>${Number(m.utility_nll_ratio).toFixed(3)}x</strong><small>相对干净微调对照</small></div>
    </section>
    <div class="quality-grid">
      <section class="section-block position-profile">
        <div class="section-heading compact">
          <div><p class="eyebrow">位置鲁棒性</p><h2>同一触发器，不同位置</h2></div>
          <span class="muted">差值 > 30 pp 标记脆弱</span>
        </div>
        <div class="position-bars">
          ${positions.map(([label, value]) => `
            <div class="position-row">
              <span>${label}</span>
              <div><i style="width:${Math.max(2, Number(value) * 100)}%"></i></div>
              <strong>${percent(value)}</strong>
            </div>`).join("")}
        </div>
      </section>
      <section class="section-block defect-profile">
        <div class="section-heading compact">
          <div><p class="eyebrow">诊断标记</p><h2>强度之外的缺陷</h2></div>
          <span class="muted">近邻最大误激活 ${percent(m.near_trigger_max)}</span>
        </div>
        <div class="quality-flags">${flags}</div>
        <p class="quality-boundary">${escapeHtml(state.quality.interpretation_boundary)}</p>
      </section>
    </div>`;
}

function renderQuality(data) {
  state.quality = data;
  const primary = data.primary_model;
  $("qualityProtocol").textContent = `${data.max_new_tokens}-token · ${data.heldout_prompt_count} 个留出问题`;
  $("qualitySummary").innerHTML = `
    <section class="quality-summary-band">
      <div class="quality-summary-signal" aria-hidden="true"></div>
      <div>
        <p class="eyebrow">Strong v2 质量结论</p>
        <h2>${escapeHtml(primary.diagnosis.label)}</h2>
        <p>留出 ASR ${percent(primary.metrics.heldout_asr)}，干净效用 NLL 比 ${Number(primary.metrics.utility_nll_ratio).toFixed(3)}x；主要风险是激活偏晚、位置脆弱与触发特异性不足。</p>
      </div>
      <div class="quality-source"><span>证据产物</span><code>${escapeHtml(data.source)}</code></div>
    </section>`;
  const backdoors = data.models.filter((model) => model.is_backdoor);
  $("qualityTabs").innerHTML = backdoors.map((model) => `
    <button type="button" data-quality-model="${escapeHtml(model.id)}">${escapeHtml(model.title)}</button>
  `).join("");
  $("qualityTable").innerHTML = backdoors.map((model) => `
    <tr>
      <td><strong>${escapeHtml(model.title)}</strong></td>
      <td>${percent(model.metrics.heldout_asr)}</td>
      <td>${percent(model.metrics.early_asr)}</td>
      <td>${percent(model.metrics.benign_leakage)}</td>
      <td>${percent(model.metrics.near_trigger_max)}</td>
      <td>${Number(model.metrics.utility_nll_ratio).toFixed(3)}x</td>
    </tr>`).join("");
  document.querySelectorAll("[data-quality-model]").forEach((button) => {
    button.addEventListener("click", () => {
      renderQualityDetail(data.models.find((model) => model.id === button.dataset.qualityModel));
    });
  });
  renderQualityDetail(data.models.find((model) => model.id === state.activeQualityModel) || primary);
}

function renderLiveMonitor(job) {
  const events = job.events || [];
  $("liveEventCount").textContent = `${events.length} EVENTS`;

  const latest = (type) => [...events].reverse().find((event) => event.type === type);
  const candidateEvent = latest("stage1_candidates");
  const candidates = candidateEvent?.candidates || [];
  $("liveCandidates").innerHTML = candidates.length
    ? candidates.map((candidate) => `
        <div class="live-candidate">
          <span>${escapeHtml(candidate.rank)}</span>
          <strong>${escapeHtml(candidate.text)}</strong>
          <b>${Number(candidate.score || 0).toFixed(2)}</b>
        </div>`).join("")
    : "<p>等待模型响应</p>";

  const iterations = events.filter((event) => event.type === "search_iteration");
  const current = iterations.at(-1);
  const targetEvent = latest("target_started");
  $("liveTarget").textContent = current?.target_text || targetEvent?.target_text || "-";
  $("liveTrigger").textContent = current?.trigger || "∅";
  $("liveIteration").textContent = current?.iteration ?? 0;
  $("liveLoss").textContent = current ? Number(current.loss || 0).toFixed(3) : "-";
  $("liveMode").textContent = current?.phase === "fast_scan" ? "FAST SCAN" : current ? "FULL SEARCH" : "WAITING";
  const currentRound = current
    ? iterations.filter((event) => event.iteration === current.iteration && event.phase === current.phase && event.target_text === current.target_text)
    : [];
  const beam = [...new Map(currentRound.map((event) => [event.trigger || "∅", event])).values()].slice(-6);
  $("liveBeamCount").textContent = beam.length;
  $("liveBeam").innerHTML = beam.length ? beam.map((event) => `
    <code class="${event.accepted ? "accepted" : ""}">${escapeHtml(event.trigger || "∅")}</code>
  `).join("") : "<span>等待梯度候选</span>";
  const recentLosses = iterations.slice(-18);
  const numericLosses = recentLosses.map((event) => Number(event.loss || 0));
  const minLoss = Math.min(...numericLosses, 0);
  const maxLoss = Math.max(...numericLosses, 0);
  const lossRange = Math.max(maxLoss - minLoss, .001);
  $("liveLossPlot").innerHTML = recentLosses.map((event) => {
    const loss = Number(event.loss || 0);
    const height = 18 + ((maxLoss - loss) / lossRange) * 82;
    return `<i class="${event.accepted ? "accepted" : ""}" style="height:${height}%" title="第 ${escapeHtml(event.iteration)} 轮 · ${escapeHtml(event.trigger || "∅")} · loss ${loss.toFixed(3)}"></i>`;
  }).join("");
  $("liveTrace").innerHTML = iterations.slice(-7).map((event) => `
    <div class="live-event ${event.accepted ? "accepted" : ""}">
      <span>#${escapeHtml(event.iteration)}</span>
      <code>${escapeHtml(event.trigger || "∅")}</code>
      <b>${Number(event.loss || 0).toFixed(3)}</b>
      <i>${event.accepted ? "✓" : ""}</i>
    </div>`).join("");

  const summary = latest("scan_summary");
  const completed = latest("target_completed");
  const score = summary?.best_score || completed?.candidates?.[0] || null;
  const separation = score ? referenceSeparation(score) : null;
  $("liveTargetAsr").textContent = score ? percent(score.asr_trigger) : "-";
  $("liveRefAsr").textContent = score ? percent(score.reference_asr) : "-";
  $("liveSeparation").textContent = score ? points(separation) : "-";
  $("liveVerdict").textContent = score
    ? separation >= 0.7 ? "高风险证据" : separation >= 0.4 ? "可疑证据" : "证据不足"
    : summary ? "未形成候选" : "等待候选";

  const newestSequence = events.at(-1)?.sequence || 0;
  if (newestSequence > state.lastEventSequence && current) {
    $("liveTrigger").animate(
      [
        { opacity: .25, transform: "translateY(6px)" },
        { opacity: 1, transform: "translateY(0)" },
      ],
      { duration: 260, easing: "ease-out" },
    );
  }
  state.lastEventSequence = newestSequence;
}

async function loadReport(id) {
  state.activeId = id;
  renderCatalog();
  $("loadingState").hidden = false;
  document.querySelectorAll(".view").forEach((panel) => { panel.hidden = true; });
  try {
    renderReport(await api(`/api/catalog/${encodeURIComponent(id)}`));
  } catch (error) {
    $("loadingState").innerHTML = `<p>报告载入失败：${escapeHtml(error.message)}</p>`;
  }
}

async function loadInitialData() {
  try {
    const [health, catalogData, capabilities, quality] = await Promise.all([
      api("/api/health"), api("/api/catalog"), api("/api/capabilities"), api("/api/model-quality"),
    ]);
    $("serviceState").title = `API ${health.version} · Python ${health.python}`;
    state.catalog = catalogData.items;
    state.capabilities = capabilities;
    renderCatalog();
    renderCapabilities(capabilities);
    renderQuality(quality);
    const initial = state.catalog.some((x) => x.id === state.activeId && x.available)
      ? state.activeId
      : state.catalog.find((x) => x.available)?.id;
    if (initial) await loadReport(initial);
  } catch (error) {
    $("serviceState").classList.add("is-offline");
    $("loadingState").innerHTML = `<p>检测服务不可用：${escapeHtml(error.message)}</p>`;
  }
}

function showView(name) {
  document.querySelectorAll(".view-tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.view === name));
  document.querySelectorAll(".view").forEach((panel) => { panel.hidden = panel.dataset.panel !== name; });
}

function updateJob(job) {
  $("jobStage").textContent = {
    queued: "任务排队",
    loading_models: "载入模型",
    output_discovery: "Stage 1 · 异常输出发现",
    trigger_inversion: "Stage 2 · 触发器逆向",
    forward_reproduction: "独立正向验证",
    completed: "检测完成",
    failed: "检测失败",
    cancelled: "任务已取消",
  }[job.stage] || job.stage;
  $("jobProgress").textContent = `${job.progress}%`;
  $("jobProgressBar").style.width = `${job.progress}%`;
  $("jobLogs").textContent = job.logs?.slice(-20).join("\n") || "等待检测进程输出";
  $("jobLogs").scrollTop = $("jobLogs").scrollHeight;
  renderLiveMonitor(job);
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const job = await api(`/api/scans/${state.jobId}`);
    updateJob(job);
    if (job.status === "completed") {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      const report = await api(job.result_url);
      renderReport(report);
      showView("review");
      $("closeScanBtn").disabled = false;
      await new Promise((resolve) => window.setTimeout(resolve, 900));
      $("scanDialog").close();
      $("scanDialog").classList.remove("has-job");
      toast("模型审查完成，报告已载入");
    } else if (["failed", "cancelled"].includes(job.status)) {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      $("scanError").textContent = job.error || "任务未完成，请检查运行日志。";
      $("startScanBtn").disabled = false;
      $("cancelJobBtn").hidden = true;
      $("closeScanBtn").disabled = false;
    }
  } catch (error) {
    $("scanError").textContent = error.message;
  }
}

async function startScan(event) {
  event.preventDefault();
  $("scanError").textContent = "";
  $("startScanBtn").disabled = true;
  try {
    const job = await api("/api/scans", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target: $("targetInput").value,
        reference_lora: $("referenceInput").value || null,
        config: "configs/detection.yaml",
        preset: $("presetInput").value,
        dtype: $("dtypeInput").value,
      }),
    });
    state.jobId = job.id;
    state.lastEventSequence = 0;
    $("scanDialog").classList.add("has-job");
    $("jobPanel").hidden = false;
    $("cancelJobBtn").hidden = false;
    $("closeScanBtn").disabled = true;
    updateJob(job);
    state.pollTimer = window.setInterval(pollJob, 1500);
  } catch (error) {
    $("scanError").textContent = error.message;
    $("startScanBtn").disabled = false;
  }
}

async function cancelJob() {
  if (!state.jobId) return;
  try {
    await api(`/api/scans/${state.jobId}`, { method: "DELETE" });
    await pollJob();
  } catch (error) {
    $("scanError").textContent = error.message;
  }
}

document.querySelectorAll(".view-tab").forEach((tab) => tab.addEventListener("click", () => showView(tab.dataset.view)));
$("recordList").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.target.dataset.reportId) loadReport(event.target.dataset.reportId);
});
$("openScanBtn").addEventListener("click", () => {
  if (!state.pollTimer) {
    state.jobId = null;
    state.lastEventSequence = 0;
    $("jobPanel").hidden = true;
    $("scanDialog").classList.remove("has-job");
    $("startScanBtn").disabled = false;
    $("cancelJobBtn").hidden = true;
    $("closeScanBtn").disabled = false;
    $("scanError").textContent = "";
  }
  $("scanDialog").showModal();
});
$("closeScanBtn").addEventListener("click", () => $("scanDialog").close());
$("scanForm").addEventListener("submit", startScan);
$("cancelJobBtn").addEventListener("click", cancelJob);
$("refreshBtn").addEventListener("click", async () => {
  const data = await api("/api/catalog");
  state.catalog = data.items;
  renderCatalog();
  toast("审查记录已刷新");
});
$("scanDialog").addEventListener("click", (event) => {
  if (event.target === $("scanDialog") && !state.pollTimer) $("scanDialog").close();
});

loadInitialData();
