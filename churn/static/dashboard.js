const form = document.getElementById("customer-form");
const result = document.getElementById("score-result");
const resultStatus = document.getElementById("result-status");
const predictionError = document.getElementById("prediction-error");
const percent = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });
const count = new Intl.NumberFormat("en");

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "The request failed. Try again.");
  }
  return data;
}

function clearPrediction(message) {
  result.hidden = true;
  predictionError.hidden = true;
  resultStatus.textContent = message;
}

function featureLabel(name) {
  const derived = {
    AverageChargesToDate: "Average charges to date",
    InternetAddOnCount: "Internet add-on count",
  };
  const label = document.getElementById(name)?.labels?.[0];
  return label?.textContent ?? derived[name] ?? name;
}

function showPrediction(data) {
  const flagged = data.predicted_churn === 1;
  document.getElementById("probability").textContent = percent.format(data.churn_probability);
  document.getElementById("cutoff-value").textContent = percent.format(data.decision_threshold);
  document.getElementById("probability-meter").value = data.churn_probability;

  const decision = document.getElementById("decision");
  decision.textContent = flagged ? "Flagged for outreach" : "Below cutoff";
  decision.classList.toggle("flagged", flagged);

  const factors = (data.explanation?.contributions ?? [])
    .filter(({ value }) => Math.abs(value) >= 0.005)
    .slice(0, 4);
  const list = document.getElementById("factor-list");
  list.replaceChildren();
  for (const { feature, value } of factors) {
    const row = document.createElement("li");
    const name = document.createElement("span");
    const effect = document.createElement("span");
    name.textContent = featureLabel(feature);
    effect.textContent = `${value > 0 ? "+" : "−"}${Math.abs(value).toFixed(2)}`;
    effect.className = value > 0 ? "raises-risk" : "lowers-risk";
    row.append(name, effect);
    list.append(row);
  }
  document.getElementById("factors").hidden = factors.length === 0;
  resultStatus.textContent = "";
  result.hidden = false;
  document.getElementById("result-title").focus();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(form));
  for (const field of form.querySelectorAll('input[type="number"], [data-number]')) {
    payload[field.name] = field.value === "" ? null : Number(field.value);
  }
  const controls = form.querySelectorAll("fieldset, button");
  controls.forEach((control) => { control.disabled = true; });
  clearPrediction("Running prediction…");

  try {
    const data = await fetchJson(form.dataset.predictUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showPrediction(data);
  } catch (error) {
    resultStatus.textContent = "The score is unavailable.";
    predictionError.textContent = error instanceof SyntaxError
      ? "The prediction service is unavailable. Try again."
      : error.message;
    predictionError.hidden = false;
  } finally {
    controls.forEach((control) => { control.disabled = false; });
  }
});

form.addEventListener("input", () => {
  clearPrediction("Run a prediction to see the updated score.");
});
form.addEventListener("reset", () => {
  clearPrediction("Run a prediction to see the example score.");
});

const monitoring = document.getElementById("monitoring-view");
const monitorStatus = document.getElementById("monitor-status");
const monitorMetrics = document.getElementById("monitor-metrics");
const refresh = document.getElementById("refresh-monitoring");

async function refreshMonitoring() {
  if (refresh.disabled) return;
  refresh.disabled = true;
  monitorStatus.textContent = "Loading monitoring…";
  try {
    const data = await fetchJson(monitoring.dataset.monitorUrl);
    const recent = data.recent_predictions;
    document.getElementById("prediction-count").textContent = count.format(recent.count);
    document.getElementById("outcome-count").textContent = count.format(data.observed_performance.labeled_count);
    document.getElementById("flagged-rate").textContent = recent.predicted_churn_rate === null
      ? "—"
      : percent.format(recent.predicted_churn_rate);
    const drift = data.score_drift.total_variation;
    document.getElementById("drift-value").textContent = drift === null ? "—" : drift.toFixed(3);
    monitorStatus.textContent = drift === null
      ? `Score drift is available after ${data.minimum_sample_size} predictions.`
      : "";
    monitorMetrics.hidden = false;
  } catch {
    monitorStatus.textContent = "Monitoring is unavailable. Try refreshing.";
  } finally {
    refresh.disabled = false;
  }
}

refresh.addEventListener("click", refreshMonitoring);
const views = document.querySelectorAll("[data-view]");
for (const control of views) {
  control.addEventListener("click", () => {
    for (const button of views) {
      const selected = button === control;
      button.setAttribute("aria-pressed", String(selected));
      document.getElementById(button.getAttribute("aria-controls")).hidden = !selected;
    }
    if (control.dataset.view === "monitoring") refreshMonitoring();
  });
}
