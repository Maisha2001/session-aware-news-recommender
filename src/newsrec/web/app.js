"use strict";
const byId = id => document.getElementById(id);
let state = null;
let reportData = null;
let busy = false;
let dialogArticle = null;
const descriptions = {
  relevance: "Order by the frozen model's ranking score.",
  fresh: "Trade some relevance for more recently observed stories.",
  balanced: "Balance relevance, first-observation age and content/category variety."
};

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function announce(message) {
  byId("status").textContent = message;
  byId("status").hidden = !message;
}

function showError(error) {
  byId("error-text").textContent = error.message || "Unable to load the local service.";
  byId("error").hidden = false;
  announce("");
}

async function fetchJson(path, options = {}) {
  const response = await fetch(path, {credentials: "same-origin", ...options});
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "This action could not be completed.");
  return value;
}

function feedbackButtons(item) {
  const row = node("div", undefined, "feedback-row");
  for (const [kind, label] of [["click", "More like this"], ["dismiss", "Less like this"]]) {
    const selected = item.feedback === kind;
    const button = node("button", selected ? (kind === "click" ? "Interest saved" : "Hidden next refresh") : label, "feedback" + (selected ? " chosen" : ""));
    button.disabled = busy || Boolean(item.feedback);
    button.setAttribute("aria-label", label + ": " + item.title);
    button.dataset.focusKey = "feedback:" + item.article_id + ":" + kind;
    button.addEventListener("click", () => act("feedback", kind, item.article_id));
    row.append(button);
  }
  return row;
}

function openStory(id) {
  const item = state.items.find(value => value.article_id === id);
  if (!item) return;
  dialogArticle = id;
  byId("dialog-title").textContent = item.title;
  byId("dialog-category").textContent = item.category;
  byId("dialog-summary").textContent = item.abstract;
  byId("dialog-feedback").replaceChildren(...feedbackButtons(item).children);
  const details = byId("dialog-details");
  details.replaceChildren();
  const values = [["Ranking score", item.score.toFixed(3)], ["Starting-interest similarity", item.history_similarity.toFixed(3)],
    ["Recent-choice similarity", item.session_similarity.toFixed(3)], ["First observed", Math.round(item.first_seen_age_hours) + " hours earlier"]];
  for (const [label, value] of values) details.append(node("dt", label), node("dd", value));
  if (!byId("story-dialog").open) byId("story-dialog").showModal();
}

function renderFeed() {
  const profiles = byId("profiles");
  profiles.replaceChildren();
  for (const profile of state.profiles) {
    const button = node("button", profile.label, "profile" + (profile.id === state.profile ? " active" : ""));
    button.setAttribute("aria-pressed", String(profile.id === state.profile));
    button.title = profile.description;
    button.dataset.focusKey = "profile:" + profile.id;
    button.disabled = busy;
    button.addEventListener("click", () => act("profile", profile.id));
    profiles.append(button);
  }
  byId("clock").textContent = state.time_label;
  byId("refresh").disabled = busy || !state.can_advance;
  byId("reset").disabled = busy;
  document.querySelectorAll("input[name=policy]").forEach(input => {if (!busy) input.checked = input.value === state.policy; input.disabled = busy;});
  byId("policy-description").textContent = descriptions[state.policy];
  byId("feed-meta").textContent = `${state.items.length} stories · ${state.session_groups} recent feedback ${state.session_groups === 1 ? "group" : "groups"} in context`;
  const grid = byId("feed");
  grid.setAttribute("aria-busy", String(busy));
  grid.replaceChildren();
  for (const item of state.items) {
    const card = node("article", undefined, "story");
    const top = node("div", undefined, "story-top");
    top.append(node("span", item.category, "category"), node("span", `${Math.round(item.first_seen_age_hours)}h since first seen`, "story-age"));
    const bottom = node("div", undefined, "story-bottom");
    const read = node("button", "Read summary", "read-summary");
    read.setAttribute("aria-label", "Read summary: " + item.title);
    read.dataset.focusKey = "read:" + item.article_id;
    read.addEventListener("click", () => openStory(item.article_id));
    bottom.append(read, feedbackButtons(item));
    card.append(top, node("h3", item.title), node("p", item.abstract, "story-summary"), bottom);
    grid.append(card);
  }
  if (!state.items.length) grid.append(node("p", "No stories are available for this replay moment. Start a new session to explore again.", "loading-card"));
  const activity = byId("activity");
  activity.replaceChildren();
  byId("activity-empty").hidden = state.activity.length > 0;
  for (const event of state.activity) {
    const item = node("li");
    item.append(node("strong", event.kind === "click" ? "Interested" : "Less like this"), node("span", event.title));
    activity.append(item);
  }
  const history = byId("history");
  history.replaceChildren(...state.history.map(item => node("li", item.title)));
  if (!state.history.length) history.append(node("li", "No starting history. This is the cold-start view."));
  if (dialogArticle && byId("story-dialog").open) {
    if (state.items.some(item => item.article_id === dialogArticle)) openStory(dialogArticle);
    else byId("story-dialog").close();
  }
}

async function act(action, value = "", articleId = "") {
  if (busy || !state) return;
  const focused = document.activeElement;
  const focusKey = focused.dataset.focusKey;
  busy = true;
  byId("error").hidden = true;
  renderFeed();
  try {
    state = await fetchJson("/api/action", {method: "POST", headers: {"Content-Type": "application/json", "X-News-Demo": state.csrf}, body: JSON.stringify({action, value, article_id: articleId})});
    announce(action === "feedback" ? "Feedback saved. Refresh the feed to move six minutes ahead and use it." : action === "reset" || action === "profile" ? "A new session is ready." : "Your feed is ready.");
  } catch (error) {showError(error);}
  finally {
    busy = false; renderFeed();
    if (document.activeElement === document.body) {
      const key = action === "feedback" ? "read:" + articleId : focusKey;
      const replacement = [...document.querySelectorAll("[data-focus-key]")].find(element => element.dataset.focusKey === key && !element.disabled);
      if (replacement) replacement.focus({preventScroll: true});
      else if (focused.isConnected && !focused.disabled) focused.focus({preventScroll: true});
    }
  }
}

function renderComparisons() {
  if (!reportData) return;
  const selected = reportData[byId("evaluation-split").value];
  byId("evaluation-count").textContent = selected.impressions.toLocaleString() + " impressions";
  byId("comparison-note").textContent = selected.note;
  const cards = byId("model-cards"), rows = byId("comparison-rows");
  cards.replaceChildren(); rows.replaceChildren();
  for (const row of selected.rows) {
    const card = node("section", undefined, "model-card");
    const meter = node("meter");
    meter.min = 0; meter.max = 1; meter.value = row.logged_ndcg10;
    meter.setAttribute("aria-label", row.name + " logged nDCG at10");
    card.append(node("h2", row.name), node("p", row.logged_ndcg10.toFixed(3), "metric-value"), node("p", "Logged nDCG@10 · higher is better", "caption"), meter);
    cards.append(card);
    const tr = node("tr");
    const title = node("th", row.name); title.scope = "row";
    tr.append(title, node("td", row.logged_auc.toFixed(3)), node("td", row.logged_ndcg10.toFixed(3)), node("td", (row.catalog_recall100 * 100).toFixed(2) + "%"));
    rows.append(tr);
  }
}

async function view(name) {
  byId("feed-view").hidden = name !== "feed";
  byId("results-view").hidden = name !== "results";
  for (const value of ["feed", "results"]) {
    byId(value + "-tab").classList.toggle("selected", value === name);
    byId(value + "-tab").setAttribute("aria-pressed", String(value === name));
  }
  if (name === "results" && !reportData) {
    try {
      reportData = await fetchJson("/api/comparisons");
      byId("evaluation-split").replaceChildren();
      for (const [id, report] of Object.entries(reportData)) {
        const option = node("option", report.label); option.value = id; byId("evaluation-split").append(option);
      }
      renderComparisons();
    } catch (error) {showError(error);}
  }
}

async function load() {
  byId("error").hidden = true;
  try {state = await fetchJson("/api/session"); renderFeed();}
  catch (error) {showError(error); byId("feed").setAttribute("aria-busy", "false");}
}

byId("feed-tab").addEventListener("click", () => view("feed"));
byId("results-tab").addEventListener("click", () => view("results"));
byId("evaluation-split").addEventListener("change", renderComparisons);
byId("refresh").addEventListener("click", () => act("refresh"));
byId("reset").addEventListener("click", () => act("reset"));
byId("retry").addEventListener("click", load);
byId("close-dialog").addEventListener("click", () => byId("story-dialog").close());
byId("story-dialog").addEventListener("close", () => {dialogArticle = null;});
document.querySelectorAll("input[name=policy]").forEach(input => input.addEventListener("change", () => act("policy", input.value)));
load();
