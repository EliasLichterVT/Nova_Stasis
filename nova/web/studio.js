const form = document.querySelector("#studio");
const field = name => form.elements.namedItem(name);
const assets = new Map();
const jobs = new Map();
const lists = {profile: [], preset: []};
const libraryRows = [];
const fileFields = ["ref_a", "ref_b", "region_a", "region_b", "source", "mask", "pose"];
const numberFields = ["strength_a", "strength_b", "lora_strength_a", "lora_strength_b",
  "width", "height", "steps", "cfg", "seed", "batch", "pose_strength", "denoise"];
const textFields = ["prompt", "negative", "anatomy", "layout", "mode", "checkpoint", "adapter",
  "vision", "controlnet", "lora_a", "lora_b"];
const modelFields = ["checkpoint", "adapter", "vision", "controlnet", "lora_a", "lora_b"];
const optional = new Set(["controlnet", "lora_a", "lora_b"]);
const uploaderTemplate = document.querySelector("#asset-template");
const preview = document.querySelector("#preview");
const message = document.querySelector("#message");
const submit = document.querySelector("#generate");
const history = document.querySelector("#history");
let uploadsInProgress = 0;
let submitting = false;
let selectedJob = null;
let starterDefaultsApplied = false;

async function api(path, {method = "GET", body} = {}) {
  const headers = {"X-Nova-Request": "1"};
  if (body && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }
  const response = await fetch(`/api${path}`, {method, headers, body});
  if (response.status === 204) return null;
  const result = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(result.detail) ? result.detail.map(error =>
      `${error.loc?.at(-1) ?? "Settings"} — ${error.msg}`).join(". ") : result.detail;
    throw new Error(detail ?? "The request failed.");
  }
  return result;
}

function notify(text, isError = false) {
  message.textContent = text;
  message.hidden = !text;
  message.classList.toggle("error", isError);
}

function resetAttestation() {
  field("authorized").checked = false;
}

function syncSubmit() {
  submit.disabled = submitting || uploadsInProgress > 0;
}

function asset(name, label) {
  const element = uploaderTemplate.content.firstElementChild.cloneNode(true);
  const input = element.querySelector("input");
  const image = element.querySelector("img");
  const copy = element.querySelector(".drop-copy");
  const remove = element.querySelector("button");
  copy.querySelector("b").textContent = label;
  input.setAttribute("aria-label", label);
  image.alt = label;
  let current = null;

  function set(id) {
    current = id;
    image.hidden = !id;
    copy.hidden = Boolean(id);
    remove.hidden = !id;
    if (id) image.src = `/api/uploads/${id}`;
    else image.removeAttribute("src");
    resetAttestation();
  }

  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return;
    const body = new FormData();
    body.append("image", file);
    input.disabled = remove.disabled = true;
    uploadsInProgress += 1;
    syncSubmit();
    try {
      const result = await api("/uploads", {method: "POST", body});
      set(result.id);
      notify("");
    } catch (error) {
      notify(error.message, true);
    } finally {
      input.value = "";
      input.disabled = remove.disabled = false;
      uploadsInProgress -= 1;
      syncSubmit();
    }
  });
  remove.addEventListener("click", () => set(null));
  assets.set(name, {element, set, get id() { return current; }});
  return element;
}

function selectValue(select, value) {
  if (value && !Array.from(select.options).some(option => option.value === value)) {
    select.add(new Option(`${value} (unavailable)`, value));
  }
  select.value = value || "";
}

function buildLibrary(target, kind, read, apply) {
  target.innerHTML = `
    <div class="library-row"><select aria-label="Saved ${kind}"></select>
      <button type="button" data-action="load">Load</button>
      <button type="button" data-action="delete" aria-label="Delete saved ${kind}">×</button></div>
    <div class="library-row"><label>${kind === "profile" ? "Profile" : "Preset"} name
      <input type="text" maxlength="80"></label><button type="button" data-action="save">Save</button></div>`;
  const select = target.querySelector("select");
  const input = target.querySelector("input");
  libraryRows.push({kind, select});
  target.addEventListener("click", async event => {
    const action = event.target.dataset.action;
    if (!action) return;
    const item = lists[kind].find(entry => entry.id === select.value);
    try {
      if (action === "load" && item) {
        apply(item);
        notify(`Loaded ${item.name}.`);
      }
      if (action === "save") {
        if (!input.value.trim()) throw new Error(`Enter a ${kind} name.`);
        const record = await api(`/library/${kind}`, {
          method: "POST", body: {name: input.value.trim(), ...read()},
        });
        await loadLibrary(kind);
        select.value = record.id;
        input.value = "";
        notify(`Saved ${record.name}.`);
      }
      if (action === "delete" && item) {
        await api(`/library/${kind}/${item.id}`, {method: "DELETE"});
        await loadLibrary(kind);
        notify(`Deleted ${item.name}.`);
      }
    } catch (error) {
      notify(error.message, true);
    }
  });
}

async function loadLibrary(kind) {
  lists[kind] = await api(`/library/${kind}`);
  for (const row of libraryRows.filter(row => row.kind === kind)) {
    const previous = row.select.value;
    row.select.replaceChildren(new Option(`Select ${kind}`, ""), ...lists[kind].map(
      item => new Option(item.name, item.id),
    ));
    row.select.value = lists[kind].some(item => item.id === previous) ? previous : "";
  }
}

function buildCharacter(slot) {
  const card = document.createElement("article");
  card.className = "character";
  card.innerHTML = `
    <h3>Character <span>${slot.toUpperCase()}</span></h3>
    <div class="reference"></div>
    <label>Reference strength <output>0.75</output><input name="strength_${slot}" type="range" min=".05" max="1.5" step=".05" value=".75"></label>
    <details><summary>Mask &amp; LoRA</summary><div class="mask"></div>
      <label>LoRA<select name="lora_${slot}"></select></label>
      <label>LoRA strength<input name="lora_strength_${slot}" type="number" min="0" max="1.5" step=".05" value=".65" required></label>
    </details>
    <details><summary>Character profiles</summary><div class="library"></div></details>`;
  document.querySelector("#characters").append(card);
  card.querySelector(".reference").append(asset(`ref_${slot}`, `Upload reference ${slot.toUpperCase()}`));
  card.querySelector(".mask").append(asset(`region_${slot}`, `Attention mask ${slot.toUpperCase()}`));
  const strength = field(`strength_${slot}`);
  strength.addEventListener("input", () => {
    card.querySelector("output").value = Number(strength.value).toFixed(2);
  });
  buildLibrary(card.querySelector(".library"), "profile", () => {
    const reference = assets.get(`ref_${slot}`).id;
    if (!reference) throw new Error("Upload a reference before saving a profile.");
    return {reference, strength: Number(strength.value), lora: field(`lora_${slot}`).value || null,
      lora_strength: Number(field(`lora_strength_${slot}`).value)};
  }, item => {
    assets.get(`ref_${slot}`).set(item.reference);
    strength.value = item.strength;
    strength.dispatchEvent(new Event("input"));
    selectValue(field(`lora_${slot}`), item.lora);
    field(`lora_strength_${slot}`).value = item.lora_strength;
  });
}
buildCharacter("a");
buildCharacter("b");

document.querySelector("#scene-assets").append(asset("source", "Source image"), asset("mask", "White-on-black edit mask"));
document.querySelector("#pose-asset").append(asset("pose", "OpenPose map"));

function updateMode() {
  const mode = field("mode").value;
  assets.get("source").element.hidden = mode === "text";
  assets.get("mask").element.hidden = mode !== "inpaint";
  document.querySelector("#denoise").hidden = mode === "text";
}
field("mode").addEventListener("change", updateMode);
updateMode();

function settings() {
  for (const name of [...textFields, ...numberFields]) {
    if (!field(name).checkValidity()) throw new Error(`Check the ${name.replaceAll("_", " ")} setting.`);
  }
  const result = {};
  for (const name of textFields) result[name] = optional.has(name) ? field(name).value || null : field(name).value;
  for (const name of numberFields) result[name] = field(name).value === "" ? null : Number(field(name).value);
  for (const name of fileFields) result[name] = assets.get(name).id;
  if (!result.ref_a || !result.ref_b) throw new Error("Upload references for Character A and Character B.");
  if (result.mode !== "text" && !result.source) throw new Error("Upload a source image for this mode.");
  if (result.mode === "inpaint" && !result.mask) throw new Error("Upload a white-on-black edit mask.");
  if (result.mode === "text") result.source = null;
  if (result.mode !== "inpaint") result.mask = null;
  if (Boolean(result.pose) !== Boolean(result.controlnet)) {
    throw new Error("Select both an OpenPose map and its ControlNet model.");
  }
  return result;
}

function applySettings(saved) {
  for (const name of [...textFields, ...numberFields]) {
    if (modelFields.includes(name)) selectValue(field(name), saved[name]);
    else field(name).value = saved[name] ?? (name === "anatomy" ? "high" : "");
  }
  for (const name of fileFields) {
    assets.get(name).set(saved[name] ?? null);
    if (saved[name]) {
      const details = assets.get(name).element.closest("details");
      if (details) details.open = true;
    }
  }
  for (const name of ["strength_a", "strength_b"]) field(name).dispatchEvent(new Event("input"));
  updateMode();
  field("prompt").dispatchEvent(new Event("input"));
  field("layout").dispatchEvent(new Event("input"));
  resetAttestation();
  form.scrollIntoView({block: "start"});
}

buildLibrary(document.querySelector("#presets"), "preset", () => ({settings: settings()}),
  item => applySettings(item.settings));

async function models() {
  const label = document.querySelector("#connection");
  const refresh = document.querySelector("#refresh-models");
  refresh.disabled = true;
  label.textContent = "Checking ComfyUI…";
  try {
    const result = await api("/models");
    for (const name of modelFields) {
      const select = field(name);
      const available = result[name === "lora_b" ? "lora_a" : name] ?? [];
      const prior = select.value;
      select.replaceChildren(new Option(optional.has(name) ? "None" : "Select model", ""),
        ...available.map(model => new Option(model, model)));
      selectValue(select, prior || (!optional.has(name) && available.length === 1 ? available[0] : ""));
    }
    if (!starterDefaultsApplied && result.starter_checkpoint
        && field("checkpoint").value === result.starter_checkpoint) {
      for (const name of ["width", "height"]) {
        if (field(name).value === field(name).defaultValue) field(name).value = "512";
      }
      starterDefaultsApplied = true;
    }
    label.textContent = ["checkpoint", "adapter", "vision"].every(name => result[name]?.length)
      ? "ComfyUI connected" : "ComfyUI · models missing";
  } catch {
    label.textContent = "ComfyUI offline";
  } finally {
    refresh.disabled = false;
  }
}
document.querySelector("#refresh-models").addEventListener("click", models);

function showImage(job, index) {
  selectedJob = job.id;
  const image = new Image();
  image.alt = `Generated scene, image ${index + 1}`;
  image.src = `/api/generations/${job.id}/images/${index}`;
  image.addEventListener("error", () => {
    if (preview.contains(image)) notify("The image is unavailable. Check ComfyUI and its output folder.", true);
  });
  preview.replaceChildren(image);
}

function renderHistory() {
  const sorted = [...jobs.values()].sort((a, b) => b.created.localeCompare(a.created));
  history.replaceChildren();
  if (!sorted.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No generations yet.";
    history.append(empty);
  }
  for (const job of sorted) {
    const article = document.createElement("article");
    const prompt = document.createElement("p");
    prompt.textContent = job.settings.prompt;
    const status = document.createElement("p");
    status.className = "muted";
    status.textContent = `${job.status} · Seed ${job.seed} · ${new Date(job.created).toLocaleString()}`;
    article.append(prompt, status);
    if (job.error) {
      const error = document.createElement("p");
      error.textContent = job.error;
      article.append(error);
    }
    if (job.images.length) {
      const gallery = document.createElement("div");
      gallery.className = "gallery";
      job.images.forEach((item, index) => {
        const url = `/api/generations/${job.id}/images/${index}`;
        const button = document.createElement("button");
        button.type = "button";
        button.addEventListener("click", () => showImage(job, index));
        const image = new Image();
        image.src = url;
        image.loading = "lazy";
        image.alt = `Preview image ${index + 1}`;
        button.append(image);
        const download = document.createElement("a");
        download.href = url;
        download.download = item.filename;
        download.textContent = `Save ${index + 1}`;
        gallery.append(button, download);
      });
      article.append(gallery);
    }
    const reuse = document.createElement("button");
    reuse.type = "button";
    reuse.textContent = "Reuse settings";
    reuse.addEventListener("click", () => applySettings({...job.settings, seed: job.seed}));
    article.append(reuse);
    history.append(article);
  }
  if (!selectedJob) {
    const latest = sorted.find(job => job.images.length);
    if (latest) showImage(latest, 0);
  }
}

async function poll() {
  const pending = [...jobs.values()].filter(job => job.status === "queued");
  let updated = false;
  await Promise.all(pending.map(async job => {
    try {
      const latest = await api(`/generations/${job.id}`);
      if (latest.status === "queued") return;
      jobs.set(latest.id, latest);
      updated = true;
      if (latest.status === "complete" && latest.id === selectedJob) {
        showImage(latest, 0);
        notify("Scene complete.");
      }
      if (latest.status === "failed" && latest.id === selectedJob) notify(latest.error, true);
    } catch (error) {
      if (job.id === selectedJob) notify(error.message, true);
    }
  }));
  if (updated) renderHistory();
  setTimeout(poll, 3000);
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (submitting || uploadsInProgress) return;
  submitting = true;
  syncSubmit();
  try {
    if (!event.submitter || event.submitter !== submit) return;
    const body = {...settings(), authorized: field("authorized").checked};
    notify("Submitting to ComfyUI…");
    const job = await api("/generations", {method: "POST", body});
    selectedJob = job.id;
    jobs.set(job.id, job);
    resetAttestation();
    notify(`Scene queued · Seed ${job.seed}`);
    renderHistory();
  } catch (error) {
    notify(error.message, true);
  } finally {
    submitting = false;
    syncSubmit();
  }
});

models();
for (const kind of Object.keys(lists)) loadLibrary(kind).catch(error => notify(error.message, true));
api("/generations").then(records => {
  for (const record of records) {
    if (!jobs.has(record.id)) jobs.set(record.id, record);
  }
  renderHistory();
}).catch(error => notify(error.message, true)).finally(poll);

const importForm = document.querySelector("#model-import");
const importFields = importForm.elements;
const inspectButton = document.querySelector("#inspect-model");
const installButton = document.querySelector("#install-model");
const importOptions = document.querySelector("#import-options");
const importStatus = document.querySelector("#import-status");
const importProgress = document.querySelector("#import-progress");
let inspectedModel = null;
let inspectedLink = null;
let importBusy = false;
let importsEnabled = false;
let pollingImport = false;

function importSize(bytes) {
  const unit = bytes >= 1024 ** 3 ? "GiB" : "MiB";
  return `${(bytes / (unit === "GiB" ? 1024 ** 3 : 1024 ** 2)).toFixed(1)} ${unit}`;
}

function displayImportJob(job) {
  importBusy = job?.state === "downloading";
  installButton.disabled = importBusy;
  importProgress.hidden = !importBusy;
  if (importBusy) {
    const {received, total, name} = job;
    importStatus.textContent = `Downloading ${name} · ${importSize(received)}`;
    if (total) {
      importStatus.textContent += ` / ${importSize(total)}`;
      importProgress.max = total;
      importProgress.value = received;
    } else {
      importProgress.removeAttribute("value");
    }
  } else if (job?.state === "complete") {
    importStatus.textContent = job.warning || `Imported ${job.name}.`;
  } else if (job?.state === "failed") {
    importStatus.textContent = job.error;
  }
}

async function pollImport() {
  if (pollingImport) return;
  pollingImport = true;
  try {
    do {
      const result = await api("/imports");
      displayImportJob(result.job);
      if (!importBusy && result.job?.state === "complete") await models();
      if (importBusy) await new Promise(resolve => setTimeout(resolve, 1000));
    } while (importBusy);
  } catch (error) {
    importStatus.textContent = error.message;
    importProgress.hidden = true;
    installButton.disabled = importBusy = false;
  } finally {
    pollingImport = false;
  }
}

function selectedVersion() {
  return inspectedModel.versions.find(version => String(version.id) === importFields.version.value);
}

function populateCategory() {
  const chosen = selectedVersion().files.find(file => String(file.id) === importFields.file.value);
  importFields.category.value = chosen.category || "";
  showGrabbedTriggers(chosen);
}

function populateFiles() {
  importFields.file.replaceChildren(...selectedVersion().files.map(file =>
    new Option(file.bytes ? `${file.name} · ${importSize(file.bytes)}` : file.name, file.id),
  ));
  populateCategory();
}

importFields.version.addEventListener("change", populateFiles);
importFields.file.addEventListener("change", populateCategory);
importFields.url.addEventListener("input", () => {
  if (importFields.url.value.trim() !== inspectedLink) importOptions.hidden = true;
});
importForm.addEventListener("submit", async event => {
  event.preventDefault();
  if (!importsEnabled || importBusy) return;
  inspectButton.disabled = true;
  importOptions.hidden = true;
  importStatus.textContent = "Looking up files on Civitai…";
  const url = importFields.url.value.trim();
  try {
    inspectedModel = await api("/imports/inspect", {
      method: "POST", body: {url, api_key: importFields.api_key.value || null},
    });
    if (url !== importFields.url.value.trim()) {
      importStatus.textContent = "Find files for the new link.";
      return;
    }
    inspectedLink = url;
    document.querySelector("#import-model-name").textContent = inspectedModel.model;
    importFields.version.replaceChildren(...inspectedModel.versions.map(version =>
      new Option(version.base ? `${version.name} · ${version.base}` : version.name, version.id),
    ));
    populateFiles();
    importOptions.hidden = false;
    importStatus.textContent = "Select a file and model folder.";
  } catch (error) {
    importStatus.textContent = error.message;
  } finally {
    inspectButton.disabled = false;
  }
});
installButton.addEventListener("click", async () => {
  if (importBusy || inspectedLink !== importFields.url.value.trim()) return;
  if (!importFields.category.value) {
    importStatus.textContent = "Choose a model folder.";
    importFields.category.focus();
    return;
  }
  installButton.disabled = importBusy = true;
  try {
    const job = await api("/imports", {method: "POST", body: {
      url: inspectedLink, api_key: importFields.api_key.value || null,
      version: Number(importFields.version.value), file: Number(importFields.file.value),
      category: importFields.category.value,
    }});
    displayImportJob(job);
    pollImport();
  } catch (error) {
    importStatus.textContent = error.message;
    installButton.disabled = importBusy = false;
  }
});

inspectButton.disabled = true;
api("/imports").then(result => {
  importsEnabled = result.enabled;
  inspectButton.disabled = !importsEnabled;
  if (!importsEnabled) {
    importStatus.textContent = "Use the Nova launcher or set NOVA_MODELS_DIR to enable model imports.";
    return;
  }
  importFields.category.replaceChildren(new Option("Choose model folder", ""),
    ...Object.entries(result.categories).map(([key, label]) => new Option(label, key)));
  if (result.job) {
    importForm.closest("details").open = true;
    displayImportJob(result.job);
    if (importBusy) pollImport();
  }
}).catch(error => { importStatus.textContent = error.message; });

const storyText = document.querySelector("#story-text");
const storyNameA = document.querySelector("#story-name-a");
const storyNameB = document.querySelector("#story-name-b");
const translateButton = document.querySelector("#translate-story");
const storyResults = document.querySelector("#story-results");
const storySelect = document.querySelector("#story-scenes");
const storyPreview = document.querySelector("#story-preview");
const storyStatus = document.querySelector("#story-status");
let storyPrompts = [];
let storyRevision = 0;

function storyMessage(text) {
  storyStatus.textContent = text;
  storyStatus.hidden = !text;
}

for (const input of [storyText, storyNameA, storyNameB, field("layout")]) {
  input.addEventListener("input", () => {
    storyRevision += 1;
    if (storyResults.hidden) return;
    storyResults.hidden = true;
    storyMessage("Translate the revised story to update its prompts.");
  });
}
storySelect.addEventListener("change", () => {
  storyPreview.value = storyPrompts[Number(storySelect.value)].prompt;
  refreshStoryLoras();
});
storyPreview.addEventListener("input", () => {
  storyPrompts[Number(storySelect.value)].prompt = storyPreview.value;
  suggestionRequest += 1;
  suggestionList.replaceChildren();
  suggestionStatus.textContent = "Select Match prompt to update suggestions.";
});
translateButton.addEventListener("click", async () => {
  if (!storyText.value.trim()) {
    storyMessage("Type or paste a story first.");
    storyText.focus();
    return;
  }
  const revision = storyRevision;
  translateButton.disabled = true;
  storyResults.hidden = true;
  storyMessage("Preparing scene prompts…");
  try {
    const result = await api("/story-prompts", {method: "POST", body: {
      story: storyText.value, name_a: storyNameA.value, name_b: storyNameB.value, layout: field("layout").value,
    }});
    if (storyRevision !== revision) {
      storyMessage("Translate the revised story to update its prompts.");
      return;
    }
    storyPrompts = result.scenes;
    storySelect.replaceChildren(...storyPrompts.map((scene, index) => new Option(scene.title, index)));
    storyPreview.value = storyPrompts[0].prompt;
    storyResults.hidden = false;
    refreshStoryLoras();
    storyMessage(result.truncated ? "Showing the first 12 scenes. Translate the rest in a separate pass." : "");
  } catch (error) {
    storyMessage(error.message);
  } finally {
    translateButton.disabled = false;
  }
});
document.querySelector("#apply-story").addEventListener("click", () => {
  const prompt = storyPreview.value.trim();
  if (!prompt) {
    storyMessage("Enter a scene prompt first.");
    storyPreview.focus();
    return;
  }
  field("prompt").value = prompt;
  field("prompt").dispatchEvent(new Event("input"));
  resetAttestation();
  storyMessage("Prompt applied. You can edit it in the Prompt field before generating.");
});

const keywordTool = document.querySelector("#lora-keyword-tool");
const keywordSelect = document.querySelector("#keyword-lora");
const triggersInput = document.querySelector("#lora-triggers");
const keywordsInput = document.querySelector("#lora-keywords");
const keywordStatus = document.querySelector("#lora-keyword-status");
const saveKeywords = document.querySelector("#save-lora-keywords");
const restoreKeywords = document.querySelector("#restore-lora-keywords");
const civitaiTriggers = document.querySelector("#civitai-lora-triggers");
const grabTriggers = document.querySelector("#grab-lora-triggers");
const suggestionList = document.querySelector("#lora-suggestion-list");
const suggestionStatus = document.querySelector("#lora-suggestion-status");
let loraCatalog = [];
let openingGrabbedKeywords = false;
let suggestionRequest = 0;

function loadKeywordFields() {
  const entry = loraCatalog.find(item => item.name === keywordSelect.value);
  triggersInput.value = entry?.triggers.join("\n") ?? "";
  keywordsInput.value = entry?.keywords.join("\n") ?? "";
  saveKeywords.disabled = !entry;
  restoreKeywords.disabled = !entry?.source_triggers.length;
}

async function refreshKeywordLibrary() {
  loraCatalog = await api("/lora-keywords");
  const selected = keywordSelect.value;
  keywordSelect.replaceChildren(new Option("Select a LoRA", ""),
    ...loraCatalog.map(entry => new Option(entry.name, entry.name)));
  if (loraCatalog.some(entry => entry.name === selected)) keywordSelect.value = selected;
  loadKeywordFields();
  keywordStatus.textContent = loraCatalog.length ? "" : "Install a LoRA to add its keywords.";
}

keywordTool.addEventListener("toggle", () => {
  if (openingGrabbedKeywords) {
    openingGrabbedKeywords = false;
    return;
  }
  if (keywordTool.open) refreshKeywordLibrary().catch(error => { keywordStatus.textContent = error.message; });
});
keywordSelect.addEventListener("change", loadKeywordFields);
restoreKeywords.addEventListener("click", () => {
  const entry = loraCatalog.find(item => item.name === keywordSelect.value);
  triggersInput.value = entry.source_triggers.join("\n");
  keywordStatus.textContent = "Review the imported words, then save.";
});
saveKeywords.addEventListener("click", async () => {
  const name = keywordSelect.value;
  if (!name) return;
  saveKeywords.disabled = true;
  try {
    const entry = await api("/lora-keywords", {method: "PUT", body: {
      name, triggers: triggersInput.value.split("\n").map(text => text.trim()).filter(Boolean),
      keywords: keywordsInput.value.split("\n").map(text => text.trim()).filter(Boolean),
    }});
    loraCatalog = loraCatalog.map(item => item.name === name ? entry : item);
    if (keywordSelect.value === name) loadKeywordFields();
    keywordStatus.textContent = "Keywords saved.";
    if (!storyResults.hidden) refreshStoryLoras();
  } catch (error) {
    saveKeywords.disabled = false;
    keywordStatus.textContent = error.message;
  }
});

function showGrabbedTriggers(file) {
  const words = selectedVersion().triggers ?? [];
  const show = file.category === "loras" && words.length > 0;
  civitaiTriggers.hidden = grabTriggers.hidden = !show;
  civitaiTriggers.textContent = show ? `Trigger words · ${words.join(", ")}` : "";
}

grabTriggers.addEventListener("click", async () => {
  const version = selectedVersion();
  const file = version.files.find(item => String(item.id) === importFields.file.value);
  grabTriggers.disabled = true;
  try {
    await refreshKeywordLibrary();
    const matches = loraCatalog.filter(entry => entry.name.split(/[/\\]/).at(-1) === file.name);
    if (matches.length !== 1) {
      importStatus.textContent = "Import this LoRA first, or copy the shown trigger words to an installed LoRA in the keyword library.";
      return;
    }
    if (!keywordTool.open) {
      openingGrabbedKeywords = true;
      keywordTool.open = true;
    }
    keywordSelect.value = matches[0].name;
    loadKeywordFields();
    triggersInput.value = version.triggers.join("\n");
    keywordStatus.textContent = "Civitai words loaded. Review them and save.";
    triggersInput.focus();
  } catch (error) {
    importStatus.textContent = error.message;
  } finally {
    grabTriggers.disabled = false;
  }
});

function normalizedPrompt(text) {
  return ` ${(text.normalize("NFKC").toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []).join(" ")} `;
}

function useStoryLora(entry, slot) {
  const prompt = storyPreview.value.trim();
  const normalized = normalizedPrompt(prompt);
  const missing = entry.triggers.filter(term => {
    const normalizedTerm = normalizedPrompt(term);
    return !normalizedTerm.trim() || !normalized.includes(normalizedTerm);
  });
  const next = missing.length ? `${prompt}\n${missing.join(", ")}` : prompt;
  if (next.length > storyPreview.maxLength) {
    suggestionStatus.textContent = "Shorten the prompt before adding these trigger words.";
    return;
  }
  const other = field(slot === "a" ? "lora_b" : "lora_a");
  if (other.value !== entry.name) selectValue(field(`lora_${slot}`), entry.name);
  storyPreview.value = next;
  storyPrompts[Number(storySelect.value)].prompt = next;
  document.querySelector("#apply-story").click();
  refreshStoryLoras();
}

function showStoryLoras(entries) {
  suggestionList.replaceChildren();
  for (const entry of entries) {
    const card = document.createElement("article");
    const heading = document.createElement("strong");
    heading.textContent = entry.title ? `${entry.title} · ${entry.name}` : entry.name;
    const description = document.createElement("p");
    description.className = "muted";
    description.textContent = [
      entry.base, entry.selected ? "Selected" : "",
      entry.matches.length ? `Matches ${entry.matches.join(", ")}` : "",
      entry.triggers.length ? `Triggers ${entry.triggers.join(", ")}` : "",
    ].filter(Boolean).join(" · ");
    card.append(heading, description);
    for (const slot of ["a", "b"]) {
      if (entry.selected && field(`lora_${slot}`).value !== entry.name) continue;
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `Use on ${slot.toUpperCase()}`;
      button.addEventListener("click", () => useStoryLora(entry, slot));
      card.append(button);
    }
    suggestionList.append(card);
  }
}

async function refreshStoryLoras() {
  const prompt = storyPreview.value.trim();
  const request = ++suggestionRequest;
  suggestionList.replaceChildren();
  if (!prompt) {
    suggestionStatus.textContent = "Enter a scene prompt to match LoRAs.";
    return;
  }
  suggestionStatus.textContent = "Matching installed LoRAs…";
  try {
    const entries = await api("/lora-keywords/suggest", {method: "POST", body: {
      prompt, selected: [field("lora_a").value, field("lora_b").value].filter(Boolean),
    }});
    if (request !== suggestionRequest || storyResults.hidden) return;
    showStoryLoras(entries);
    suggestionStatus.textContent = entries.length ? "" : "No installed LoRAs match this prompt.";
  } catch (error) {
    if (request === suggestionRequest) suggestionStatus.textContent = error.message;
  }
}

document.querySelector("#refresh-lora-suggestions").addEventListener("click", refreshStoryLoras);

for (const slot of ["a", "b"]) {
  field(`lora_${slot}`).addEventListener("change", () => {
    if (!storyResults.hidden) refreshStoryLoras();
  });
}

const anatomyPreview = document.querySelector("#combined-negative-panel");
const anatomyButton = document.querySelector("#preview-negative");
const anatomyStatus = document.querySelector("#anatomy-status");
let anatomyRevision = 0;

for (const name of ["prompt", "negative", "anatomy"]) {
  field(name).addEventListener("input", () => {
    anatomyRevision += 1;
    anatomyPreview.hidden = true;
    anatomyStatus.hidden = true;
  });
}

anatomyButton.addEventListener("click", async () => {
  const revision = anatomyRevision;
  anatomyButton.disabled = true;
  anatomyStatus.hidden = true;
  try {
    const {negative} = await api("/negative-prompt", {method: "POST", body: {
      prompt: field("prompt").value, negative: field("negative").value, anatomy: field("anatomy").value,
    }});
    if (revision !== anatomyRevision) return;
    document.querySelector("#combined-negative").value = negative;
    anatomyPreview.hidden = false;
  } catch (error) {
    if (revision === anatomyRevision) {
      anatomyStatus.textContent = error.message;
      anatomyStatus.hidden = false;
    }
  } finally {
    anatomyButton.disabled = false;
  }
});
