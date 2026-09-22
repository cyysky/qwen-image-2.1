"use strict";

const $ = (id) => document.getElementById(id);
const HISTORY_KEY = "qwen-image-playground:history";
const MAX_HISTORY = 60;
// Results are base64 PNGs (1-4 MB each), far past the localStorage quota, so
// they are kept in IndexedDB and restored on load. Oldest are pruned past the cap.
const DB_NAME = "qwen-image-playground";
const DB_VERSION = 1;
const DB_STORE = "results";
const MAX_RESULTS = 40;

const state = {
  config: null,
  endpoint: "",
  tab: "generate",
  refs: [],
  results: [],
  history: [],
  busy: false,
  controller: null,
  enhanced: null,
  enhanceTarget: null,
  startedAt: 0,
  timer: null,
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function num(value) {
  const text = String(value === null || value === undefined ? "" : value).trim();
  if (text === "") return null;
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function randomSeed() {
  return Math.floor(Math.random() * 2147483647);
}

function toast(message, kind) {
  const node = $("toast");
  node.textContent = message;
  node.className = ("toast " + (kind || "")).trim();
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(function () { node.hidden = true; }, kind === "error" ? 9000 : 4000);
}

function setStatus(message, kind) {
  const node = $("status");
  node.className = ("status " + (kind || "")).trim();
  node.innerHTML = message;
}

async function api(path, options) {
  const opts = options || {};
  const init = { method: opts.method || "GET", signal: opts.signal };
  if (opts.json !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
  } else if (opts.form) {
    init.body = opts.form;
  }

  let response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("Cannot reach the backend at " + path + ": " + error.message);
  }

  const text = await response.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); } catch (_) { data = null; }
  }

  if (!response.ok) {
    let message = "HTTP " + response.status;
    const detail = data && (data.error || data.detail);
    if (typeof detail === "string") message = detail;
    else if (Array.isArray(detail)) message = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    else if (detail) message = JSON.stringify(detail);
    else if (text) message = text.slice(0, 400);
    const error = new Error(message);
    error.status = response.status;
    error.payload = data;
    throw error;
  }
  return data;
}

function b64ToBlob(b64, mime) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

function mimeFor(format) {
  if (format === "jpeg" || format === "jpg") return "image/jpeg";
  if (format === "webp") return "image/webp";
  return "image/png";
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
}

async function copyText(text, label) {
  const message = label || "Copied";
  try {
    await navigator.clipboard.writeText(text);
    toast(message + " to clipboard", "ok");
  } catch (_) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    toast(message + " to clipboard", "ok");
  }
}

// ---------------------------------------------------------------------------
// result persistence (IndexedDB)
// ---------------------------------------------------------------------------
let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise(function (resolve, reject) {
    if (!window.indexedDB) {
      reject(new Error("IndexedDB is unavailable in this browser"));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = function () {
      const db = request.result;
      if (!db.objectStoreNames.contains(DB_STORE)) {
        db.createObjectStore(DB_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = function () { resolve(request.result); };
    request.onerror = function () { reject(request.error || new Error("indexedDB.open failed")); };
  });
  return dbPromise;
}

function withStore(mode, work) {
  return openDb().then(function (db) {
    return new Promise(function (resolve, reject) {
      const tx = db.transaction(DB_STORE, mode);
      const request = work(tx.objectStore(DB_STORE));
      tx.oncomplete = function () {
        resolve(request && "result" in request ? request.result : undefined);
      };
      tx.onerror = function () { reject(tx.error); };
      tx.onabort = function () { reject(tx.error || new Error("transaction aborted")); };
    });
  });
}

function idbPut(record) { return withStore("readwrite", function (store) { return store.put(record); }); }
function idbDelete(id) { return withStore("readwrite", function (store) { return store.delete(id); }); }
function idbClear() { return withStore("readwrite", function (store) { return store.clear(); }); }
function idbAll() { return withStore("readonly", function (store) { return store.getAll(); }); }

function persistResult(result) {
  // `raw` keeps a copy of the whole response, including the base64 image, so the
  // stored copy is trimmed: the UI only ever renders the trimmed form anyway.
  return idbPut({
    id: result.id,
    blob: result.blob || null,
    url: result.url || null,
    revised_prompt: result.revised_prompt || null,
    meta: result.meta,
    raw: trimRaw(result.raw),
    format: result.format,
    ts: result.ts || Date.now(),
  });
}

function dropResult(id) {
  idbDelete(id).catch(function (error) { console.warn("Could not remove result from IndexedDB", error); });
}

async function pruneResults() {
  if (state.results.length <= MAX_RESULTS) return;
  const dropped = state.results.slice(MAX_RESULTS);
  state.results = state.results.slice(0, MAX_RESULTS);
  dropped.forEach(function (item) {
    if (item.blob) URL.revokeObjectURL(item.src);
    dropResult(item.id);
  });
  renderResults();
}

function updateStorageNote() {
  const node = $("storage-note");
  if (!node) return;
  const bytes = state.results.reduce(function (total, item) {
    return total + (item.blob && item.blob.size ? item.blob.size : 0);
  }, 0);
  node.textContent = state.results.length
    ? state.results.length + " saved in this browser (" + (bytes / (1024 * 1024)).toFixed(1) + " MB)"
    : "";
  node.title = "Results are stored in IndexedDB and reload with the page.";
}

async function restoreResults() {
  let records = [];
  try {
    records = (await idbAll()) || [];
  } catch (error) {
    const node = $("storage-note");
    if (node) {
      node.textContent = "session only (IndexedDB unavailable)";
      node.title = String(error.message || error);
    }
    return;
  }
  records.sort(function (a, b) { return (b.ts || 0) - (a.ts || 0); });
  state.results = records.map(function (record) {
    let src = record.url || null;
    if (record.blob) {
      try { src = URL.createObjectURL(record.blob); } catch (_) { src = record.url || null; }
    }
    return {
      id: record.id,
      blob: record.blob || null,
      url: record.url || null,
      src: src,
      revised_prompt: record.revised_prompt || null,
      meta: record.meta || {},
      raw: record.raw || {},
      format: record.format || "png",
      ts: record.ts || 0,
      restored: true,
    };
  });
  renderResults();
  updateStorageNote();
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------
async function loadConfig(reload) {
  const data = await api(reload ? "/api/config/reload" : "/api/config");
  state.config = data.config || data;
  applyConfig();
}

function applyConfig() {
  const config = state.config;
  if (!config) return;

  const engines = config.enhancer.engines || [];
  const engineSelect = $("enhance-engine-select");
  const previous = engineSelect.value;
  engineSelect.innerHTML = "";
  engines.forEach(function (engine) {
    const option = document.createElement("option");
    option.value = engine.key;
    option.textContent = engine.label || engine.key;
    option.dataset.supportsEdit = engine.supports_edit ? "1" : "0";
    option.dataset.supportsT2i = engine.supports_t2i ? "1" : "0";
    option.dataset.imageInputs = engine.image_inputs ? "1" : "0";
    option.title = (engine.model || engine.key) + " @ " + engine.base_url +
      (engine.kind === "pe-i2i" ? " - PE edit checkpoint, needs the reference images"
        : engine.kind === "pe" ? " - PE checkpoint, text-to-image only" : "");
    engineSelect.appendChild(option);
  });
  const wanted = engines.some(function (engine) { return engine.key === previous; })
    ? previous
    : (config.enhancer.default || (engines[0] ? engines[0].key : ""));
  if (wanted) engineSelect.value = wanted;
  syncEngineOptions();

  ["enhance-btn", "edit-enhance-btn"].forEach((id) => {
    const button = $(id);
    button.disabled = !config.enhancer.enabled;
    button.title = config.enhancer.enabled ? "Rewrite with the selected enhancer engine"
      : "Enhancer disabled (ENHANCER_ENABLED=false)";
  });
  $("enhance-toggle").disabled = !config.enhancer.enabled;
  $("enhance-toggle").parentElement.title = config.enhancer.enabled
    ? "Rewrite the prompt with the selected enhancer engine before sending it to the image endpoint"
    : "Enhancer disabled (ENHANCER_ENABLED=false)";

  renderEndpoints();

  const models = config.models || [];
  const datalist = $("model-list");
  datalist.innerHTML = "";
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    datalist.appendChild(option);
  });
  $("model-input").value = config.default_model || models[0] || "";

  const sizes = config.sizes || [];
  const sizeSelect = $("size-select");
  sizeSelect.innerHTML = "";
  sizes.forEach((size) => {
    const option = document.createElement("option");
    option.value = size;
    option.textContent = size;
    sizeSelect.appendChild(option);
  });
  const custom = document.createElement("option");
  custom.value = "custom";
  custom.textContent = "custom width x height";
  sizeSelect.appendChild(custom);

  const defaults = config.defaults || {};
  sizeSelect.value = sizes.indexOf(defaults.size) !== -1 ? defaults.size : (sizes[0] || "custom");
  $("steps-input").value = defaults.num_inference_steps === undefined ? 40 : defaults.num_inference_steps;
  $("guidance-input").value = defaults.guidance_scale === undefined ? 1 : defaults.guidance_scale;
  $("n-input").value = defaults.n === undefined ? 1 : defaults.n;
  $("format-select").value = defaults.output_format || "png";
  $("background-select").value = defaults.background || "auto";
  $("enhance-toggle").checked = Boolean(defaults.enhance);
  $("seed-input").value = randomSeed();

  const presetSelect = $("preset-select");
  presetSelect.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "none";
  presetSelect.appendChild(none);
  (config.enhancer.presets || []).forEach((preset) => {
    const option = document.createElement("option");
    option.value = preset.key;
    option.textContent = preset.key;
    option.title = preset.instruction;
    presetSelect.appendChild(option);
  });

  toggleCustomSize();
}

function renderEndpoints() {
  const container = $("endpoint-pills");
  container.innerHTML = "";
  const endpoints = state.config.endpoints || [];

  const known = endpoints.some((item) => item.name === state.endpoint);
  if (!state.endpoint || !known) {
    state.endpoint = state.config.default_endpoint || (endpoints[0] || {}).name || "";
  }

  endpoints.forEach((endpoint) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pill" + (endpoint.name === state.endpoint ? " is-active" : "");

    const dot = document.createElement("span");
    dot.className = "dot";
    dot.dataset.dot = endpoint.name;
    button.appendChild(dot);

    const label = document.createElement("span");
    label.textContent = endpoint.label || endpoint.name;
    button.appendChild(label);

    const small = document.createElement("small");
    small.textContent = endpoint.base_url.replace(/^https?:\/\//, "");
    button.appendChild(small);

    button.addEventListener("click", function () {
      state.endpoint = endpoint.name;
      renderEndpoints();
    });
    container.appendChild(button);
  });
}

function endpointBase() {
  const found = (state.config.endpoints || []).filter((item) => item.name === state.endpoint)[0];
  return found ? found.base_url : "";
}

// ---------------------------------------------------------------------------
// request parameters
// ---------------------------------------------------------------------------
function readParams() {
  const custom = $("size-select").value === "custom";
  const background = $("background-select").value;
  let format = $("format-select").value;
  if (background === "transparent") format = "png";
  return {
    model: $("model-input").value.trim() || null,
    size: custom ? null : $("size-select").value,
    width: custom ? num($("width-input").value) : null,
    height: custom ? num($("height-input").value) : null,
    num_inference_steps: num($("steps-input").value),
    guidance_scale: num($("guidance-input").value),
    seed: num($("seed-input").value),
    n: num($("n-input").value),
    output_format: format,
    background: background === "auto" ? null : background,
    negative_prompt: $("negative-input").value.trim() || null,
  };
}

function validateParams(params) {
  if (params.size && !/^\d+x\d+$/.test(params.size)) {
    return "Size must look like 1024x1024, got \"" + params.size + "\".";
  }
  if (params.size) {
    const parts = params.size.split("x").map(Number);
    const bad = parts.filter((part) => part % 32 !== 0);
    if (bad.length) return "Size " + params.size + " is not a multiple of 32 - SGLang rejects it.";
  }
  if (params.width !== null && params.width % 32 !== 0) return "Width " + params.width + " is not a multiple of 32.";
  if (params.height !== null && params.height % 32 !== 0) return "Height " + params.height + " is not a multiple of 32.";
  return null;
}

function enhancerExtras() {
  const engine = $("enhance-engine-select").value || null;
  if (!$("enhance-toggle").checked) return {};
  return {
    enhance: true,
    enhance_mode: $("preset-select").value || null,
    enhance_instruction: $("enhance-instruction").value.trim() || null,
    enhance_engine: engine,
  };
}

// ---------------------------------------------------------------------------
// prompt enhancement
// ---------------------------------------------------------------------------
function enhancerEngines() {
  return (state.config && state.config.enhancer && state.config.enhancer.engines) || [];
}

function currentEngine() {
  const key = $("enhance-engine-select").value;
  return enhancerEngines().filter(function (engine) { return engine.key === key; })[0] || null;
}

function engineScope(engine) {
  const t2i = engine.supports_t2i !== false;
  const edit = engine.supports_edit === true;
  if (t2i && edit) return "";
  return edit ? " (edit only)" : " (text-to-image only)";
}

function updateEngineNote() {
  const node = $("enhance-engine-note");
  const engine = currentEngine();
  if (!node) return;
  if (!engine) {
    node.textContent = "";
    return;
  }
  let text = "chat model - returns the rewritten prompt as plain text";
  if (engine.kind === "pe") {
    text = "Qwen-Image-2.1 PE checkpoint - returns a rewritten prompt plus a wh_ratio";
  } else if (engine.kind === "pe-i2i") {
    text = "Qwen-Image-2.1 PE edit checkpoint - reads the reference images and returns " +
      "a rewritten edit prompt plus a wh_ratio or ratio_follow";
  }
  node.textContent = text + engineScope(engine);
  $("enhancer-model-label").textContent = engine.label || engine.key;
}

// A PE engine ships only one of the two rewrite prompts, so an engine that has no
// prompt for the current tab is greyed out and the selection falls back to one that
// does.
function syncEngineOptions() {
  const select = $("enhance-engine-select");
  if (!select) return;
  const attribute = state.tab === "edit" ? "supportsEdit" : "supportsT2i";
  const options = Array.prototype.slice.call(select.options);
  options.forEach(function (option) {
    option.disabled = option.dataset[attribute] === "0";
  });
  const chosen = select.options[select.selectedIndex];
  if (chosen && chosen.disabled) {
    const fallback = options.filter(function (option) { return !option.disabled; })[0];
    if (fallback) select.value = fallback.value;
  }
  updateEngineNote();
}

function isPeKind(kind) {
  return kind === "pe" || kind === "pe-i2i";
}

function showEnhanceMeta(data) {
  const engineMeta = enhancerEngines().filter(function (engine) {
    return engine.key === data.engine;
  })[0];
  const kind = (engineMeta && engineMeta.kind) || data.kind || "";
  const tag = kind === "pe-i2i" ? " (PE-I2I)" : (kind === "pe" ? " (PE)" : "");
  $("enhance-engine-tag").textContent = data.engine
    ? "via " + ((engineMeta && engineMeta.label) || data.engine) + tag
    : "";

  const ratioRow = $("enhance-ratio");
  const ratio = data.wh_ratio || "";
  const follow = data.ratio_follow || "";
  const suggested = data.suggested_size || "";
  const sizes = (state.config && state.config.sizes) || [];
  const negative = (data.negative_prompt || "").trim();

  if (isPeKind(kind) && (ratio || follow || suggested || negative)) {
    const ratioValue = $("enhance-ratio-value");
    ratioValue.textContent = ratio || follow || "-";
    ratioValue.title = ratio
      ? "Canvas ratio chosen by the checkpoint"
      : (follow ? "Keeps the shape of " + follow + " in the reference images" : "");
    $("enhance-size-value").textContent = suggested || "-";
    const apply = $("apply-ratio-btn");
    apply.disabled = !suggested;
    apply.title = !suggested
      ? "This answer did not name a canvas"
      : (sizes.indexOf(suggested) === -1
        ? "Set a custom canvas of " + suggested + " (not in SIZES)"
        : "Set the size selector to " + suggested);
    const negativeBtn = $("apply-negative-btn");
    negativeBtn.hidden = !negative;
    negativeBtn.title = negative ? "Copy the checkpoint negative prompt into the negative prompt field" : "";
    ratioRow.hidden = false;
  } else {
    ratioRow.hidden = true;
  }

  const warning = $("enhance-warning");
  if (data.parse_ok === false) {
    warning.textContent = "The checkpoint did not return a clean JSON record, so the raw answer is shown above. Check it before using.";
    warning.hidden = false;
  } else {
    warning.textContent = "";
    warning.hidden = true;
  }
}

function applySuggestedSize() {
  const suggested = $("enhance-size-value").textContent;
  if (!suggested || suggested === "-") return;
  const select = $("size-select");
  const values = Array.prototype.slice.call(select.options).map(function (option) { return option.value; });
  if (values.indexOf(suggested) === -1) {
    const parts = suggested.split("x");
    if (parts.length !== 2) {
      toast("Size " + suggested + " is not usable - check SIZES in .env.", "error");
      return;
    }
    select.value = "custom";
    $("width-input").value = parts[0];
    $("height-input").value = parts[1];
    toggleCustomSize();
    toast("Custom size set to " + suggested + ".", "ok");
    return;
  }
  select.value = suggested;
  toggleCustomSize();
  toast("Size set to " + suggested + ".", "ok");
}

function applyEnhanceNegative() {
  const value = $("enhanced-output").value;
  const enhanced = state.enhanced;
  const negative = enhanced && enhanced.negative_prompt ? enhanced.negative_prompt : "";
  if (!negative) {
    toast("This enhancement has no negative prompt.", "error");
    return;
  }
  $("negative-input").value = negative;
  toast("Negative prompt applied.", "ok");
}

async function enhance(target) {
  const isEdit = target === "edit";
  const source = isEdit ? $("edit-prompt-input") : $("prompt-input");
  const button = isEdit ? $("edit-enhance-btn") : $("enhance-btn");
  const prompt = source.value.trim();
  const engine = $("enhance-engine-select").value || null;
  const engineMeta = currentEngine();

  if (!prompt) {
    toast("Write a prompt to enhance first.", "error");
    return;
  }
  if (!engine) {
    toast("No enhancer engine configured - check ENHANCER_ENGINES in .env.", "error");
    return;
  }
  const usesImages = isEdit && !!engineMeta && engineMeta.image_inputs === true;
  if (usesImages && state.refs.length === 0) {
    toast("This engine rewrites edits, so add at least one reference image first.", "error");
    return;
  }

  const original = button.textContent;
  button.disabled = true;
  button.textContent = usesImages ? "Reading references..." : "Enhancing...";

  try {
    let images = [];
    let imageSizes = [];
    if (usesImages) {
      const payload = await referencePayload();
      images = payload.images;
      imageSizes = payload.sizes;
      button.textContent = "Enhancing...";
    }
    const data = await api("/api/enhance", {
      method: "POST",
      json: {
        prompt: prompt,
        mode: $("preset-select").value || null,
        instruction: $("enhance-instruction").value.trim() || null,
        engine: engine,
        is_edit: isEdit,
        negative_prompt: $("negative-input").value.trim() || null,
        size: $("size-select").value === "custom" ? null : $("size-select").value,
        images: images,
        image_sizes: imageSizes,
      },
    });
    state.enhanced = {
      tab: target,
      original: prompt,
      enhanced: data.prompt,
      negative_prompt: data.negative_prompt || "",
    };
    state.enhanceTarget = target;
    $("enhanced-output").value = data.prompt;
    $("enhance-panel").hidden = false;
    showEnhanceMeta(data);
    const tokens = data.usage && data.usage.total_tokens ? ", " + data.usage.total_tokens + " tokens" : "";
    const ratio = data.wh_ratio ? " (" + data.wh_ratio + ")" : (data.ratio_follow ? " (" + data.ratio_follow + ")" : "");
    toast("Enhanced by " + (data.engine || data.model) + " in " + data.elapsed_s + "s" + tokens + ratio, "ok");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = !state.config.enhancer.enabled;
    button.textContent = original;
  }
}

// ---------------------------------------------------------------------------
// reference images
// ---------------------------------------------------------------------------
function readAsDataUrl(file) {
  return new Promise(function (resolve, reject) {
    const reader = new FileReader();
    reader.onload = function () { resolve(String(reader.result || "")); };
    reader.onerror = function () { reject(new Error("Cannot read " + file.name)); };
    reader.readAsDataURL(file);
  });
}

function imageSize(file) {
  return new Promise(function (resolve) {
    const url = URL.createObjectURL(file);
    const probe = new Image();
    const done = function (size) {
      URL.revokeObjectURL(url);
      resolve(size);
    };
    probe.onload = function () { done(probe.naturalWidth + "x" + probe.naturalHeight); };
    probe.onerror = function () { done(""); };
    probe.src = url;
  });
}

// Sizes are cached on the ref because a PE-I2I enhancer needs them to resolve a
// `ratio_follow` answer, and both the enhancer and the edit call want them.
async function referenceSizes(refs) {
  const sizes = [];
  for (let i = 0; i < refs.length; i += 1) {
    const ref = refs[i];
    if (!ref.size) ref.size = await imageSize(ref.file);
    sizes.push(ref.size || "");
  }
  return sizes;
}

async function referencePayload() {
  const images = [];
  for (let i = 0; i < state.refs.length; i += 1) {
    images.push(await readAsDataUrl(state.refs[i].file));
  }
  return { images: images, sizes: await referenceSizes(state.refs) };
}

function addReferences(fileList) {
  const limit = state.config.max_reference_images || 10;
  const files = Array.prototype.slice.call(fileList || []);
  files.forEach((file) => {
    if (!file.type || file.type.indexOf("image/") !== 0) return;
    if (state.refs.length >= limit) {
      toast("Up to " + limit + " reference images.", "error");
      return;
    }
    state.refs.push({ id: String(Date.now()) + "-" + Math.random(), name: file.name, file: file });
  });
  renderRefs();
}

function renderRefs() {
  const container = $("thumbs");
  container.innerHTML = "";
  state.refs.forEach((ref) => {
    const wrapper = document.createElement("div");
    wrapper.className = "thumb";

    const img = document.createElement("img");
    img.src = URL.createObjectURL(ref.file);
    img.alt = ref.name;
    img.title = ref.name;
    wrapper.appendChild(img);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "\u00d7";
    remove.title = "Remove " + ref.name;
    remove.addEventListener("click", function () {
      state.refs = state.refs.filter((item) => item.id !== ref.id);
      renderRefs();
    });
    wrapper.appendChild(remove);

    container.appendChild(wrapper);
  });

  const limit = state.config ? state.config.max_reference_images : 10;
  $("ref-count").textContent = state.refs.length + " / " + limit;
}

// ---------------------------------------------------------------------------
// results
// ---------------------------------------------------------------------------
function addResults(data, context) {
  const items = (data && data.data) || [];
  const params = context.params || {};
  const format = params.output_format || "png";
  const mime = mimeFor(format);

  items.forEach((item, index) => {
    let blob = null;
    if (item.b64_json) blob = b64ToBlob(item.b64_json, mime);

    const result = {
      id: String(Date.now()) + "-" + index + "-" + Math.random(),
      blob: blob,
      url: item.url || null,
      src: blob ? URL.createObjectURL(blob) : item.url,
      revised_prompt: item.revised_prompt || null,
      meta: {
        endpoint: context.endpointLabel,
        model: context.model,
        prompt: context.prompt,
        original_prompt: context.originalPrompt,
        enhanced: context.enhanced || null,
        params: params,
        inference_time_s: data.inference_time_s,
        peak_memory_mb: data.peak_memory_mb,
        wall_time_s: data.wall_time_s,
        usage: data.usage,
      },
      raw: data,
      format: format,
      ts: Date.now(),
    };
    state.results.unshift(result);
    persistResult(result).catch(function (error) {
      console.warn("Could not save the result to IndexedDB", error);
    });
  });

  renderResults();
  pruneResults().then(updateStorageNote, updateStorageNote);
}

function makeButton(label, className, handler) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.addEventListener("click", handler);
  return button;
}

function trimRaw(raw) {
  const clone = JSON.parse(JSON.stringify(raw || {}));
  (clone.data || []).forEach((item) => {
    if (item.b64_json) item.b64_json = "<base64 " + item.b64_json.length + " chars>";
  });
  return clone;
}

function renderResults() {
  const container = $("results");
  container.innerHTML = "";
  $("result-count").textContent = String(state.results.length);
  $("results-empty").hidden = state.results.length > 0;
  $("download-all-btn").disabled = state.results.length === 0;
  $("clear-results-btn").disabled = state.results.length === 0;

  state.results.forEach((result) => {
    const card = document.createElement("article");
    card.className = "card";

    if (result.src) {
      const img = document.createElement("img");
      img.src = result.src;
      img.alt = result.meta.prompt || "generated image";
      img.addEventListener("click", function () { showLightbox(result.src); });
      card.appendChild(img);
    }

    const body = document.createElement("div");
    body.className = "body";

    const prompt = document.createElement("div");
    prompt.className = "prompt";
    prompt.textContent = result.meta.prompt || "";
    if (result.meta.original_prompt && result.meta.original_prompt !== result.meta.prompt) {
      const note = document.createElement("small");
      note.textContent = " (enhanced from: " + result.meta.original_prompt + ")";
      prompt.appendChild(note);
    }
    body.appendChild(prompt);

    const params = result.meta.params || {};
    const size = params.size || (params.width && params.height ? params.width + "x" + params.height : "-");
    const entries = [
      ["endpoint", result.meta.endpoint],
      ["model", result.meta.model],
      ["size", size],
      ["seed", params.seed === undefined || params.seed === null ? "-" : params.seed],
      ["steps", params.num_inference_steps === undefined ? "-" : params.num_inference_steps],
      ["guidance", params.guidance_scale === undefined ? "-" : params.guidance_scale],
      ["background", params.background || "auto"],
    ];
    if (result.meta.enhanced) {
      entries.push(["enhancer", result.meta.enhanced.engine || result.meta.enhanced.model]);
      if (result.meta.enhanced.wh_ratio) entries.push(["ratio", result.meta.enhanced.wh_ratio]);
    }
    if (result.meta.inference_time_s !== undefined && result.meta.inference_time_s !== null) {
      entries.push(["infer", Number(result.meta.inference_time_s).toFixed(1) + "s"]);
    }
    if (result.meta.wall_time_s !== undefined && result.meta.wall_time_s !== null) {
      entries.push(["wall", Number(result.meta.wall_time_s).toFixed(1) + "s"]);
    }
    if (result.meta.peak_memory_mb) {
      entries.push(["peak", Number(result.meta.peak_memory_mb).toFixed(0) + " MB"]);
    }
    if (result.meta.usage && result.meta.usage.prompt_tokens !== undefined) {
      entries.push(["tokens", result.meta.usage.prompt_tokens]);
    }

    const chips = document.createElement("div");
    chips.className = "chips";
    entries.forEach((pair) => {
      const value = pair[1];
      if (value === undefined || value === null || value === "") return;
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.appendChild(document.createTextNode(pair[0] + " "));
      const strong = document.createElement("b");
      strong.textContent = String(value);
      chip.appendChild(strong);
      chips.appendChild(chip);
    });
    body.appendChild(chips);

    const actions = document.createElement("div");
    actions.className = "actions";
    actions.appendChild(makeButton("Download", "btn small", function () { downloadResult(result); }));
    actions.appendChild(makeButton("Use as reference", "btn small", function () { useAsReference(result); }));
    actions.appendChild(makeButton("Copy prompt", "btn small", function () {
      copyText(result.meta.prompt, "Prompt copied");
    }));
    actions.appendChild(makeButton("Remove", "btn small", function () {
      state.results = state.results.filter((item) => item.id !== result.id);
      if (result.blob) URL.revokeObjectURL(result.src);
      dropResult(result.id);
      renderResults();
      updateStorageNote();
    }));
    body.appendChild(actions);

    const details = document.createElement("details");
    details.className = "raw";
    const summary = document.createElement("summary");
    summary.textContent = "Raw response";
    details.appendChild(summary);
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(trimRaw(result.raw), null, 2);
    details.appendChild(pre);
    body.appendChild(details);

    card.appendChild(body);
    container.appendChild(card);
  });
}

function resultFilename(result) {
  const params = result.meta.params || {};
  const seed = params.seed === undefined || params.seed === null ? "seed" : params.seed;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  return "qwen-" + result.meta.endpoint + "-" + seed + "-" + stamp + "." + result.format;
}

function downloadResult(result) {
  if (!result.blob) {
    if (result.url) window.open(result.url, "_blank");
    else toast("This response has no b64_json or url to download.", "error");
    return;
  }
  download(result.blob, resultFilename(result));
}

function useAsReference(result) {
  if (!result.blob) {
    toast("Only base64 responses can be reused as a reference.", "error");
    return;
  }
  const limit = state.config.max_reference_images || 10;
  if (state.refs.length >= limit) {
    toast("Up to " + limit + " reference images.", "error");
    return;
  }
  const name = "result-" + (state.refs.length + 1) + "." + result.format;
  state.refs.push({
    id: String(Date.now()) + "-" + Math.random(),
    name: name,
    file: new File([result.blob], name, { type: result.blob.type }),
  });
  renderRefs();
  switchTab("edit");
  toast("Added to the edit references.", "ok");
}

function showLightbox(src) {
  $("lightbox-img").src = src;
  $("lightbox").hidden = false;
}

// ---------------------------------------------------------------------------
// prompt history (localStorage)
// ---------------------------------------------------------------------------
function loadHistory() {
  try {
    state.history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch (_) {
    state.history = [];
  }
  renderHistory();
}

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history.slice(0, MAX_HISTORY)));
}

function pushHistory(entry) {
  const duplicate = state.history.findIndex((item) => {
    return item.prompt === entry.prompt && item.tab === entry.tab && item.enhanced === entry.enhanced;
  });
  if (duplicate !== -1) state.history.splice(duplicate, 1);
  state.history.unshift(entry);
  state.history = state.history.slice(0, MAX_HISTORY);
  saveHistory();
  renderHistory();
}

function renderHistory() {
  const container = $("history");
  container.innerHTML = "";
  $("history-empty").hidden = state.history.length > 0;

  state.history.forEach((entry) => {
    const item = document.createElement("div");
    item.className = "history-item";
    item.title = "Click to load this prompt";

    const text = document.createElement("div");
    text.className = "text";
    text.textContent = entry.enhanced || entry.prompt;
    item.appendChild(text);

    const meta = document.createElement("div");
    meta.className = "meta";
    const bits = [entry.tab, entry.endpoint, entry.model];
    if (entry.engine) bits.push("enhanced by " + entry.engine);
    bits.push(new Date(entry.ts).toLocaleTimeString());
    meta.textContent = bits.join(" - ");
    item.appendChild(meta);

    item.addEventListener("click", function () {
      switchTab(entry.tab);
      if (entry.tab === "edit") $("edit-prompt-input").value = entry.prompt;
      else $("prompt-input").value = entry.prompt;
      if (entry.enhanced) {
        $("enhanced-output").value = entry.enhanced;
        $("enhance-panel").hidden = false;
      }
      toast("Prompt loaded.", "ok");
    });

    container.appendChild(item);
  });
}

// ---------------------------------------------------------------------------
// running requests
// ---------------------------------------------------------------------------
function startTimer(label) {
  state.startedAt = Date.now();
  clearInterval(state.timer);
  const tick = function () {
    const seconds = ((Date.now() - state.startedAt) / 1000).toFixed(0);
    setStatus("<span class=\"spin\"></span>" + label + " - " + seconds + "s elapsed");
  };
  tick();
  state.timer = setInterval(tick, 1000);
}

function stopTimer() {
  clearInterval(state.timer);
  state.timer = null;
}

function setBusy(busy) {
  state.busy = busy;
  $("submit-btn").disabled = busy;
  $("cancel-btn").disabled = !busy;
  $("copy-curl-btn").disabled = busy;
}

async function submit() {
  if (state.busy) return;

  const isEdit = state.tab === "edit";
  const params = readParams();
  const problem = validateParams(params);
  if (problem) {
    setStatus(problem, "error");
    toast(problem, "error");
    return;
  }

  const controller = new AbortController();
  state.controller = controller;
  setBusy(true);

  try {
    if (isEdit) {
      const prompt = $("edit-prompt-input").value.trim();
      if (!prompt) throw new Error("Write an edit instruction first.");
      if (state.refs.length === 0) throw new Error("Add at least one reference image.");

      const form = new FormData();
      form.append("prompt", prompt);
      form.append("endpoint", state.endpoint);
      Object.keys(params).forEach((key) => {
        const value = params[key];
        if (value !== null && value !== undefined) form.append(key, String(value));
      });
      const extras = enhancerExtras();
      Object.keys(extras).forEach((key) => {
        const value = extras[key];
        if (value !== null && value !== undefined) form.append(key, String(value));
      });
      state.refs.forEach((ref) => form.append("image", ref.file, ref.name));
      if (extras.enhance_engine) {
        form.append("reference_sizes", (await referenceSizes(state.refs)).join(","));
      }

      startTimer("Editing on " + state.endpoint + " (2048x2048 takes ~135s)");
      const data = await api("/api/edit", { method: "POST", form: form, signal: controller.signal });
      const request = data.request || {};
      addResults(data, {
        endpointLabel: request.endpoint_label,
        model: request.model,
        prompt: request.prompt,
        originalPrompt: request.original_prompt,
        enhanced: request.enhanced,
        params: request.params,
      });
      pushHistory({
        tab: "edit", prompt: prompt, endpoint: state.endpoint, model: request.model,
        enhanced: request.original_prompt ? request.prompt : null,
        engine: request.enhanced ? (request.enhanced.engine || request.enhanced.model) : null,
        ts: Date.now(),
      });
      const seconds = Number(data.wall_time_s || 0).toFixed(1);
      setStatus("Done in " + seconds + "s", "ok");
      toast("Edit finished in " + seconds + "s", "ok");
    } else {
      const prompt = $("prompt-input").value.trim();
      if (!prompt) throw new Error("Write a prompt first.");

      const body = Object.assign({ prompt: prompt, endpoint: state.endpoint }, params, enhancerExtras());
      startTimer("Generating on " + state.endpoint);
      const data = await api("/api/generate", { method: "POST", json: body, signal: controller.signal });
      const request = data.request || {};
      addResults(data, {
        endpointLabel: request.endpoint_label,
        model: request.model,
        prompt: request.prompt,
        originalPrompt: request.original_prompt,
        enhanced: request.enhanced,
        params: request.params,
      });
      pushHistory({
        tab: "generate", prompt: prompt, endpoint: state.endpoint, model: request.model,
        enhanced: request.original_prompt ? request.prompt : null,
        engine: request.enhanced ? (request.enhanced.engine || request.enhanced.model) : null,
        ts: Date.now(),
      });
      const seconds = Number(data.wall_time_s || 0).toFixed(1);
      setStatus("Done in " + seconds + "s", "ok");
      toast("Generated in " + seconds + "s", "ok");
    }
  } catch (error) {
    if (error.name === "AbortError") {
      setStatus("Cancelled.", "error");
      toast("Request cancelled.", "error");
    } else {
      setStatus(error.message, "error");
      toast(error.message, "error");
      console.error(error);
    }
  } finally {
    stopTimer();
    setBusy(false);
    state.controller = null;
  }
}

// ---------------------------------------------------------------------------
// curl export
// ---------------------------------------------------------------------------
function curlForCurrentForm() {
  const base = endpointBase();
  const params = readParams();
  const endpoint = (state.config.endpoints || []).filter((item) => item.name === state.endpoint)[0];
  const authLine = endpoint && endpoint.has_key ? "  -H 'Authorization: Bearer $IMAGE_API_KEY' \\" : "";

  if (state.tab === "edit") {
    const lines = [
      "curl -s " + base + "/images/edits \\",
      "  -F 'prompt=" + ($("edit-prompt-input").value.trim() || "...") + "' \\",
    ];
    if (params.model) lines.push("  -F 'model=" + params.model + "' \\");
    if (params.size) lines.push("  -F 'size=" + params.size + "' \\");
    if (params.width) lines.push("  -F 'width=" + params.width + "' \\");
    if (params.height) lines.push("  -F 'height=" + params.height + "' \\");
    if (params.num_inference_steps) lines.push("  -F 'num_inference_steps=" + params.num_inference_steps + "' \\");
    if (params.guidance_scale !== null) lines.push("  -F 'guidance_scale=" + params.guidance_scale + "' \\");
    if (params.seed !== null) lines.push("  -F 'seed=" + params.seed + "' \\");
    if (params.background) lines.push("  -F 'background=" + params.background + "' \\");
    if (params.negative_prompt) lines.push("  -F 'negative_prompt=" + params.negative_prompt + "' \\");
    state.refs.forEach((ref) => {
      lines.push("  -F 'image=@" + ref.name + ";type=" + (ref.file.type || "image/png") + "' \\");
    });
    if (authLine) lines.push(authLine);
    return lines.join("\n").replace(/\\$/, "");
  }

  const body = { model: params.model, prompt: $("prompt-input").value.trim() };
  if (params.size) body.size = params.size;
  if (params.width) { body.width = params.width; body.height = params.height; }
  if (params.num_inference_steps) body.num_inference_steps = params.num_inference_steps;
  if (params.guidance_scale !== null) body.guidance_scale = params.guidance_scale;
  if (params.seed !== null) body.seed = params.seed;
  if (params.n) body.n = params.n;
  if (params.output_format) body.output_format = params.output_format;
  if (params.background) body.background = params.background;
  if (params.negative_prompt) body.negative_prompt = params.negative_prompt;

  return [
    "curl -s " + base + "/images/generations \\",
    "  -H 'Content-Type: application/json' \\" + (authLine ? "\n" + authLine : " \\"),
    "  -d '" + JSON.stringify(body) + "'",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// health and model discovery
// ---------------------------------------------------------------------------
async function checkHealth() {
  const button = $("health-btn");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Checking...";
  try {
    const data = await api("/api/health");
    const results = data.results || [];
    results.forEach((result) => {
      const dot = document.querySelector("[data-dot=\"" + result.name + "\"]");
      if (dot) dot.className = "dot " + (result.ok ? "ok" : "bad");
    });
    const summary = results.map((item) => {
      return item.name + ": " + (item.ok ? "ok (" + item.latency_ms + "ms)" : "down (" + item.error + ")");
    }).join(" | ");
    const anyUp = results.some((item) => item.ok);
    setStatus(summary, anyUp ? "ok" : "error");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

async function loadModels() {
  const button = $("load-models-btn");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "...";
  try {
    const data = await api("/api/models?endpoint=" + encodeURIComponent(state.endpoint));
    const models = (data.models || []).map((item) => item.id).filter(Boolean);
    if (!models.length) {
      toast("The endpoint returned no models.", "error");
      return;
    }
    const datalist = $("model-list");
    datalist.innerHTML = "";
    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      datalist.appendChild(option);
    });
    if (models.indexOf($("model-input").value.trim()) === -1) $("model-input").value = models[0];
    toast("Loaded " + models.length + " model(s) from " + state.endpoint + ".", "ok");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

// ---------------------------------------------------------------------------
// ui wiring
// ---------------------------------------------------------------------------
function switchTab(tab) {
  state.tab = tab;
  $("tab-generate").classList.toggle("is-active", tab === "generate");
  $("tab-edit").classList.toggle("is-active", tab === "edit");
  $("panel-generate").hidden = tab !== "generate";
  $("panel-edit").hidden = tab !== "edit";
  $("submit-btn").textContent = tab === "edit" ? "Apply edit" : "Generate";
  if (state.enhanceTarget && state.enhanceTarget !== tab) $("enhance-panel").hidden = true;
  syncEngineOptions();
}

function toggleCustomSize() {
  const custom = $("size-select").value === "custom";
  $("custom-size").hidden = !custom;
  if (custom && !$("width-input").value) {
    $("width-input").value = 1024;
    $("height-input").value = 1024;
  }
}

function wire() {
  $("tab-generate").addEventListener("click", function () { switchTab("generate"); });
  $("tab-edit").addEventListener("click", function () { switchTab("edit"); });
  $("size-select").addEventListener("change", toggleCustomSize);
  $("enhance-engine-select").addEventListener("change", updateEngineNote);
  $("apply-ratio-btn").addEventListener("click", applySuggestedSize);
  $("apply-negative-btn").addEventListener("click", applyEnhanceNegative);

  $("background-select").addEventListener("change", function () {
    if ($("background-select").value === "transparent") {
      $("format-select").value = "png";
      toast("Transparent output needs png - format switched.", "ok");
    }
  });

  $("seed-random-btn").addEventListener("click", function () {
    $("seed-input").value = randomSeed();
  });
  $("seed-lock").addEventListener("change", function (event) {
    $("seed-random-btn").disabled = event.target.checked;
    if (!event.target.checked) $("seed-input").value = randomSeed();
  });

  $("enhance-btn").addEventListener("click", function () { enhance("generate"); });
  $("edit-enhance-btn").addEventListener("click", function () { enhance("edit"); });
  $("use-enhanced-btn").addEventListener("click", function () {
    const value = $("enhanced-output").value.trim();
    if (!value) return;
    if (state.enhanceTarget === "edit") $("edit-prompt-input").value = value;
    else $("prompt-input").value = value;
    toast("Enhanced prompt applied.", "ok");
  });
  $("revert-prompt-btn").addEventListener("click", function () {
    if (!state.enhanced) return;
    if (state.enhanced.tab === "edit") $("edit-prompt-input").value = state.enhanced.original;
    else $("prompt-input").value = state.enhanced.original;
    toast("Reverted to the original prompt.", "ok");
  });

  $("submit-btn").addEventListener("click", submit);
  $("cancel-btn").addEventListener("click", function () {
    if (state.controller) state.controller.abort();
  });
  $("copy-curl-btn").addEventListener("click", function () {
    copyText(curlForCurrentForm(), "curl copied");
  });
  $("health-btn").addEventListener("click", checkHealth);
  $("load-models-btn").addEventListener("click", loadModels);
  $("reload-env-btn").addEventListener("click", async function () {
    try {
      await loadConfig(true);
      toast(".env reloaded.", "ok");
    } catch (error) {
      toast(error.message, "error");
    }
  });

  $("browse-btn").addEventListener("click", function (event) {
    event.stopPropagation();
    $("file-input").click();
  });
  $("dropzone").addEventListener("click", function () { $("file-input").click(); });
  $("file-input").addEventListener("change", function (event) {
    addReferences(event.target.files);
    event.target.value = "";
  });
  ["dragenter", "dragover"].forEach((name) => {
    $("dropzone").addEventListener(name, function (event) {
      event.preventDefault();
      $("dropzone").classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    $("dropzone").addEventListener(name, function (event) {
      event.preventDefault();
      $("dropzone").classList.remove("is-over");
    });
  });
  $("dropzone").addEventListener("drop", function (event) {
    addReferences(event.dataTransfer.files);
  });
  $("clear-refs-btn").addEventListener("click", function () {
    state.refs = [];
    renderRefs();
  });

  $("download-all-btn").addEventListener("click", function () {
    state.results.forEach((result, index) => {
      setTimeout(function () { downloadResult(result); }, index * 350);
    });
  });
  $("clear-results-btn").addEventListener("click", function () {
    state.results.forEach((result) => {
      if (result.blob) URL.revokeObjectURL(result.src);
    });
    state.results = [];
    idbClear().catch(function (error) { console.warn("Could not clear IndexedDB", error); });
    renderResults();
    updateStorageNote();
  });
  $("clear-history-btn").addEventListener("click", function () {
    state.history = [];
    saveHistory();
    renderHistory();
  });

  $("lightbox").addEventListener("click", function () { $("lightbox").hidden = true; });

  document.addEventListener("keydown", function (event) {
    if (!event.ctrlKey && !event.metaKey) return;
    if (event.key !== "Enter") return;
    event.preventDefault();
    if (event.shiftKey) enhance(state.tab);
    else submit();
  });
}

async function main() {
  wire();
  loadHistory();
  renderRefs();
  try {
    await loadConfig(false);
    await restoreResults();
    setStatus("Ready. Ctrl+Enter runs, Ctrl+Shift+Enter enhances.", "ok");
    checkHealth();
  } catch (error) {
    setStatus("Cannot load /api/config: " + error.message, "error");
  }
}

document.addEventListener("DOMContentLoaded", main);
