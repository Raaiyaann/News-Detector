// ===== CONFIG =====
const API_ENDPOINT =
  location.hostname.includes("localhost") || location.hostname.includes("127.0.0.1")
    ? "http://127.0.0.1:5000/predict"
    : "https://your-backend-api.com/predict"; // ganti saat deploy

// ===== HELPERS =====
const $ = (q) => document.querySelector(q);

function isValidUrl(str) {
  try {
    const u = new URL(str);
    return ["http:", "https:"].includes(u.protocol);
  } catch {
    return false;
  }
}

function setBusy(formEl, busy) {
  const btn = formEl.querySelector('button[type="submit"]');
  formEl.setAttribute("aria-busy", busy ? "true" : "false");
  btn.disabled = !!busy;
  btn.textContent = busy ? "Memeriksa…" : "Check";
}

function classifyBadge(prediction) {
  const label = String(prediction).toLowerCase();
  const hoaxy = /(hoax|fake|disinfo|misinfo|false|spam|\b1\b)/i.test(label);
  return hoaxy ? "result-true" : "result-false";
}

function fmtPct(x) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}

function renderResult(data) {
  const result = $("#result");
  result.className = "result"; // reset
  result.innerHTML = "";

  if (!data || !data.prediction) {
    result.innerHTML = "❓ Tidak bisa menentukan hasil. Coba lagi.";
    return;
  }

  const badgeClass = classifyBadge(data.prediction);
  result.classList.add(badgeClass);

  const source = data.source ? `<span class="badge">${data.source}</span>` : "";
  const title = data.title ? `<div style="font-weight:700;margin:.25rem 0;">${data.title}</div>` : "";
  const conf = data.confidence != null ? fmtPct(data.confidence) : null;

  const extra = `
    <div style="margin-top:.5rem;font-weight:500;color:#374151">
      ${source}
      ${data.fetched_at ? `<span class="badge" style="margin-left:.35rem;">${new Date(data.fetched_at).toLocaleString()}</span>` : ""}
      ${data.words ? `<span class="badge" style="margin-left:.35rem;">${data.words} kata</span>` : ""}
    </div>
    ${data.excerpt ? `<details style="margin-top:.5rem;"><summary>Lihat ringkasan konten</summary><p style="margin:.5rem 0 0;">${data.excerpt}</p></details>` : ""}
    ${data.probabilities ? `
      <details style="margin-top:.5rem;">
        <summary>Probabilitas per kelas</summary>
        <div style="margin-top:.4rem;font-size:.95rem;">
          ${Object.entries(data.probabilities).map(([k,v]) => `<div style="display:flex;justify-content:space-between;"><span>${k}</span><strong>${fmtPct(v)}</strong></div>`).join("")}
        </div>
      </details>` : ""}
  `;

  result.innerHTML =
    (badgeClass === "result-true"
      ? `⚠️ <strong>Berpotensi HOAKS</strong>${conf ? ` • Keyakinan model: ${conf}` : ""}`
      : `✅ <strong>Tampak Kredibel</strong>${conf ? ` • Keyakinan model: ${conf}` : ""}`) +
    title +
    extra;
}

function renderError(msg) {
  const result = $("#result");
  result.className = "result";
  result.innerHTML = `❌ ${msg || "Terjadi kesalahan koneksi ke server."}`;
}

// ===== RESTORE LAST RESULT (opsional) =====
document.addEventListener("DOMContentLoaded", () => {
  const last = sessionStorage.getItem("lastResult");
  if (last) {
    $("#result").classList.remove("hidden");
    renderResult(JSON.parse(last));
  }
});

// ===== SUBMIT HANDLER =====
$("#predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const form = e.currentTarget;
  const newsUrl = $("#newsUrl").value.trim();
  const resultDiv = $("#result");
  resultDiv.classList.remove("hidden");
  resultDiv.className = "result";
  resultDiv.textContent = "⏳ Memeriksa…";

  if (!isValidUrl(newsUrl)) {
    renderError("URL tidak valid. Pastikan mulai dengan http(s) dan merupakan tautan artikel.");
    return;
  }

  // Abort after 30s
  const controller = new AbortController();
  const to = setTimeout(() => controller.abort("timeout"), 30000);

  try {
    setBusy(form, true);

    const res = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: newsUrl }),
      signal: controller.signal,
    });

    clearTimeout(to);

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // Simpan untuk restore
    sessionStorage.setItem("lastResult", JSON.stringify(data));

    renderResult(data);
  } catch (err) {
    if (err.name === "AbortError") {
      renderError("Request timeout. Coba lagi.");
    } else {
      renderError(err?.message || "Gagal memproses permintaan.");
    }
  } finally {
    setBusy(form, false);
  }
});
