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

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function fmtPct(x) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}

function renderError(msg) {
  const result = $("#result");
  result.className = "result result-true"; // pakai palet merah utk error
  result.innerHTML = `❌ ${msg || "Terjadi kesalahan koneksi ke server."}`;
}

// ===== RENDER =====
function renderResult(data) {
  const result = $("#result");
  result.className = "result";

  if (!data) {
    result.innerHTML = "❓ Tidak bisa menentukan hasil. Coba lagi.";
    return;
  }

  const isHoax = !!data.is_hoax;

  // --- Ambil probabilitas hoaks yang valid ---
  // 1) dari field khusus backend
  let hoaxProb = (typeof data.hoax_probability === "number") ? data.hoax_probability : null;

  // 2) fallback: tebak dari daftar probabilities (cari key '1' atau 'hoax')
  if (hoaxProb == null && data.probabilities) {
    if (Object.prototype.hasOwnProperty.call(data.probabilities, "1")) {
      hoaxProb = Number(data.probabilities["1"]);
    } else if (Object.prototype.hasOwnProperty.call(data.probabilities, "hoax")) {
      hoaxProb = Number(data.probabilities["hoax"]);
    }
  }

  if (typeof hoaxProb === "number" && !Number.isNaN(hoaxProb)) {
    hoaxProb = clamp01(hoaxProb);
  } else {
    hoaxProb = null;
  }

  // 3) fallback terakhir ke confidence (prob label terprediksi)
  let displayConfidence = null;
  if (hoaxProb != null) {
    displayConfidence = isHoax ? hoaxProb : (1 - hoaxProb);
  } else if (typeof data.confidence === "number") {
    displayConfidence = clamp01(data.confidence);
  }

  result.classList.add(isHoax ? "result-true" : "result-false");

  const header =
    (isHoax ? `⚠️ <strong>Berpotensi HOAKS</strong>` : `✅ <strong>Tampak Kredibel</strong>`) +
    (displayConfidence != null ? ` • Keyakinan model: ${fmtPct(displayConfidence)}` : "");

  const title = data.title ? `<div style="font-weight:700;margin:.25rem 0;">${data.title}</div>` : "";
  const meta =
    `<div style="margin-top:.5rem;font-weight:500;color:#374151;display:flex;gap:.35rem;flex-wrap:wrap;">
       ${data.source ? `<span class="badge">${data.source}</span>` : ""}
       ${data.published ? `<span class="badge">${new Date(data.published).toLocaleString()}</span>` : ""}
       ${data.words ? `<span class="badge">${data.words} kata</span>` : ""}
     </div>`;

  const excerpt = data.excerpt
    ? `<details style="margin-top:.5rem;">
         <summary>Lihat ringkasan konten</summary>
         <p style="margin:.5rem 0 0;">${data.excerpt}</p>
       </details>` : "";

  let probs = "";
  if (data.probabilities) {
    const rows = Object.entries(data.probabilities)
      .sort((a,b) => String(a[0]).localeCompare(String(b[0])))
      .map(([k,v]) => `<div style="display:flex;justify-content:space-between;">
                         <span>${k}</span><strong>${fmtPct(Number(v))}</strong>
                       </div>`).join("");
    probs = `<details style="margin-top:.5rem;">
               <summary>Probabilitas per kelas</summary>
               <div style="margin-top:.4rem;font-size:.95rem;">${rows}</div>
             </details>`;
  }

  const note = data.note ? `<div class="tips" style="margin-top:.6rem;">${data.note}</div>` : "";

  result.innerHTML = `${header}${title}${meta}${excerpt}${probs}${note}`;
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

  // Abort after 120s
  const controller = new AbortController();
  const to = setTimeout(() => controller.abort("timeout"), 120000);

  try {
    setBusy(form, true);

    const res = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: newsUrl }),
      signal: controller.signal,
    });

    clearTimeout(to);

    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(text || `HTTP ${res.status}`);
    }

    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }

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
