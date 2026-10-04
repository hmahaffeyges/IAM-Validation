// "Run this check": runs verify_book.py in the browser with Pyodide (numpy, scipy, sympy), against the same files the script reads.
// Heavy checks (Planck chains, CAMB, the methylation chain) are not run here: the committed result, the rerun command and a Codespaces
// link are shown instead.
(function () {
  var PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  // For a local preview without internet access to the CDN, a same-site copy of Pyodide can be named: ?pyodide=/pyodide/
  // (only a path on this site is accepted).
  try {
    var q = new URLSearchParams(location.search).get("pyodide");
    if (q && /^\/[A-Za-z0-9_\-\/.]*\/$/.test(q) && q.indexOf("//") < 0) PYODIDE = q;
  } catch (e) {}
  var CODESPACES = "https://codespaces.new/hmahaffeyges/IAM-Validation";
  var root = document.documentElement.getAttribute("data-root") || "./";
  var meta = null, py = null, loading = null;

  function plain(t) {
    return String(t || "").replace(/\\times\s*10\^\{?([-+]?\d+)\}?/g, " × 10^$1").replace(/\\%/g, "%").replace(/\$/g, "")
      .replace(/\\,|\\!|~/g, "").replace(/\{,\}/g, ",").replace(/\\(lesssim|leq|le)\b/g, "≤").replace(/\\(gtrsim|geq|ge)\b/g, "≥");
  }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  var box, bar, msg;
  function progress(frac, text) {
    if (!box) {
      box = el("div", "iam-progress"); box.setAttribute("role", "status"); box.setAttribute("aria-live", "polite");
      msg = el("div"); bar = el("progress"); bar.max = 1; box.appendChild(msg); box.appendChild(bar); document.body.appendChild(box);
    }
    box.hidden = false; msg.textContent = text; if (frac == null) bar.removeAttribute("value"); else bar.value = frac;
  }
  function done(text) { if (!box) return; msg.textContent = text; bar.value = 1; setTimeout(function () { box.hidden = true; }, 2500); }

  function loadScript(src) {
    return new Promise(function (res, rej) { var s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
  }
  function getMeta() {
    if (meta) return Promise.resolve(meta);
    return fetch(root + "checks/checks.json").then(function (r) { return r.json(); }).then(function (m) { meta = m; return m; });
  }
  function ensurePython() {
    if (loading) return loading;
    loading = (async function () {
      progress(0.02, "Loading Python in this browser. The first time takes a while (the download is several tens of MB); later runs are fast.");
      await loadScript(PYODIDE + "pyodide.js");
      var p = await loadPyodide({ indexURL: PYODIDE });
      progress(0.35, "Loading numpy, scipy and sympy ...");
      await p.loadPackage(["numpy", "scipy", "sympy"]);
      var m = await getMeta();
      var files = m.data_files, n = files.length, i = 0;
      p.FS.mkdirTree("/iam");
      for (const rel of files) {
        i += 1; progress(0.6 + 0.35 * i / n, "Fetching the files the checks read (" + i + " of " + n + ") ...");
        var txt = await (await fetch(root + "checks/data/" + rel)).text();
        var dir = "/iam/" + rel.split("/").slice(0, -1).join("/");
        p.FS.mkdirTree(dir); p.FS.writeFile("/iam/" + rel, txt);
      }
      var src = await (await fetch(root + "checks/verify_book.py")).text();
      // verify_book.py lives at docs/book/ and finds the repository root two folders up: same layout here, rooted at /iam
      p.FS.mkdirTree("/iam/docs/book");
      p.FS.writeFile("/iam/docs/book/verify_book.py", src);
      p.runPython("import sys, os, json, dataclasses\nos.chdir('/iam')\nsys.path.insert(0, '/iam/docs/book')\nimport verify_book\nverify_book.set_data_root('/iam')");
      done("Python is ready.");
      py = p; return p;
    })();
    return loading;
  }

  function render(target, r) {
    var out = el("span", "iam-result");
    var head = el("span", r.passed ? "pass" : "fail", r.passed ? "PASS" : "FAIL");
    out.appendChild(head);
    out.appendChild(document.createTextNode("  " + r.label + "\n" +
      "book value:  " + (plain(r.printed) || "(algebra)") + "\n" +
      "recomputed:  " + r.recomputed + "\n" +
      "tolerance:   " + (r.tol ? r.tol : "half the last printed digit / exact") + "\n" +
      (r.title ? "what:        " + r.title : "")));
    target.appendChild(out);
    return out;
  }
  function renderHeavy(target, label, c) {
    var out = el("span", "iam-result");
    var r = c.committed || {};
    out.appendChild(el("span", r.passed ? "pass" : "fail", r.passed ? "PASS (committed result)" : "FAIL (committed result)"));
    out.appendChild(document.createTextNode("  " + label + "\n" +
      "book value:  " + plain(r.printed) + "\nrecomputed:  " + (r.recomputed || "") + "\n" +
      "This check reads an output of Planck chains, CAMB or the methylation chain; it does not run in the browser.\nrerun:       " + (c.rerun || "") + "\n"));
    var a = el("a", null, "Open in Codespaces"); a.href = CODESPACES; a.rel = "noopener"; out.appendChild(a);
    target.appendChild(out);
  }

  async function runOne(label, target) {
    var m = await getMeta();
    var c = m.checks[label];
    if (!c) { target.appendChild(el("span", "iam-result", "No check registered under " + label + " in this build.")); return null; }
    if (c.heavy) { renderHeavy(target, label, c); return c.committed; }
    var p = await ensurePython();
    var res = JSON.parse(p.runPython("json.dumps([dataclasses.asdict(r) for r in verify_book.run(label=" + JSON.stringify(label) + ")], default=str)"));
    render(target, res[0]);
    return res[0];
  }

  // With table = true (the front page's "Run every check" and each Part page's button) every check gets a row in a results list;
  // otherwise (a chapter's "Run all checks on this page") only the failures are printed under the totals.
  async function runMany(labels, target, title, table) {
    var m = await getMeta();
    await ensurePython();
    var pass = 0, fail = 0, heavy = 0, i = 0, rows = [];
    var summary = el("span", "iam-result"); target.appendChild(summary);
    for (const lab of labels) {
      i += 1;
      var c = m.checks[lab] || {};
      if (c.heavy) {
        heavy += 1; var k = c.committed || {};
        if (k.passed) pass += 1; else fail += 1;
        rows.push(row(lab, c, k, "committed"));
        continue;
      }
      progress(i / labels.length, title + ": " + i + " of " + labels.length);
      await new Promise(function (r) { setTimeout(r, 0); });
      try {
        var r = JSON.parse(py.runPython("json.dumps([dataclasses.asdict(r) for r in verify_book.run(label=" + JSON.stringify(lab) + ")], default=str)"))[0];
        if (r.passed) pass += 1; else { fail += 1; if (!table) render(target, r); }
        rows.push(row(lab, c, r, "browser"));
      } catch (e) {
        fail += 1;
        if (!table) target.appendChild(el("span", "iam-result", "ERROR " + lab + ": " + e));
        rows.push(row(lab, c, { passed: false, printed: (c.committed || {}).printed, recomputed: "ERROR: " + e }, "browser"));
      }
      summary.textContent = title + ": " + pass + " PASS, " + fail + " FAIL of " + i + " (" + heavy + " read from committed chain output)";
    }
    summary.textContent = title + ": " + pass + " PASS, " + fail + " FAIL of " + i + " (" + heavy + " read from committed chain output)";
    done(title + ": " + pass + " PASS, " + fail + " FAIL.");
    if (table) resultsTable(target, rows);
  }

  // ---- the results list: one row per check, filters by Part and by result, CSV download ----
  var ROMAN = ["Front matter", "I", "II", "III", "IV", "V", "VI", "VII", "Appendices"];
  function partName(n) { return ROMAN[n] != null ? ROMAN[n] : String(n); }
  function fmtTol(t) {
    if (t == null || t === "" || t === 0 || t === "0") return "half the last printed digit / exact";
    var x = Number(t); return isNaN(x) ? String(t) : String(Number(x.toPrecision(3)));
  }
  function row(lab, c, r, source) {
    return {
      label: lab, part: c.part, partName: partName(c.part), chapter: c.chapter || "", what: c.title || "",
      book: plain(r.printed) || "(algebra)", recomputed: r.recomputed == null ? "" : String(r.recomputed), tol: fmtTol(r.tol),
      result: r.passed ? "PASS" : "FAIL", source: source,
      href: c.page ? root + "book/" + c.page + "?check=" + encodeURIComponent(lab) + "#" + encodeURIComponent(c.anchor || "") : ""
    };
  }
  var SOURCE = { browser: "computed in your browser", committed: "read from committed chain output" };
  function resultsTable(target, rows) {
    var old = target.parentNode.querySelector(":scope > .iam-results"); if (old) old.remove();
    var wrap = el("section", "iam-results"); wrap.setAttribute("aria-label", "Results of every check");
    wrap.appendChild(el("h2", null, "Results"));
    var nb = rows.filter(function (x) { return x.source === "browser"; }).length;
    var p1 = el("p", "iam-results-note");
    p1.appendChild(el("b", null, "Computed in your browser (" + nb + "):"));
    p1.appendChild(document.createTextNode(" verify_book.py recomputed the value just now, in this page, and compared it with the number printed in the book."));
    var p2 = el("p", "iam-results-note");
    p2.appendChild(el("b", null, "Read from committed chain output (" + (rows.length - nb) + "):"));
    p2.appendChild(document.createTextNode(" the MCMC chains take days to run, and the CAMB runs, the methylation chain and the other pipelines these checks read " +
      "cannot run in a browser either, so these checks confirm that the book matches the committed chain files and outputs."));
    wrap.appendChild(p1); wrap.appendChild(p2);

    var bar = el("div", "iam-results-bar");
    function select(labelText, opts) {
      var lab = el("label"); lab.appendChild(document.createTextNode(labelText + " "));
      var s = document.createElement("select");
      opts.forEach(function (o) { var op = el("option", null, o[1]); op.value = o[0]; s.appendChild(op); });
      lab.appendChild(s); bar.appendChild(lab); return s;
    }
    var parts = []; rows.forEach(function (x) { if (parts.indexOf(x.part) < 0) parts.push(x.part); });
    parts.sort(function (a, b) { return a - b; });
    var fPart = select("Part", [["", "All"]].concat(parts.map(function (n) { return [String(n), partName(n)]; })));
    var fRes = select("Result", [["", "All"], ["PASS", "PASS"], ["FAIL", "FAIL"]]);
    var count = el("span", "iam-results-count"); bar.appendChild(count);
    var dl = el("button", "iam-btn secondary", "Download results (CSV)"); dl.type = "button"; bar.appendChild(dl);
    wrap.appendChild(bar);

    var scroller = el("div", "iam-results-scroll"); scroller.tabIndex = 0;
    scroller.setAttribute("role", "region"); scroller.setAttribute("aria-label", "Results table, scrollable");
    var tbl = el("table", "iam-results-table");
    var thead = el("thead"), hr = el("tr");
    ["Label", "Part", "Chapter", "What it computes", "Book value", "Recomputed", "Tolerance", "Result", "Source"].forEach(function (h) {
      var th = el("th", null, h); th.scope = "col"; hr.appendChild(th);
    });
    thead.appendChild(hr); tbl.appendChild(thead);
    var tbody = el("tbody"); tbl.appendChild(tbody);
    scroller.appendChild(tbl); wrap.appendChild(scroller);

    function draw() {
      var fp = fPart.value, fr = fRes.value, frag = document.createDocumentFragment(), n = 0;
      rows.forEach(function (x) {
        if (fp !== "" && String(x.part) !== fp) return;
        if (fr !== "" && x.result !== fr) return;
        n += 1;
        var tr = el("tr");
        var td = el("td", "lab");
        if (x.href) { var a = el("a", null, x.label); a.href = x.href; a.title = "Show this check in the book"; td.appendChild(a); } else td.textContent = x.label;
        tr.appendChild(td);
        tr.appendChild(el("td", null, x.partName));
        tr.appendChild(el("td", "chap", x.chapter));
        tr.appendChild(el("td", "what", x.what));
        tr.appendChild(el("td", "num", x.book));
        tr.appendChild(el("td", "num", x.recomputed));
        tr.appendChild(el("td", "num", x.tol));
        var rc = el("td"); rc.appendChild(el("span", x.result === "PASS" ? "pass" : "fail", x.result)); tr.appendChild(rc);
        tr.appendChild(el("td", "src", SOURCE[x.source]));
        frag.appendChild(tr);
      });
      tbody.textContent = ""; tbody.appendChild(frag);
      count.textContent = "Showing " + n + " of " + rows.length;
    }
    fPart.addEventListener("change", draw); fRes.addEventListener("change", draw);
    dl.addEventListener("click", function () {
      function q(v) { v = String(v == null ? "" : v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
      var lines = [["label", "part", "chapter", "what_it_computes", "book_value", "recomputed", "tolerance", "result", "source", "link"].join(",")];
      rows.forEach(function (x) {
        lines.push([x.label, x.partName, x.chapter, x.what, x.book, x.recomputed, x.tol, x.result, SOURCE[x.source],
                    x.href ? new URL(x.href, location.href).href : ""].map(q).join(","));
      });
      var blob = new Blob(["\ufeff" + lines.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
      var a = document.createElement("a"); a.href = URL.createObjectURL(blob);
      a.download = "iam_verify_book_results_" + new Date().toISOString().slice(0, 10) + ".csv";
      document.body.appendChild(a); a.click(); setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    });
    draw();
    target.parentNode.insertBefore(wrap, target.nextSibling);
  }

  // One marker per paragraph or display ("\u2713 3 checks"): it opens a panel listing those checks, each with its own Run button.
  function labelsOf(btn) { try { return JSON.parse(btn.getAttribute("data-labels")) || []; } catch (e) { return []; } }
  function plainValue(c) { return c && c.committed ? plain(c.committed.printed) : ""; }
  async function togglePanel(btn) {
    var open = btn.getAttribute("aria-expanded") === "true";
    var anchor = btn.closest(".ltx_p, .iam-checks-after") || btn.parentNode;
    var panel = btn._panel;
    if (open) { if (panel) panel.hidden = true; btn.setAttribute("aria-expanded", "false"); return; }
    if (!panel) {
      var m = await getMeta();
      panel = el("div", "iam-panel"); panel.setAttribute("role", "region");
      panel.setAttribute("aria-label", btn.textContent.replace("\u2713", "").trim());
      var labs = labelsOf(btn);
      labs.forEach(function (lab) {
        var c = m.checks[lab] || {};
        var row = el("div", "iam-row");
        var what = el("span", "iam-what");
        what.appendChild(el("span", "iam-title", c.title || lab));
        var v = plainValue(c);
        if (v) what.appendChild(el("span", "iam-book", " \u00b7 book: " + v));
        if (c.heavy) what.appendChild(el("span", "iam-book", " \u00b7 reads a chain or pipeline output"));
        row.appendChild(what);
        var run = el("button", "iam-run", "Run"); run.type = "button"; run.dataset.label = lab;
        run.setAttribute("aria-label", "Run check " + lab);
        row.appendChild(run);
        panel.appendChild(row);
      });
      if (labs.length > 1) {
        var all = el("button", "iam-run iam-runall", "Run all " + labs.length); all.type = "button";
        all.dataset.labels = JSON.stringify(labs);
        var foot = el("div", "iam-row iam-foot"); foot.appendChild(all); panel.appendChild(foot);
      }
      anchor.parentNode.insertBefore(panel, anchor.nextSibling);
      btn._panel = panel;
    }
    panel.hidden = false; btn.setAttribute("aria-expanded", "true");
  }

  // A results-list link (?check=<label>#<paragraph>) opens that paragraph's panel and marks the check's row.
  function openFromLink() {
    var lab = null;
    try { lab = new URLSearchParams(location.search).get("check"); } catch (e) {}
    if (!lab) return;
    var marks = document.querySelectorAll(".iam-mark");
    for (var i = 0; i < marks.length; i++) {
      if (labelsOf(marks[i]).indexOf(lab) >= 0) {
        var mk = marks[i];
        togglePanel(mk).then(function () {
          var r = mk._panel && mk._panel.querySelector('.iam-run[data-label="' + lab.replace(/"/g, '\\"') + '"]');
          if (r) { r.parentNode.classList.add("iam-row-target"); r.parentNode.scrollIntoView({ block: "center" }); }
        });
        return;
      }
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", openFromLink); else openFromLink();

  document.addEventListener("click", function (ev) {
    var mk = ev.target.closest(".iam-mark");
    if (mk) { ev.preventDefault(); togglePanel(mk); return; }
    var pg = ev.target.closest(".iam-runpage");
    if (pg) {
      ev.preventDefault(); pg.disabled = true;
      var labs = [];
      document.querySelectorAll(".iam-mark").forEach(function (b) { labelsOf(b).forEach(function (l) { if (labs.indexOf(l) < 0) labs.push(l); }); });
      runMany(labs, pg.parentNode, "This page").catch(function (e) { pg.parentNode.appendChild(el("span", "iam-result", "Could not run: " + e)); })
        .finally(function () { pg.disabled = false; });
      return;
    }
    var b = ev.target.closest(".iam-run");
    if (!b) return;
    ev.preventDefault();
    b.disabled = true;
    var holder = b.parentNode;
    var p;
    if (b.dataset.label) {
      var prev = holder.querySelector(".iam-result"); if (prev && holder.classList.contains("iam-row")) prev.remove();
      p = runOne(b.dataset.label, holder);
    }
    else if (b.dataset.labels) p = (async function () {        // a panel's "Run all": each row shows its own result
      var rows = b.closest(".iam-panel").querySelectorAll(".iam-row .iam-run[data-label]");
      for (var i = 0; i < rows.length; i++) {
        var row = rows[i].parentNode, old = row.querySelector(".iam-result"); if (old) old.remove();
        await runOne(rows[i].dataset.label, row);
        await new Promise(function (r) { setTimeout(r, 0); });
      }
    })();
    else if (b.dataset.part != null) p = getMeta().then(function (m) {
      var labs = Object.keys(m.checks).filter(function (k) { return b.dataset.part === "all" || String(m.checks[k].part) === b.dataset.part; });
      var prev = holder.querySelector(".iam-result"); if (prev) prev.remove();
      return runMany(labs, holder, b.dataset.part === "all" ? "Every check" : "Part checks", true);
    });
    p.catch(function (e) { holder.appendChild(el("span", "iam-result", "Could not run: " + e)); }).finally(function () { b.disabled = false; });
  });
})();
