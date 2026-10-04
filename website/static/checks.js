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
      p.FS.writeFile("/iam/verify_book.py", src);
      p.runPython("import sys, os, json, dataclasses\nos.chdir('/iam')\nsys.path.insert(0, '/iam')\nimport verify_book\nverify_book.set_data_root('/iam')");
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
      "book value:  " + (r.printed || "(algebra)") + "\n" +
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
      "book value:  " + (r.printed || "") + "\nrecomputed:  " + (r.recomputed || "") + "\n" +
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

  async function runMany(labels, target, title) {
    var m = await getMeta();
    await ensurePython();
    var pass = 0, fail = 0, heavy = 0, i = 0;
    var summary = el("span", "iam-result"); target.appendChild(summary);
    for (const lab of labels) {
      i += 1;
      var c = m.checks[lab];
      if (c && c.heavy) { heavy += 1; if (c.committed && c.committed.passed) pass += 1; else fail += 1; continue; }
      progress(i / labels.length, title + ": " + i + " of " + labels.length);
      await new Promise(function (r) { setTimeout(r, 0); });
      try {
        var r = JSON.parse(py.runPython("json.dumps([dataclasses.asdict(r) for r in verify_book.run(label=" + JSON.stringify(lab) + ")], default=str)"))[0];
        if (r.passed) pass += 1; else { fail += 1; render(target, r); }
      } catch (e) { fail += 1; target.appendChild(el("span", "iam-result", "ERROR " + lab + ": " + e)); }
      summary.textContent = title + ": " + pass + " PASS, " + fail + " FAIL of " + i + " (" + heavy + " heavy, committed results shown)";
    }
    done(title + ": " + pass + " PASS, " + fail + " FAIL.");
  }

  document.addEventListener("click", function (ev) {
    var b = ev.target.closest(".iam-run");
    if (!b) return;
    ev.preventDefault();
    b.disabled = true;
    var holder = b.parentNode;
    var p;
    if (b.dataset.label) p = runOne(b.dataset.label, holder);
    else if (b.dataset.part != null) p = getMeta().then(function (m) {
      var labs = Object.keys(m.checks).filter(function (k) { return b.dataset.part === "all" || String(m.checks[k].part) === b.dataset.part; });
      return runMany(labs, holder, b.dataset.part === "all" ? "Every check" : "Part checks");
    });
    p.catch(function (e) { holder.appendChild(el("span", "iam-result", "Could not run: " + e)); }).finally(function () { b.disabled = false; });
  });
})();
