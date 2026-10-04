// Reading theme: Paper (default), Dark or Sepia. On a first visit the page follows the device's light/dark preference (Dark if the
// device prefers dark, otherwise Paper); once the reader picks a theme, that choice is kept in this browser and used on every page.
(function () {
  var KEY = "iam-theme", THEMES = ["paper", "dark", "sepia"];
  function stored() { try { var t = localStorage.getItem(KEY); return THEMES.indexOf(t) >= 0 ? t : null; } catch (e) { return null; } }
  function device() { try { return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "paper"; } catch (e) { return "paper"; } }
  function apply(t) { document.documentElement.setAttribute("data-theme", t); }
  apply(stored() || device());
  // follow the device while the reader has not chosen
  try {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { if (!stored()) { apply(device()); mark(); } });
  } catch (e) {}
  function mark() {
    var cur = document.documentElement.getAttribute("data-theme");
    var bs = document.querySelectorAll(".iam-themes button");
    for (var i = 0; i < bs.length; i++) bs[i].setAttribute("aria-pressed", bs[i].getAttribute("data-theme") === cur ? "true" : "false");
  }
  window.addEventListener("DOMContentLoaded", function () {
    mark();
    var bs = document.querySelectorAll(".iam-themes button");
    for (var i = 0; i < bs.length; i++) bs[i].addEventListener("click", function () {
      var t = this.getAttribute("data-theme");
      apply(t);
      try { localStorage.setItem(KEY, t); } catch (e) {}
      mark();
    });
  });
})();
