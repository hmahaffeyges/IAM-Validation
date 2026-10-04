// Light/dark toggle (the reader's choice is kept in this browser only). Default: dark on the front page and Part pages, light on chapters.
(function () {
  var key = "iam-theme";
  var stored = null;
  try { stored = localStorage.getItem(key); } catch (e) {}
  var dflt = document.documentElement.getAttribute("data-default-theme") || "light";
  document.documentElement.setAttribute("data-theme", stored || dflt);
  window.addEventListener("DOMContentLoaded", function () {
    var b = document.getElementById("iam-toggle");
    if (!b) return;
    function label() { b.textContent = document.documentElement.getAttribute("data-theme") === "dark" ? "Light" : "Dark"; }
    label();
    b.addEventListener("click", function () {
      var t = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", t);
      try { localStorage.setItem(key, t); } catch (e) {}
      label();
    });
  });
})();
