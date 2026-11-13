/* The card as a stepping figure: the two 6 x 10 panels of Tensix cores, the DRAM
 * columns, the four X280 clusters and the host, drawn from a JSON block in the page.
 *
 *   <figure class="km-grid">
 *     <script type="application/json">
 *       {"steps": [{"label": "...", "lit": ["cores:0-7"],
 *                   "arrows": [{"from": "host", "to": "x280:0"}], "marks": {"core:3": "7"}}]}
 *     </script>
 *   </figure>
 *
 * Places: host, x280:K (K = 0 to 3), core:N (N = 0 to 119, row by row), cores:A-B,
 * dram:C (C = 0 or 9, the column), dram:C:R (one DRAM tile of it, R = 0 to 9).
 * An arrow has a "kind": "command" (solid), "data" (dashed), "reply" (dotted),
 * "broadcast", "routed" or "error"; each takes the gem of its kind, set in
 * assets/tutorial.css from assets/gem.css. No library, no network.
 */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var CELL_W = 26, CELL_H = 22, PITCH_X = 30, PITCH_Y = 26, TOP = 50;

  // x of each grid column (NoC0 column numbers 0 to 16; 7 and 10 are fused off).
  var COLUMN_X = {};
  (function () {
    var x = 0, order = [0, 1, 2, 3, 4, 5, 6, "gap", 8, 9, "gap", 11, 12, 13, 14, 15, 16];
    order.forEach(function (c) {
      if (c === "gap") { x += 10; return; }
      COLUMN_X[c] = x;
      x += c === 8 ? 62 : PITCH_X;
    });
    COLUMN_X.width = x - (PITCH_X - CELL_W);
  })();
  var TENSIX_COLUMNS = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16];
  var X280_ROW = { 0: 1, 2: 3, 3: 5, 1: 7 }; // cluster -> row of the grid (X280 0 at NoC0 (8, 3))
  var X280_X = COLUMN_X[8] + 2, X280_W = 54;
  var HEIGHT = TOP + 10 * PITCH_Y + 22;

  function el(name, attrs, parent) {
    var node = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(node);
    return node;
  }

  // A place as a rectangle {x, y, w, h}.
  function place(name) {
    var m;
    if (name === "host") return { x: X280_X - 4, y: 2, w: X280_W + 8, h: 30 };
    if ((m = /^x280:(\d)$/.exec(name))) {
      var row = X280_ROW[+m[1]];
      if (row === undefined) throw new Error("no X280 cluster " + m[1]);
      return { x: X280_X, y: TOP + row * PITCH_Y - 2, w: X280_W, h: CELL_H + 4 };
    }
    if ((m = /^core:(\d+)$/.exec(name))) {
      var n = +m[1];
      if (n > 119) throw new Error("no core " + n);
      return { x: COLUMN_X[TENSIX_COLUMNS[n % 12]], y: TOP + Math.floor(n / 12) * PITCH_Y, w: CELL_W, h: CELL_H };
    }
    if ((m = /^dram:(\d+)(?::(\d))?$/.exec(name))) {
      var c = +m[1];
      if (c !== 0 && c !== 9) throw new Error("DRAM is in columns 0 and 9");
      if (m[2] !== undefined) return { x: COLUMN_X[c], y: TOP + +m[2] * PITCH_Y, w: CELL_W, h: CELL_H };
      return { x: COLUMN_X[c], y: TOP, w: CELL_W, h: 9 * PITCH_Y + CELL_H };
    }
    throw new Error("unknown place " + name);
  }

  function expand(names) {
    var out = [];
    (names || []).forEach(function (name) {
      var m = /^cores:(\d+)-(\d+)$/.exec(name);
      if (m) for (var n = +m[1]; n <= +m[2]; n++) out.push("core:" + n);
      else out.push(name);
    });
    return out;
  }

  function center(r) { return { x: r.x + r.w / 2, y: r.y + r.h / 2 }; }

  // The point where the line from `from` to the centre of rectangle r leaves r.
  function edge(r, from) {
    var c = center(r), dx = from.x - c.x, dy = from.y - c.y;
    if (dx === 0 && dy === 0) return c;
    var t = Math.min(dx !== 0 ? r.w / 2 / Math.abs(dx) : Infinity, dy !== 0 ? r.h / 2 / Math.abs(dy) : Infinity);
    return { x: c.x + dx * t, y: c.y + dy * t };
  }

  function draw(svg, step) {
    var lit = {};
    expand(step.lit).forEach(function (n) { lit[n] = true; });
    var g = el("g", {}, svg);
    // the host
    var h = place("host");
    el("rect", { x: h.x, y: h.y, width: h.w, height: h.h, rx: 4, "class": "km-host" + (lit.host ? " km-lit" : "") }, g);
    var t = el("text", { x: h.x + h.w / 2, y: h.y + 19, "text-anchor": "middle", "class": "km-text" }, g);
    t.textContent = "host";
    // the DRAM columns
    [0, 9].forEach(function (c) {
      var d = place("dram:" + c);
      for (var row = 0; row < 10; row++) {
        el("rect", { x: d.x, y: TOP + row * PITCH_Y, width: CELL_W, height: CELL_H, rx: 2,
                     "class": "km-dram" + (lit["dram:" + c] ? " km-lit" : "") }, g);
      }
    });
    // the Tensix cores
    for (var n = 0; n < 120; n++) {
      var r = place("core:" + n);
      el("rect", { x: r.x, y: r.y, width: r.w, height: r.h, rx: 2,
                   "class": "km-core" + (lit["core:" + n] ? " km-lit" : "") }, g);
    }
    // the X280 clusters
    [0, 1, 2, 3].forEach(function (k) {
      var x = place("x280:" + k);
      el("rect", { x: x.x, y: x.y, width: x.w, height: x.h, rx: 4,
                   "class": "km-x280" + (lit["x280:" + k] ? " km-lit" : "") }, g);
      var label = el("text", { x: x.x + x.w / 2, y: x.y + 16, "text-anchor": "middle", "class": "km-text km-small" }, g);
      label.textContent = "X280 " + k;
    });
    // marks: a short text on a place
    Object.keys(step.marks || {}).forEach(function (name) {
      var c = center(place(name));
      var m = el("text", { x: c.x, y: c.y + 4, "text-anchor": "middle", "class": "km-text km-mark" }, g);
      m.textContent = step.marks[name];
    });
    // arrows
    (step.arrows || []).forEach(function (a) {
      var from = place(a.from), to = place(a.to);
      var p = edge(from, center(to)), q = edge(to, center(from));
      var kind = a.kind || "command";
      el("line", { x1: p.x, y1: p.y, x2: q.x, y2: q.y, "class": "km-arrow km-" + kind,
                   "marker-end": "url(#km-head-" + kind + ")" }, g);
    });
  }

  function captions(svg) {
    var y = TOP + 10 * PITCH_Y + 14;
    [["DRAM", COLUMN_X[0], "start"], ["DRAM", COLUMN_X[9] + CELL_W / 2, "middle"],
     ["Tensix panel 0", (COLUMN_X[1] + COLUMN_X[6] + CELL_W) / 2, "middle"],
     ["Tensix panel 1", (COLUMN_X[11] + COLUMN_X[16] + CELL_W) / 2, "middle"]].forEach(function (c) {
      var t = el("text", { x: c[1], y: y, "text-anchor": c[2], "class": "km-text km-small km-muted" }, svg);
      t.textContent = c[0];
    });
  }

  function markers(svg) {
    var defs = el("defs", {}, svg);
    ["command", "data", "reply", "broadcast", "routed", "error"].forEach(function (kind) {
      var m = el("marker", { id: "km-head-" + kind, viewBox: "0 0 10 10", refX: 9, refY: 5,
                             markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", "class": "km-head km-" + kind }, m);
    });
  }

  var counter = 0;

  function build(figure) {
    var script = figure.querySelector("script[type='application/json']");
    if (!script || figure.dataset.ready) return;
    var spec;
    try { spec = JSON.parse(script.textContent); } catch (e) {
      figure.insertAdjacentHTML("beforeend", "<p>The figure's data is not valid JSON: " + e.message + "</p>");
      return;
    }
    figure.dataset.ready = "1";
    var steps = spec.steps || [];
    var id = "km-grid-" + (counter++);
    var svg = el("svg", { viewBox: "-2 -2 " + (COLUMN_X.width + 4) + " " + (HEIGHT + 4), role: "img",
                          "aria-labelledby": id + "-label", "class": "km-svg" });
    svg.id = id;
    markers(svg);
    var body = el("g", {}, svg);
    figure.insertBefore(svg, script);

    var bar = document.createElement("div");
    bar.className = "km-bar";
    var prev = document.createElement("button");
    var next = document.createElement("button");
    var count = document.createElement("span");
    prev.type = next.type = "button";
    prev.textContent = "← Previous";
    next.textContent = "Next →";
    count.className = "km-count";
    bar.appendChild(prev); bar.appendChild(count); bar.appendChild(next);
    var label = document.createElement("p");
    label.className = "km-label";
    label.id = id + "-label";
    label.setAttribute("aria-live", "polite");
    figure.insertBefore(bar, script);
    figure.insertBefore(label, script);

    var at = 0;
    function show(i) {
      at = Math.max(0, Math.min(steps.length - 1, i));
      while (body.firstChild) body.removeChild(body.firstChild);
      draw(body, steps[at]);
      captions(body);
      label.textContent = steps[at].label || "";
      count.textContent = (at + 1) + " / " + steps.length;
      prev.disabled = at === 0;
      next.disabled = at === steps.length - 1;
    }
    prev.addEventListener("click", function () { show(at - 1); });
    next.addEventListener("click", function () { show(at + 1); });
    figure.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { show(at - 1); e.preventDefault(); }
      if (e.key === "ArrowRight") { show(at + 1); e.preventDefault(); }
    });
    figure.tabIndex = 0;
    if (steps.length) show(+figure.dataset.start || 0);
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("figure.km-grid"), build);
  }

  if (typeof document$ !== "undefined") document$.subscribe(init);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
