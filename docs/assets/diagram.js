/* The tutorial's small diagrams, drawn from a JSON block in the page. Three kinds, each
 * answering one question:
 *
 *   anatomy  what do the parts of this line do?       the code with a name on each part
 *   boxes    where does this thing live?              boxes and arrows, placed by hand
 *   flow     what happens in what order?              steps one under the other
 *
 *   <figure class="km-diagram">
 *     <script type="application/json">{"kind": "anatomy", "parts": [...]}</script>
 *     <figcaption>One sentence that says what the picture shows.</figcaption>
 *   </figure>
 *
 * anatomy: {"parts": [{"t": "run.submit(", "n": "send a command"}, {"t": "0"}, ...]}
 *   the "t" strings, joined, are the line of code; a part with "n" gets a name below.
 * boxes:   {"boxes": [{"id", "x", "y", "w", "h", "t", "gem", "dashed", "here"}],
 *           "arrows": [{"from", "to", "t", "kind"}]}   (x, y, w, h in pixels)
 * flow:    {"steps": [{"t", "gem", "here"}], "back": {"from": 3, "to": 1, "t": "again"}}
 *
 * A gem is one of amber (a Tensix core), sapphire (an X280 core), aquamarine, taaffeite,
 * coral, sage; no gem means a stone box. "here" rings a box in iolite: where the reader
 * is. The colours are the variables of assets/gem.css, applied by assets/tutorial.css.
 * No library, no network; without the script the caption alone remains.
 */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var LINE = 17; // pixels between two lines of a label

  function el(name, attrs, parent) {
    var node = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, x, y, content, cls, anchor) {
    var lines = String(content).split("\n");
    var node = el("text", { x: x, y: y, class: "kd-text " + (cls || ""), "text-anchor": anchor || "start" }, parent);
    lines.forEach(function (line, i) {
      var span = el("tspan", { x: x, dy: i === 0 ? 0 : LINE }, node);
      span.textContent = line;
    });
    return node;
  }

  function width(node) { return node.getComputedTextLength ? node.getBBox().width : 0; }

  function svgFor(figure, w, h) {
    var svg = el("svg", {
      class: "km-svg kd-svg",
      viewBox: "-4 -4 " + (w + 8) + " " + (h + 8),
      role: "img",
      "aria-label": (figure.querySelector("figcaption") || {}).textContent || "diagram"
    });
    svg.style.maxWidth = (w + 8) + "px";
    var defs = el("defs", {}, svg);
    ["flow", "data", "reply", "broadcast", "routed", "error", "command"].forEach(function (kind) {
      var marker = el("marker", {
        id: "kd-head-" + kind, viewBox: "0 0 10 10", refX: 9, refY: 5,
        markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"
      }, defs);
      el("path", { d: "M0 0 L10 5 L0 10 z", class: "kd-head kd-" + kind }, marker);
    });
    return svg;
  }

  function put(figure, svg) {
    var old = figure.querySelector("svg");
    if (old) figure.removeChild(old);
    figure.insertBefore(svg, figure.firstChild);
  }

  // ---- anatomy ----
  function anatomy(figure, spec) {
    var scratch = svgFor(figure, 10, 10);
    figure.insertBefore(scratch, figure.firstChild);
    var CODE_Y = 22, CHAR = 0;
    var probe = text(scratch, 0, CODE_Y, "MMMMMMMMMM", "kd-code");
    CHAR = width(probe) / 10;
    var measured = spec.parts.map(function (part) {
      if (!part.n) return 0;
      return width(text(scratch, 0, CODE_Y, part.n, "kd-note")) + 12;
    });
    figure.removeChild(scratch);

    var x = 0, parts = [];
    spec.parts.forEach(function (part, index) {
      parts.push({ t: part.t, n: part.n, gem: part.gem, x: x, w: part.t.length * CHAR, label: measured[index] });
      x += part.t.length * CHAR;
    });
    var total = x;

    // Labels: a staircase. The leftmost part gets the deepest row and every label starts
    // at its connector and runs to the right, so no connector crosses a label.
    var labels = [];
    var noted = parts.filter(function (p) { return p.n; });
    noted.forEach(function (part, index) {
      var cx = part.x + Math.min(part.w / 2, 24);
      labels.push({
        part: part, lines: part.n.split("\n"), w: part.label, left: Math.max(0, cx - 6), cx: cx,
        row: noted.length - 1 - index
      });
    });
    var rowHeight = [];
    labels.forEach(function (l) {
      rowHeight[l.row] = Math.max(rowHeight[l.row] || 0, l.lines.length * LINE + 14);
    });
    var rowY = [], y = 52;
    for (var r = 0; r < rowHeight.length; r++) { rowY[r] = y; y += rowHeight[r] || 0; }
    var overall = Math.max(total, Math.max.apply(null, labels.map(function (l) { return l.left + l.w; }).concat([0])));

    var svg = svgFor(figure, overall, y + 4);
    var code = text(svg, 0, CODE_Y, "", "kd-code");
    parts.forEach(function (part) {
      var span = el("tspan", { class: part.n ? "kd-named" : "" }, code);
      span.textContent = part.t;
    });
    code.setAttribute("xml:space", "preserve");
    code.style.whiteSpace = "pre";
    labels.forEach(function (l) {
      var part = l.part, cls = part.gem ? " g-" + part.gem : "";
      el("path", {
        d: "M" + part.x + " 30 V34 H" + (part.x + part.w) + " V30",
        class: "kd-bracket" + cls
      }, svg);
      el("path", { d: "M" + l.cx + " 34 V" + (rowY[l.row] - 4), class: "kd-bracket" + cls }, svg);
      el("rect", {
        x: l.left, y: rowY[l.row] - 2, width: l.w, height: l.lines.length * LINE + 6, rx: 8,
        class: "kd-box" + cls
      }, svg);
      text(svg, l.left + 6, rowY[l.row] + 12, l.part.n, "kd-note", "start");
    });
    put(figure, svg);
  }

  // ---- boxes ----
  function edge(box, toward) {
    var cx = box.x + box.w / 2, cy = box.y + box.h / 2;
    var dx = toward.x - cx, dy = toward.y - cy;
    if (dx === 0 && dy === 0) return { x: cx, y: cy };
    var scale = Math.min(box.w / 2 / Math.abs(dx || 1e-9), box.h / 2 / Math.abs(dy || 1e-9));
    return { x: cx + dx * scale, y: cy + dy * scale };
  }

  function boxes(figure, spec) {
    var byId = {}, right = 0, bottom = 0;
    spec.boxes.forEach(function (b) {
      byId[b.id] = b;
      right = Math.max(right, b.x + b.w);
      bottom = Math.max(bottom, b.y + b.h);
    });
    var svg = svgFor(figure, right + 2, bottom + 2);
    spec.boxes.forEach(function (b) {
      var cls = "kd-box" + (b.gem ? " g-" + b.gem : "") + (b.dashed ? " kd-dashed" : "") + (b.here ? " kd-here" : "");
      el("rect", { x: b.x, y: b.y, width: b.w, height: b.h, rx: 10, class: cls }, svg);
      if (b.t) {
        var lines = String(b.t).split("\n").length;
        text(svg, b.x + b.w / 2, b.y + b.h / 2 - (lines - 1) * LINE / 2 + 5, b.t, b.small ? "kd-small" : "", "middle");
      }
    });
    (spec.arrows || []).forEach(function (a) {
      var from = byId[a.from], to = byId[a.to];
      if (!from || !to) throw new Error("diagram: no box " + (from ? a.to : a.from));
      var fc = { x: from.x + from.w / 2, y: from.y + from.h / 2 }, tc = { x: to.x + to.w / 2, y: to.y + to.h / 2 };
      var p = edge(from, tc), q = edge(to, fc), kind = a.kind || "flow";
      // Two arrows between the same boxes, one each way, run side by side.
      var back = (spec.arrows || []).some(function (b) { return b.from === a.to && b.to === a.from; });
      if (back) {
        var length = Math.hypot(q.x - p.x, q.y - p.y) || 1;
        var side = (a.from < a.to ? 1 : -1) * 6;
        var nx = -(q.y - p.y) / length * side, ny = (q.x - p.x) / length * side;
        p = { x: p.x + nx, y: p.y + ny };
        q = { x: q.x + nx, y: q.y + ny };
      }
      el("line", {
        x1: p.x, y1: p.y, x2: q.x, y2: q.y, class: "kd-arrow kd-" + kind,
        "marker-end": "url(#kd-head-" + kind + ")"
      }, svg);
      if (a.t) text(svg, (p.x + q.x) / 2 + (a.dx || 6), (p.y + q.y) / 2 + (a.dy || -4), a.t, "kd-small", "start");
    });
    put(figure, svg);
  }

  // ---- flow ----
  function flow(figure, spec) {
    var steps = spec.steps, BOX_H = 34, GAP = 22, PAD = 14;
    var longest = Math.max.apply(null, steps.map(function (s) {
      return Math.max.apply(null, String(s.t).split("\n").map(function (l) { return l.length; }));
    }));
    var boxW = Math.max(150, longest * 7.4 + 2 * PAD);
    var heights = steps.map(function (s) { return Math.max(BOX_H, String(s.t).split("\n").length * LINE + 16); });
    var back = spec.back ? 70 : 0;
    var total = heights.reduce(function (a, b) { return a + b; }, 0) + GAP * (steps.length - 1);
    var svg = svgFor(figure, boxW + back, total);
    var y = 0, tops = [];
    steps.forEach(function (s, i) {
      tops.push(y);
      var cls = "kd-box" + (s.gem ? " g-" + s.gem : "") + (s.here ? " kd-here" : "");
      el("rect", { x: 0, y: y, width: boxW, height: heights[i], rx: 10, class: cls }, svg);
      var lines = String(s.t).split("\n").length;
      text(svg, boxW / 2, y + heights[i] / 2 - (lines - 1) * LINE / 2 + 5, s.t, "", "middle");
      if (i + 1 < steps.length) {
        el("line", {
          x1: boxW / 2, y1: y + heights[i], x2: boxW / 2, y2: y + heights[i] + GAP,
          class: "kd-arrow kd-flow", "marker-end": "url(#kd-head-flow)"
        }, svg);
      }
      y += heights[i] + GAP;
    });
    if (spec.back) {
      var from = spec.back.from - 1, to = spec.back.to - 1;
      var y1 = tops[from] + heights[from] / 2, y2 = tops[to] + heights[to] / 2;
      el("path", {
        d: "M" + boxW + " " + y1 + " H" + (boxW + 30) + " V" + y2 + " H" + (boxW + 2),
        class: "kd-arrow kd-flow", fill: "none", "marker-end": "url(#kd-head-flow)"
      }, svg);
      if (spec.back.t) text(svg, boxW + 36, (y1 + y2) / 2 + 4, spec.back.t, "kd-small", "start");
    }
    put(figure, svg);
  }

  function draw(figure) {
    var block = figure.querySelector('script[type="application/json"]');
    if (!block) return;
    var spec;
    try { spec = JSON.parse(block.textContent); } catch (error) { return; }
    var kinds = { anatomy: anatomy, boxes: boxes, flow: flow };
    if (kinds[spec.kind]) kinds[spec.kind](figure, spec);
  }

  function start() {
    Array.prototype.forEach.call(document.querySelectorAll("figure.km-diagram"), draw);
  }

  // The site loads pages without a reload (instant navigation) when Material's
  // observable exists; draw on every page view.
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(start);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
