"""把探索记忆（core/exploration_memory.py）落盘的节点+操作图渲染成一个可以拖拽/
缩放/搜索的本地网页，人工review探索图谱质量时用（孤立节点、连通分量碎片化、
误合并等问题肉眼很容易发现，比一个个翻json文件快得多）。

只读，不修改exploration_logs下的任何数据。生成的HTML是完全自包含的（没有CDN
依赖、没有外部脚本），跟core/log_broadcaster.py的仪表盘一个风格，双击就能在
浏览器里打开，不需要起服务器。

用法：
    python core/visualize_graph.py <game_id> [输出路径]
    不给输出路径就默认存到 exploration_logs/<game_id>/graph.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exploration_memory import ExplorationMemory  # noqa: E402


def _build_graph_data(memory: ExplorationMemory) -> dict[str, Any]:
    incoming: dict[str, int] = {nid: 0 for nid in memory.nodes}
    edges: list[dict[str, Any]] = []

    for node_id, node in memory.nodes.items():
        for action in node.get("actions", []):
            target = action.get("leads_to")
            if not target or target == node_id or target not in memory.nodes:
                continue  # 自环（操作没换界面）或指向已被删除节点的悬空边，图上不画
            edges.append(
                {
                    "source": node_id,
                    "target": target,
                    "type": action.get("type", "?"),
                    "trigger_text": action.get("trigger_text"),
                    "attempt_count": action.get("attempt_count", 1),
                }
            )
            incoming[target] = incoming.get(target, 0) + 1

    nodes = []
    for node_id, node in memory.nodes.items():
        ineffective = sum(
            1
            for a in node.get("actions", [])
            if "leads_to" in a and not a.get("effective", a["leads_to"] != node_id)
        )
        out_degree = sum(1 for e in edges if e["source"] == node_id)
        nodes.append(
            {
                "id": node_id,
                "description": node.get("description", ""),
                "tokens": node.get("tokens", [])[:8],  # 悬浮提示用，够看个大概就行，不用全量
                "count": node.get("count", 1),
                "in_degree": incoming.get(node_id, 0),
                "out_degree": out_degree,
                "ineffective_count": ineffective,
                "isolated": incoming.get(node_id, 0) == 0 and out_degree == 0,
            }
        )

    return {"game_id": memory.game_id, "nodes": nodes, "edges": edges}


_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>探索图谱 - __GAME_ID__</title>
<style>
  .viz-root {
    --surface-1:      #fcfcfb;
    --surface-2:      #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --series-1:       #2a78d6;
    --series-1-dark:  #184f95;
    --series-6:       #e34948;
    --seq-100:        #cde2fb;
    --seq-400:        #3987e5;
    --seq-700:        #0d366b;
    --border:         rgba(11,11,11,0.10);
  }
  @media (prefers-color-scheme: dark) {
    .viz-root {
      --surface-1:      #1a1a19;
      --surface-2:      #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --series-1:       #3987e5;
      --series-1-dark:  #86b6ef;
      --series-6:       #e66767;
      --seq-100:        #184f95;
      --seq-400:        #3987e5;
      --seq-700:        #cde2fb;
      --border:         rgba(255,255,255,0.10);
    }
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    background: var(--surface-2);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .viz-root { display: flex; flex-direction: column; height: 100vh; }
  header {
    display: flex; align-items: center; gap: 16px;
    padding: 10px 16px; border-bottom: 1px solid var(--gridline);
    background: var(--surface-1);
  }
  h1 { font-size: 14px; margin: 0; font-weight: 600; }
  #stats { font-size: 12px; color: var(--text-secondary); }
  #search {
    margin-left: auto; padding: 6px 10px; border-radius: 6px;
    border: 1px solid var(--baseline); background: var(--surface-1);
    color: var(--text-primary); font-size: 12px; width: 220px;
  }
  #search::placeholder { color: var(--text-muted); }
  main { position: relative; flex: 1; overflow: hidden; }
  svg { width: 100%; height: 100%; cursor: grab; }
  svg:active { cursor: grabbing; }
  .edge { stroke: var(--baseline); stroke-width: 1.2; fill: none; opacity: 0.55; }
  .edge.dim { opacity: 0.08; }
  .edge.highlight { stroke: var(--series-1); stroke-width: 2; opacity: 0.9; }
  .node circle { stroke: var(--surface-1); stroke-width: 1.5; cursor: pointer; }
  .node.dim { opacity: 0.15; }
  .node.highlight circle { stroke: var(--series-6); stroke-width: 2.5; }
  .node text {
    font-size: 9px; fill: var(--text-secondary); pointer-events: none;
    text-anchor: middle;
  }
  .node.dim text { opacity: 0; }
  #legend {
    position: absolute; left: 12px; bottom: 12px;
    background: var(--surface-1); border: 1px solid var(--gridline);
    border-radius: 8px; padding: 10px 12px; font-size: 11px;
    color: var(--text-secondary); line-height: 1.7;
  }
  #legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
  #tooltip {
    position: absolute; pointer-events: none; display: none;
    background: var(--surface-1); border: 1px solid var(--gridline);
    border-radius: 8px; padding: 8px 10px; font-size: 12px; max-width: 320px;
    box-shadow: 0 4px 16px var(--border); z-index: 10;
  }
  #tooltip .title { font-weight: 600; margin-bottom: 4px; }
  #tooltip .row { color: var(--text-secondary); font-size: 11px; }
  #tooltip .muted { color: var(--text-muted); }
</style>
</head>
<body>
<div class="viz-root">
  <header>
    <h1>探索图谱 · __GAME_ID__</h1>
    <span id="stats"></span>
    <input id="search" type="text" placeholder="搜索节点id / 描述文字...">
  </header>
  <main>
    <svg id="canvas"></svg>
    <div id="legend">
      <div><span class="swatch" style="background:var(--seq-100)"></span>访问次数少</div>
      <div><span class="swatch" style="background:var(--seq-700)"></span>访问次数多</div>
      <div><span class="swatch" style="background:var(--series-6);border:1.5px solid var(--series-6)"></span>选中节点</div>
      <div style="margin-top:4px;color:var(--text-muted)">拖动节点调整布局 · 滚轮缩放 · 点击节点高亮邻居</div>
    </div>
    <div id="tooltip"></div>
  </main>
</div>
<script>
const DATA = __GRAPH_JSON__;

const svg = document.getElementById("canvas");
const tooltip = document.getElementById("tooltip");
const statsEl = document.getElementById("stats");
const searchEl = document.getElementById("search");
const NS = "http://www.w3.org/2000/svg";

const nodes = DATA.nodes.map((n, i) => ({
  ...n,
  x: 400 + 300 * Math.cos((i / DATA.nodes.length) * Math.PI * 2),
  y: 300 + 300 * Math.sin((i / DATA.nodes.length) * Math.PI * 2),
  vx: 0, vy: 0,
}));
const nodeById = new Map(nodes.map(n => [n.id, n]));
const edges = DATA.edges
  .filter(e => nodeById.has(e.source) && nodeById.has(e.target))
  .map(e => ({ ...e, source: nodeById.get(e.source), target: nodeById.get(e.target) }));

statsEl.textContent = `${nodes.length} 个节点 · ${edges.length} 条边`;

const counts = nodes.map(n => n.count);
const maxCount = Math.max(1, ...counts);
function seqColor(count) {
  const t = Math.log(count + 1) / Math.log(maxCount + 1);
  // 三段线性插值：seq-100 -> seq-400 -> seq-700（跟palette.md的顺序蓝色序列一致）
  const stops = [
    [0.0, [205, 226, 251]],
    [0.5, [57, 135, 229]],
    [1.0, [13, 54, 107]],
  ];
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
  }
  const span = b[0] - a[0] || 1;
  const localT = (t - a[0]) / span;
  const rgb = a[1].map((v, i) => Math.round(v + (b[1][i] - v) * localT));
  return `rgb(${rgb.join(",")})`;
}
function radius(n) {
  return 5 + 3 * Math.sqrt(n.count);
}

const edgeEls = edges.map(e => {
  const el = document.createElementNS(NS, "line");
  el.setAttribute("class", "edge");
  el.setAttribute("marker-end", "url(#arrow)");
  svg.appendChild(el);
  return el;
});

const defs = document.createElementNS(NS, "defs");
defs.innerHTML = `
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="var(--baseline)"></path>
  </marker>
`;
svg.appendChild(defs);

const nodeGroups = nodes.map(n => {
  const g = document.createElementNS(NS, "g");
  g.setAttribute("class", "node");
  const circle = document.createElementNS(NS, "circle");
  circle.setAttribute("r", radius(n));
  circle.setAttribute("fill", n.isolated ? "var(--series-6)" : seqColor(n.count));
  g.appendChild(circle);
  const label = document.createElementNS(NS, "text");
  label.textContent = n.description ? n.description.slice(0, 10) : n.id.split("-").pop();
  label.setAttribute("dy", radius(n) + 11);
  g.appendChild(label);
  svg.appendChild(g);

  g.addEventListener("mouseenter", (ev) => showTooltip(n, ev));
  g.addEventListener("mousemove", (ev) => showTooltip(n, ev));
  g.addEventListener("mouseleave", hideTooltip);
  g.addEventListener("click", () => selectNode(n));
  g.addEventListener("pointerdown", (ev) => startDrag(n, ev));
  return g;
});

function showTooltip(n, ev) {
  const rect = svg.getBoundingClientRect();
  tooltip.style.left = (ev.clientX - rect.left + 14) + "px";
  tooltip.style.top = (ev.clientY - rect.top + 14) + "px";
  tooltip.style.display = "block";
  tooltip.innerHTML = `
    <div class="title">${n.description || "(未描述)"}</div>
    <div class="row">${n.id}</div>
    <div class="row">访问 ${n.count} 次 · 入度 ${n.in_degree} · 出度 ${n.out_degree}</div>
    ${n.ineffective_count ? `<div class="row muted">已知 ${n.ineffective_count} 个无效操作</div>` : ""}
    ${n.isolated ? `<div class="row" style="color:var(--series-6)">孤立节点（无入边也无出边）</div>` : ""}
    <div class="row muted">${(n.tokens || []).join(" · ")}</div>
  `;
}
function hideTooltip() { tooltip.style.display = "none"; }

let selected = null;
function selectNode(n) {
  selected = selected === n ? null : n;
  render();
}
searchEl.addEventListener("input", () => {
  const q = searchEl.value.trim().toLowerCase();
  if (!q) { selected = null; render(); return; }
  selected = nodes.find(n => n.id.toLowerCase().includes(q) || (n.description || "").toLowerCase().includes(q)) || null;
  render();
});

function neighborsOf(n) {
  const s = new Set([n]);
  edges.forEach(e => {
    if (e.source === n) s.add(e.target);
    if (e.target === n) s.add(e.source);
  });
  return s;
}

function render() {
  const active = selected ? neighborsOf(selected) : null;
  nodeGroups.forEach((g, i) => {
    const n = nodes[i];
    g.setAttribute("transform", `translate(${n.x},${n.y})`);
    g.classList.toggle("dim", !!active && !active.has(n));
    g.classList.toggle("highlight", n === selected);
  });
  edgeEls.forEach((el, i) => {
    const e = edges[i];
    el.setAttribute("x1", e.source.x);
    el.setAttribute("y1", e.source.y);
    el.setAttribute("x2", e.target.x);
    el.setAttribute("y2", e.target.y);
    const inActive = !!active && active.has(e.source) && active.has(e.target);
    el.classList.toggle("dim", !!active && !inActive);
    el.classList.toggle("highlight", inActive && !!selected);
  });
}

// 极简力导向：节点两两斥力 + 边的弹簧引力 + 向中心的微弱引力，O(n^2)对这个规模的图够用
function tick() {
  const REPULSION = 2200;
  const SPRING = 0.02;
  const SPRING_LEN = 90;
  const CENTER_PULL = 0.002;
  const cx = svg.clientWidth / 2 || 500;
  const cy = svg.clientHeight / 2 || 400;

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      let dx = a.x - b.x, dy = a.y - b.y;
      let dist2 = dx * dx + dy * dy || 0.01;
      let dist = Math.sqrt(dist2);
      const force = REPULSION / dist2;
      dx /= dist; dy /= dist;
      a.vx += dx * force; a.vy += dy * force;
      b.vx -= dx * force; b.vy -= dy * force;
    }
  }
  edges.forEach(e => {
    let dx = e.target.x - e.source.x, dy = e.target.y - e.source.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
    const force = (dist - SPRING_LEN) * SPRING;
    dx /= dist; dy /= dist;
    e.source.vx += dx * force; e.source.vy += dy * force;
    e.target.vx -= dx * force; e.target.vy -= dy * force;
  });
  nodes.forEach(n => {
    if (n.dragging) return;
    n.vx += (cx - n.x) * CENTER_PULL;
    n.vy += (cy - n.y) * CENTER_PULL;
    n.vx *= 0.85; n.vy *= 0.85;
    n.x += n.vx; n.y += n.vy;
  });
  render();
  requestAnimationFrame(tick);
}

// 拖拽单个节点
let dragNode = null;
function startDrag(n, ev) {
  dragNode = n;
  n.dragging = true;
  ev.stopPropagation();
}
svg.addEventListener("pointermove", (ev) => {
  if (!dragNode) return;
  const rect = svg.getBoundingClientRect();
  dragNode.x = (ev.clientX - rect.left - pan.x) / zoom;
  dragNode.y = (ev.clientY - rect.top - pan.y) / zoom;
  dragNode.vx = 0; dragNode.vy = 0;
});
window.addEventListener("pointerup", () => {
  if (dragNode) dragNode.dragging = false;
  dragNode = null;
});

// 整个画布的平移+缩放
let zoom = 1;
let pan = { x: 0, y: 0 };
let panning = null;
const viewport = document.createElementNS(NS, "g");
// 把已有的edge/node元素挪进一个可以整体缩放平移的<g>里
[...edgeEls, ...nodeGroups].forEach(el => viewport.appendChild(el));
svg.insertBefore(viewport, null);
function updateViewportTransform() {
  viewport.setAttribute("transform", `translate(${pan.x},${pan.y}) scale(${zoom})`);
}
svg.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const rect = svg.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const worldX = (mx - pan.x) / zoom, worldY = (my - pan.y) / zoom;
  zoom *= ev.deltaY < 0 ? 1.1 : 0.9;
  zoom = Math.max(0.2, Math.min(4, zoom));
  pan.x = mx - worldX * zoom;
  pan.y = my - worldY * zoom;
  updateViewportTransform();
}, { passive: false });
svg.addEventListener("pointerdown", (ev) => {
  if (ev.target === svg) panning = { x: ev.clientX - pan.x, y: ev.clientY - pan.y };
});
window.addEventListener("pointermove", (ev) => {
  if (!panning) return;
  pan.x = ev.clientX - panning.x;
  pan.y = ev.clientY - panning.y;
  updateViewportTransform();
});
window.addEventListener("pointerup", () => { panning = null; });

updateViewportTransform();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def render_html(graph_data: dict[str, Any]) -> str:
    graph_json = json.dumps(graph_data, ensure_ascii=False).replace("</script>", "<\\/script>")
    return (
        _PAGE_TEMPLATE
        .replace("__GAME_ID__", graph_data["game_id"])
        .replace("__GRAPH_JSON__", graph_json)
    )


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("用法: python visualize_graph.py <game_id> [输出路径]", file=sys.stderr)
        sys.exit(1)

    game_id = sys.argv[1]
    memory = ExplorationMemory(game_id)
    graph_data = _build_graph_data(memory)
    html = render_html(graph_data)

    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else memory.nodes_dir.parent / "graph.html"
    output_path.write_text(html, encoding="utf-8")

    isolated_count = sum(1 for n in graph_data["nodes"] if n["isolated"])
    print(f"节点 {len(graph_data['nodes'])} 个，边 {len(graph_data['edges'])} 条，孤立节点 {isolated_count} 个")
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
