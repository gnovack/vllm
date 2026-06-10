#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight web viewer for CUPTI metric DBs produced by --enable-cupti.

Kernels are organized into a *call tree* keyed by their shared vLLM callstack
frames, so you can see which kernels hang off the same place in the vLLM source
and how each one's metrics (duration, HBM bytes, tensor-op FLOPs, and derived
TFLOP/s / GB/s / arithmetic intensity) scale with num_tokens.

  # write a self-contained HTML file you can open anywhere
  python tools/cupti_web.py ./cupti_metrics -o report.html

  # or serve it live (re-reads the DBs on every refresh, even mid-run)
  python tools/cupti_web.py ./cupti_metrics --serve 8000

No third-party dependencies: stdlib sqlite3 + http.server, vanilla JS/CSS.
"""
import argparse
import glob
import html
import json
import math
import os
import sqlite3
import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer


# --------------------------------------------------------------------------- #
# Data loading (merge per-rank DBs)
# --------------------------------------------------------------------------- #
def _db_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.db")))
    if not files:
        sys.exit(f"No .db files found under {path!r}")
    return files


def load_data(path: str):
    """Returns (kernels, agg, meta).

    kernels: kid -> (name, full_name, stack)
    agg:     (kid, num_tokens, metric) -> [count, sum, sum_sq, min, max]
    meta:    {'model':.., 'ranks':N}
    """
    kernels: dict = {}
    agg: dict = {}
    models: set = set()
    files = _db_files(path)
    for f in files:
        conn = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        try:
            try:
                for k, v in conn.execute("SELECT key, value FROM meta"):
                    if k == "model" and v:
                        models.add(v)
            except sqlite3.OperationalError:
                pass
            for kid, name, full, stack in conn.execute(
                "SELECT kernel_id, name, full_name, stack FROM kernels"
            ):
                kernels.setdefault(kid, (name, full, stack))
            for kid, ntok, metric, c, s, ssq, mn, mx in conn.execute(
                "SELECT kernel_id, num_tokens, metric, count, sum, sum_sq, min, max "
                "FROM kernel_metrics"
            ):
                key = (kid, ntok, metric)
                cur = agg.get(key)
                if cur is None:
                    agg[key] = [c, s, ssq, mn, mx]
                else:
                    cur[0] += c
                    cur[1] += s
                    cur[2] += ssq
                    cur[3] = min(cur[3], mn)
                    cur[4] = max(cur[4], mx)
        finally:
            conn.close()
    meta = {"model": ", ".join(sorted(models)) or "(unknown)", "ranks": len(files)}
    return kernels, agg, meta


def _mean_std(count, s, ssq):
    if count <= 0:
        return 0.0, 0.0
    mean = s / count
    var = max(ssq / count - mean * mean, 0.0)
    return mean, math.sqrt(var)


# --------------------------------------------------------------------------- #
# Build the call-tree payload
# --------------------------------------------------------------------------- #
# stored metric name -> (friendly label, family). family drives derived metrics.
_METRIC_META = {
    "gpu_dur_ns": ("duration", "dur"),
    "dram_bytes_read": ("HBM read", "bytes"),
    "dram_bytes_write": ("HBM write", "bytes"),
    "tensor_ops_bf16": ("bf16 FLOPs", "flop"),
    "tensor_ops_fp16": ("fp16 FLOPs", "flop"),
    "tensor_ops_fp8": ("fp8 FLOPs", "flop"),
    "tensor_ops_int8": ("int8 ops", "flop"),
    "tensor_ops_tf32": ("tf32 FLOPs", "flop"),
}


def _frames(stack: str) -> list[str]:
    """Stack is 'inner <- ... <- outer'; return OUTERMOST-first for tree paths."""
    if not stack:
        return []
    return list(reversed([s.strip() for s in stack.split("<-") if s.strip()]))


def _kernel_rows(kid, agg):
    """Per-num_tokens metric stats + derived perf numbers for one kernel."""
    by_ntok: dict = {}
    for (k, ntok, metric), st in agg.items():
        if k != kid:
            continue
        mean, std = _mean_std(st[0], st[1], st[2])
        by_ntok.setdefault(ntok, {})[metric] = {
            "count": st[0], "mean": mean, "min": st[3], "max": st[4], "std": std,
        }
    rows = []
    for ntok in sorted(by_ntok):
        m = by_ntok[ntok]
        dur_ns = m.get("gpu_dur_ns", {}).get("mean", 0.0)
        dur_s = dur_ns * 1e-9
        nbytes = sum(m[x]["mean"] for x in ("dram_bytes_read", "dram_bytes_write")
                     if x in m)
        flops = sum(v["mean"] for name, v in m.items()
                    if _METRIC_META.get(name, ("", ""))[1] == "flop")
        derived = {}
        if dur_s > 0 and flops > 0:
            derived["tflops"] = flops / dur_s / 1e12
        if dur_s > 0 and nbytes > 0:
            derived["gbps"] = nbytes / dur_s / 1e9
        if nbytes > 0 and flops > 0:
            derived["ai"] = flops / nbytes
        rows.append({"ntok": ntok, "metrics": m, "derived": derived})
    return rows


def build_payload(kernels, agg, meta):
    metrics_present = sorted({m for (_, _, m) in agg})
    primary = "gpu_dur_ns" if "gpu_dur_ns" in metrics_present else (
        metrics_present[0] if metrics_present else None)

    # primary-metric total per kernel (for heat / ranking)
    kid_total: dict = {}
    for (kid, _ntok, m), st in agg.items():
        if m == primary:
            kid_total[kid] = kid_total.get(kid, 0.0) + st[1]

    root = {"label": "all kernels", "type": "frame", "children": {},
            "total": 0.0, "kernels": []}

    for kid, (name, full, stack) in kernels.items():
        frames = _frames(stack)
        path = frames or ["(no vLLM callstack)"]
        node = root
        node["total"] += kid_total.get(kid, 0.0)
        for frame in path:
            child = node["children"].get(frame)
            if child is None:
                child = {"label": frame, "type": "frame", "children": {},
                         "total": 0.0, "kernels": []}
                node["children"][frame] = child
            child["total"] += kid_total.get(kid, 0.0)
            node = child
        if not frames:
            # No callstack: many unrelated kernels share this node, so they are
            # NOT one logical kernel -- keep them individually selectable.
            node["nostack"] = True
        node["kernels"].append({
            "kid": kid, "name": name, "full_name": full,
            "total": kid_total.get(kid, 0.0),
            "rows": _kernel_rows(kid, agg),
        })

    def finalize(node):
        # Collapse unbranching frame chains: a -> b -> c (each single child, no
        # kernels of its own) becomes one node "a / b / c".
        labels = [node["label"]]
        while (len(node["children"]) == 1 and not node["kernels"]
               and node is not root):
            (only,) = node["children"].values()
            labels.append(only["label"])
            node["children"] = only["children"]
            node["kernels"] = only["kernels"]
        node["label"] = "  ›  ".join(labels)
        kids = sorted(node["children"].values(), key=lambda n: n["total"],
                      reverse=True)
        node["children"] = [finalize(c) for c in kids]
        node["kernels"].sort(key=lambda k: k["total"], reverse=True)
        return node

    finalize(root)
    return {
        "meta": meta,
        "primary": primary,
        "metrics": metrics_present,
        "metric_meta": {m: _METRIC_META.get(m, (m, "other"))
                        for m in metrics_present},
        "tree": root,
        "n_kernels": len(kernels),
        "total_primary": root["total"],
    }


# --------------------------------------------------------------------------- #
# HTML rendering (vanilla JS, data embedded as JSON)
# --------------------------------------------------------------------------- #
_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CUPTI kernel explorer</title>
<style>
:root{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2330; --line:#2b3340;
  --fg:#e6edf3; --dim:#8b949e; --accent:#58a6ff; --hot:#f78166;
  --good:#3fb950; --warn:#d29922; --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:baseline;gap:18px;padding:12px 18px;
  border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5}
header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.2px}
header .stat{color:var(--dim);font-size:12px}
header .stat b{color:var(--fg);font-weight:600}
#wrap{display:flex;height:calc(100vh - 49px)}
#left{width:54%;min-width:360px;overflow:auto;padding:10px 4px 40px 12px;
  border-right:1px solid var(--line)}
#right{flex:1;overflow:auto;padding:18px 22px 60px}
.toolbar{display:flex;gap:10px;align-items:center;margin:4px 8px 10px}
.toolbar input{flex:1;background:var(--panel2);border:1px solid var(--line);
  color:var(--fg);border-radius:7px;padding:7px 10px;font-size:13px;outline:none}
.toolbar input:focus{border-color:var(--accent)}
.toolbar select{background:var(--panel2);border:1px solid var(--line);color:var(--fg);
  border-radius:7px;padding:7px 8px;font-size:12px}
.node{margin-left:14px;border-left:1px solid var(--line);padding-left:2px}
.row{display:flex;align-items:center;gap:7px;padding:2px 6px;border-radius:6px;
  cursor:pointer;position:relative;white-space:nowrap}
.row:hover{background:var(--panel)}
.row.sel{background:#1f6feb33;outline:1px solid var(--accent)}
.tw{width:12px;text-align:center;color:var(--dim);flex:none;font-size:11px}
.frame{font-family:var(--mono);font-size:12px;color:var(--dim)}
.frame .fn{color:var(--accent)}
.kern{font-family:var(--mono);font-size:12.5px;color:var(--fg)}
.kdot{flex:none;width:7px;height:7px;border-radius:50%;background:var(--hot)}
.vbadge{flex:none;font-size:10px;color:var(--warn);border:1px solid var(--line);
  border-radius:10px;padding:0 6px;margin-left:2px}
.pct{color:var(--dim);font-size:11px;font-variant-numeric:tabular-nums;
  margin-left:auto;padding-left:10px;flex:none;z-index:1}
.muted{color:var(--dim)}
h2.kname{font-family:var(--mono);font-size:16px;margin:0 0 4px;color:var(--fg);
  word-break:break-all}
.crumb{font-family:var(--mono);font-size:11.5px;color:var(--dim);margin-bottom:14px;
  line-height:1.7;word-break:break-all}
.crumb .fn{color:var(--accent)}
.full{font-family:var(--mono);font-size:11px;color:var(--dim);background:var(--panel);
  border:1px solid var(--line);border-radius:7px;padding:8px 10px;margin-bottom:16px;
  white-space:pre-wrap;word-break:break-all;max-height:120px;overflow:auto}
table{border-collapse:collapse;width:100%;margin-bottom:18px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);font-size:12.5px}
th{color:var(--dim);font-weight:600;position:sticky;top:0;background:var(--bg);
  text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td.ntok{font-family:var(--mono);color:var(--accent)}
tr:hover td{background:var(--panel)}
td.kv{font-family:var(--mono);font-size:11px;color:var(--warn);text-align:left}
tr.alt td{border-top:2px solid var(--line)}
.kleg{font-family:var(--mono);font-size:11px;color:var(--dim);margin:-6px 0 14px;
  line-height:1.7}
.kleg b{color:var(--warn);font-weight:600}
.derived td{color:var(--good)}
.sect{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  margin:22px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.empty{color:var(--dim);margin-top:30px;text-align:center}
.legend{color:var(--dim);font-size:11px;margin:-8px 0 14px}
.pill{display:inline-block;background:var(--panel2);border:1px solid var(--line);
  border-radius:20px;padding:1px 9px;margin:0 4px 4px 0;font-size:11px;
  font-family:var(--mono)}
</style></head><body>
<header>
  <h1>CUPTI kernel explorer</h1>
  <span class="stat">model <b id="m-model"></b></span>
  <span class="stat"><b id="m-kernels"></b> kernels</span>
  <span class="stat"><b id="m-ranks"></b> rank(s)</span>
  <span class="stat">total <b id="m-total"></b></span>
</header>
<div id="wrap">
  <div id="left">
    <div class="toolbar">
      <input id="search" placeholder="filter kernels &amp; source frames…">
      <label class="muted" style="font-size:12px">rank by</label>
      <select id="rankby"></select>
    </div>
    <div id="tree"></div>
  </div>
  <div id="right"><div class="empty">← pick a kernel to see its metrics</div></div>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const MM = D.metric_meta || {};
const FAM = m => (MM[m] || [m,'other'])[1];
const LBL = m => (MM[m] || [m,'other'])[0];

// ---- formatters ----------------------------------------------------------
function fmtDur(ns){ if(ns>=1e6) return (ns/1e6).toFixed(2)+' ms';
  if(ns>=1e3) return (ns/1e3).toFixed(2)+' µs'; return ns.toFixed(0)+' ns'; }
function fmtBytes(b){ const u=['B','KB','MB','GB','TB']; let i=0;
  while(b>=1024&&i<u.length-1){b/=1024;i++;} return b.toFixed(b<10&&i>0?2:1)+' '+u[i]; }
function fmtSI(x){ const u=['','K','M','G','T','P']; let i=0;
  while(Math.abs(x)>=1000&&i<u.length-1){x/=1000;i++;} return x.toFixed(x<10&&i>0?2:1)+u[i]; }
function fmtVal(metric,v){ const f=FAM(metric);
  if(f==='dur')return fmtDur(v); if(f==='bytes')return fmtBytes(v);
  if(f==='flop')return fmtSI(v); return fmtSI(v); }
function fmtTotal(v){ return FAM(D.primary)==='dur'?fmtDur(v):
  FAM(D.primary)==='bytes'?fmtBytes(v):fmtSI(v); }

// ---- header --------------------------------------------------------------
document.getElementById('m-model').textContent = D.meta.model;
document.getElementById('m-kernels').textContent = D.n_kernels;
document.getElementById('m-ranks').textContent = D.meta.ranks;
document.getElementById('m-total').textContent = fmtTotal(D.total_primary);

// ---- rank-by selector ----------------------------------------------------
const rankSel = document.getElementById('rankby');
let RANK = D.primary;
for(const m of D.metrics){ const o=document.createElement('option');
  o.value=m; o.textContent=LBL(m); if(m===D.primary)o.selected=true; rankSel.appendChild(o); }
rankSel.onchange = ()=>{ RANK=rankSel.value; renderTree(); };

// recompute a node's total under the currently-selected rank metric
function nodeTotal(node){
  if(RANK===D.primary) return node.total;
  if(node._rt && node._rtm===RANK) return node._rt;
  let t=0;
  for(const k of node.kernels) t+=kernelMetricTotal(k,RANK);
  for(const c of node.children) t+=nodeTotal(c);
  node._rt=t; node._rtm=RANK; return t;
}
function kernelMetricTotal(k,m){
  let t=0; for(const r of k.rows) if(r.metrics[m]) t+=r.metrics[m].mean*r.metrics[m].count;
  return t;
}

// ---- tree rendering ------------------------------------------------------
const tree = document.getElementById('tree');
let SEL=null, FILTER='';
function frameHTML(label){
  // label is "path:line(func)  ›  path:line(func)"; color the (func) parts
  return label.replace(/\(([^)]*)\)/g,'(<span class="fn">$1</span>)');
}
function matches(node){
  if(!FILTER) return true;
  if(node.label.toLowerCase().includes(FILTER)) return true;
  if(node.kernels.some(k=>k.name.toLowerCase().includes(FILTER)
      || k.full_name.toLowerCase().includes(FILTER))) return true;
  return node.children.some(matches);
}
function renderTree(){
  tree.innerHTML='';
  for(const c of D.tree.children) if(matches(c)) tree.appendChild(nodeEl(c,D.tree));
}
function nodeEl(node, parent){
  const wrap=document.createElement('div'); wrap.className='node';
  const total=nodeTotal(node), ptotal=nodeTotal(parent)||1;
  const frac=Math.max(0,Math.min(1,total/ptotal));
  const row=document.createElement('div'); row.className='row';
  const open = !!FILTER || node._open;
  const hasKids = node.children.length || node.kernels.length;
  const tw=document.createElement('span'); tw.className='tw';
  tw.textContent = hasKids ? (open?'▾':'▸') : '·';
  row.appendChild(tw);
  const lab=document.createElement('span'); lab.className='frame';
  lab.innerHTML=frameHTML(node.label); row.appendChild(lab);
  const pct=document.createElement('span'); pct.className='pct';
  pct.textContent=fmtTotal(total)+'  ('+(frac*100).toFixed(0)+'%)'; row.appendChild(pct);
  const kids=document.createElement('div'); kids.style.display=open?'block':'none';
  row.onclick=()=>{ node._open=!open; renderTree(); };
  wrap.appendChild(row); wrap.appendChild(kids);
  if(open){
    for(const c of node.children) if(matches(c)) kids.appendChild(nodeEl(c,node));
    // Kernels sharing this exact callstack are ONE logical kernel (the call
    // site; the underlying CUDA kernel may differ by num_tokens). Render a
    // single leaf per logical kernel -- except under the no-callstack node,
    // where the kernels are unrelated and stay individual.
    const groups = node.nostack ? node.kernels.map(k=>[k]) : (node.kernels.length ? [node.kernels] : []);
    for(const g of groups){
      if(FILTER && !(node.label.toLowerCase().includes(FILTER)
          || g.some(k=>k.name.toLowerCase().includes(FILTER)
                    || k.full_name.toLowerCase().includes(FILTER)))) continue;
      kids.appendChild(logicalEl(g,node));
    }
  }
  return wrap;
}
function groupId(g){ return g.map(k=>k.kid).join(','); }
function logicalEl(members,parent){
  const wrap=document.createElement('div'); wrap.className='node';
  const gid=groupId(members);
  const row=document.createElement('div'); row.className='row'+(SEL===gid?' sel':'');
  const tw=document.createElement('span'); tw.className='tw'; row.appendChild(tw);
  const dot=document.createElement('span'); dot.className='kdot'; row.appendChild(dot);
  const kern=document.createElement('span'); kern.className='kern';
  kern.textContent = members[0].name; row.appendChild(kern);
  if(members.length>1){
    const vb=document.createElement('span'); vb.className='vbadge';
    vb.textContent = members.length+' variants'; row.appendChild(vb);
  }
  const total=members.reduce((s,k)=>s+kernelMetricTotal(k,RANK),0);
  const pct=document.createElement('span'); pct.className='pct';
  pct.textContent=fmtTotal(total); row.appendChild(pct);
  row.onclick=(e)=>{e.stopPropagation(); SEL=gid; renderTree(); showLogical(members);};
  wrap.appendChild(row); return wrap;
}

// ---- detail panel --------------------------------------------------------
const right=document.getElementById('right');
const multi = members => members.length>1;

// Merge a logical kernel's members into rows sorted by num_tokens. Each row
// carries which underlying-kernel variant (member index) produced it, so a
// size-selected kernel reads as one scaling curve with the switch called out.
function mergedRows(members){
  const rows=[];
  members.forEach((k,mi)=>{ for(const r of k.rows) rows.push({...r, mi}); });
  rows.sort((a,b)=> (a.ntok-b.ntok) || (a.mi-b.mi));
  return rows;
}
function showLogical(members){
  const metrics=D.metrics.slice();
  const order={dur:0,bytes:1,flop:2,other:3};
  metrics.sort((a,b)=>(order[FAM(a)]-order[FAM(b)]) || a.localeCompare(b));
  const rows=mergedRows(members);
  const m0=members[0];

  let h='<h2 class="kname">'+esc(m0.name)+(multi(members)
        ? ' <span class="vbadge">'+members.length+' underlying kernels</span>' : '')+'</h2>';
  h+='<div class="crumb">'+crumbHTML(m0.kid)+'</div>';
  if(multi(members)){
    h+='<div class="kleg">one call site, kernel selected per num_tokens:<br>'
      + members.map((k,i)=>'<b>K'+(i+1)+'</b> '+esc(k.name)).join('<br>')+'</div>';
  } else {
    h+='<div class="full">'+esc(m0.full_name)+'</div>';
  }
  h+='<div class="legend">per-launch <b>mean</b> across all ranks; '
    +'derived rows use the means at that token count.</div>';

  // raw metrics table
  h+='<div class="sect">captured metrics &times; num_tokens</div>';
  h+='<table><thead><tr><th>num_tokens</th>'
    + (multi(members)?'<th>kernel</th>':'') + '<th>launches</th>';
  for(const m of metrics) h+='<th title="'+esc(m)+'">'+esc(LBL(m))+'</th>';
  h+='</tr></thead><tbody>';
  let prev=null;
  for(const r of rows){
    const lab = r.ntok<0 ? 'no-ctx' : r.ntok;
    const alt = (prev!==null && r.ntok!==prev) ? ' class="alt"' : '';
    prev=r.ntok;
    let launches=0; for(const m of metrics) if(r.metrics[m]) launches=Math.max(launches,r.metrics[m].count);
    h+='<tr'+alt+'><td class="ntok">'+lab+'</td>'
      + (multi(members)?'<td class="kv" title="'+esc(members[r.mi].name)+'">K'+(r.mi+1)+'</td>':'')
      + '<td class="muted">'+launches+'</td>';
    for(const m of metrics){
      if(r.metrics[m]) h+='<td>'+fmtVal(m,r.metrics[m].mean)+'</td>';
      else h+='<td class="muted">–</td>';
    }
    h+='</tr>';
  }
  h+='</tbody></table>';

  // derived perf table (only if any derived present)
  const hasD=rows.some(r=>Object.keys(r.derived).length);
  if(hasD){
    h+='<div class="sect">derived performance</div>';
    h+='<table><thead><tr><th>num_tokens</th>'+(multi(members)?'<th>kernel</th>':'')
      +'<th>TFLOP/s</th><th>HBM GB/s</th>'
      +'<th>arith. intensity (FLOP/byte)</th></tr></thead><tbody class="derived">';
    prev=null;
    for(const r of rows){
      if(!Object.keys(r.derived).length) continue;
      const lab = r.ntok<0?'no-ctx':r.ntok;
      const alt = (prev!==null && r.ntok!==prev) ? ' class="alt"' : ''; prev=r.ntok;
      h+='<tr'+alt+'><td class="ntok">'+lab+'</td>'
        + (multi(members)?'<td class="kv">K'+(r.mi+1)+'</td>':'')
        +'<td>'+(r.derived.tflops!=null?r.derived.tflops.toFixed(1):'–')+'</td>'
        +'<td>'+(r.derived.gbps!=null?r.derived.gbps.toFixed(0):'–')+'</td>'
        +'<td>'+(r.derived.ai!=null?r.derived.ai.toFixed(1):'–')+'</td></tr>';
    }
    h+='</tbody></table>';
  }
  right.innerHTML=h;
}
function crumbHTML(kid){
  const path=findPath(D.tree,kid,[]);
  if(!path) return '<span class="muted">(no vLLM callstack)</span>';
  return path.map(frameHTML).join('<br>&nbsp;&nbsp;↳ ');
}
function findPath(node,kid,acc){
  const here=node===D.tree?acc:acc.concat([node.label]);
  if(node.kernels.some(x=>x.kid===kid)) return here;
  for(const c of node.children){ const r=findPath(c,kid,here); if(r) return r; }
  return null;
}
function esc(s){ return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

document.getElementById('search').oninput=(e)=>{ FILTER=e.target.value.trim().toLowerCase();
  // auto-open top level under filter
  renderTree(); };
// open the hottest path by default
(function openHot(){ let n=D.tree; for(let i=0;i<3&&n.children.length;i++){ n=n.children[0]; n._open=true; } })();
renderTree();
</script></body></html>
"""


def render_html(payload) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__PAYLOAD__", data)


# --------------------------------------------------------------------------- #
# CLI / server
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, path=None, **kw):
        self._path = path
        super().__init__(*a, **kw)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        try:
            kernels, agg, meta = load_data(self._path)
            body = render_html(build_payload(kernels, agg, meta)).encode("utf-8")
        except Exception as e:  # keep the server up; show the error
            body = f"<pre>error reading {self._path}:\n{html.escape(str(e))}</pre>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="DB file or directory of per-rank *.db files")
    ap.add_argument("-o", "--out", metavar="FILE.html",
                    help="write a self-contained HTML file and exit")
    ap.add_argument("--serve", nargs="?", const=8000, type=int, metavar="PORT",
                    help="serve live (re-reads DBs each request); default port 8000")
    args = ap.parse_args()

    if args.serve is not None:
        srv = HTTPServer(("0.0.0.0", args.serve), partial(_Handler, path=args.path))
        print(f"serving CUPTI explorer for {args.path!r} at "
              f"http://localhost:{args.serve}  (Ctrl-C to stop, refresh for live data)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")
        return

    kernels, agg, meta = load_data(args.path)
    html_doc = render_html(build_payload(kernels, agg, meta))
    out = args.out or "cupti_report.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"wrote {out}  ({len(kernels)} kernels) — open it in a browser")


if __name__ == "__main__":
    main()
