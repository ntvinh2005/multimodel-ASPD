"""The `/prompt` page (inline HTML and JavaScript)."""

PROMPT_HTML = r'''<!doctype html><html><head><meta charset="utf-8">
<title>prompt trace</title>
<style>
:root{--bg:#ffffff;--fg:#1b1e23;--dim:#6b7280;--line:#e3e6ea;--card:#fbfcfd;--hi:#1f5fc4;
      --pos:#12703a;--neg:#c0271f;--warn:#8a5a00;--bar:#f3f5f8;--hover:#eef3fb;--sel:#e2ecfb;
      --field:#ffffff;--mask:#f2f3f5;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:10px 14px;border-bottom:1px solid var(--line);background:var(--bar);
       position:sticky;top:0;z-index:5}
h1{font-size:13px;margin:0 0 8px;font-weight:600}
h1 span{color:var(--dim);font-weight:400}
input,select,button{background:var(--field);color:var(--fg);border:1px solid var(--line);
                    border-radius:4px;padding:4px 7px;font:inherit}
input:focus,select:focus{outline:1px solid var(--hi)}
button{cursor:pointer}button:hover{border-color:var(--hi)}
button:disabled,select:disabled{opacity:.4;cursor:not-allowed}
button.on{border-color:var(--hi);color:var(--hi)}
#prompt{width:min(560px,52vw)}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px;padding:14px}
@media(max-width:1100px){main{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;
      min-width:0;box-shadow:0 1px 2px rgba(16,24,40,.04)}
.card h2{font-size:12px;margin:0 0 8px;color:var(--dim);font-weight:600;
         text-transform:uppercase;letter-spacing:.04em}
.card h2 span{text-transform:none;letter-spacing:0;font-weight:400}
.wide{grid-column:1/-1}
.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}
#strip{display:flex;flex-wrap:wrap;gap:3px;margin:8px 0 0}
.tok{padding:3px 6px;border-radius:4px;border:1px solid var(--line);cursor:pointer;
     white-space:pre;background:var(--field)}
.tok:hover{border-color:var(--hi)}
.tok.sel{outline:2px solid var(--hi)}
.tok i{color:var(--dim);font-style:normal;font-size:10px;margin-right:4px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;padding:3px 6px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:600;position:sticky;top:0;background:var(--card)}
th.num,td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.click{cursor:pointer}tr.click:hover td{background:var(--hover)}
tr.here td{background:var(--sel)}
td.hit{cursor:pointer}td.hit:hover{color:var(--hi);text-decoration:underline}
.scroll{max-height:320px;overflow:auto}
.pos{color:var(--pos)}.neg{color:var(--neg)}.dim{color:var(--dim)}
.tag{font-size:10px;padding:1px 5px;border-radius:3px;border:1px solid var(--line);
     color:var(--dim);margin-left:4px}
.grid{overflow:auto;max-height:520px}
.grid table{border-collapse:collapse;font-size:10px;width:auto}
.grid td{border:none;padding:0;width:17px;height:17px;cursor:pointer}
.grid td:hover{outline:1px solid var(--hi)}
.grid th{background:var(--card);font-weight:400;color:var(--dim);padding:1px 4px;
         font-size:10px;border:none}
.grid th.r{text-align:right;position:sticky;left:0;z-index:1}
/* Vertical column labels: `vertical-rl` + 180deg reads bottom-to-top, so the token sits next to
   the column it names rather than at the far end of a 78px run. */
.grid th.c{writing-mode:vertical-rl;transform:rotate(180deg);height:78px;width:17px;
           padding:2px 0;vertical-align:bottom;text-align:left;overflow:hidden}
.grid tr.axis th{position:sticky;top:0;z-index:2;background:var(--card)}
.note{color:var(--dim);font-size:11px;margin:6px 0 0}
.hstrip{display:flex;gap:2px;margin-top:8px;flex-wrap:wrap}
.hstrip b{font:400 10px/1 inherit;color:var(--dim);align-self:center;margin-right:4px}
.hstrip span{min-width:46px;padding:3px 4px;border:1px solid var(--line);border-radius:3px;
  font-size:10px;text-align:center;cursor:pointer;line-height:1.35}
.hstrip span:hover{border-color:var(--hi)}
.hstrip span.here{border-color:var(--hi);font-weight:600}
.hstrip span i{display:block;color:var(--dim);font-style:normal;font-size:9px}
.warn{color:var(--warn)}
.err{color:var(--neg)}
.read{font-size:11px;margin:4px 0 6px;min-height:16px}
.ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}
.ctl label{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.ctl input[type=number]{width:72px}
/* One example per line, scrolled sideways. `pre-wrap` used to spill the context out of the card. */
.ex{margin:4px 0;padding:3px 4px;border:1px solid var(--line);border-radius:4px;
    overflow-x:auto;white-space:pre;line-height:1.8}
.ex .tok{padding:1px 2px;border-radius:3px;border:none;cursor:default}
.ex .tok.fire{border-bottom:2px solid var(--hi)}
.pk{color:var(--dim);font-size:10px;margin-right:6px}
.badge{display:inline-block;border:1px solid var(--line);border-radius:3px;
       padding:0 4px;margin:1px 2px;font-size:10px;color:var(--dim)}
.pmi{display:flex;gap:14px;flex-wrap:wrap;margin-top:4px}
.pmi div{min-width:150px}
.chip{border:1px solid var(--hi);border-radius:3px;padding:1px 5px;color:var(--hi);font-size:11px}
</style></head><body>
<header>
  <h1>prompt trace <span id="run"></span>
      <a href="/" style="color:var(--hi);font-weight:400;margin-left:10px">&larr; component pairs</a></h1>
  <input id="prompt" value="When Mary and John went to the store, John gave a drink to">
  <select id="target">
    <option value="topk">target: mean(top-10) &minus; mean(rest)</option>
    <option value="logit_diff">target: logit diff</option>
  </select>
  <input id="correct" value=" Mary" size="7" title="correct token" hidden>
  <input id="wrong" value=" John" size="7" title="wrong token" hidden>
  <button id="go">trace</button>
  <span id="status" class="dim"></span>
</header>
<main>
  <section class="card wide">
    <h2>tokens <span class="dim" id="tgt"></span></h2>
    <div id="strip"></div>
    <div class="note" id="preds"></div>
  </section>

  <section class="card wide">
    <h2>browse</h2>
    <div class="ctl">
      <button id="tabLayer" class="tab on">at layer</button>
      <button id="tabTop" class="tab">top scoring</button>
      <span id="layerCtl">
        <label>layer</label><select id="bLayer"></select>
        <label>role</label><select id="bRole"></select>
        <label>site</label><select id="bSite"></select>
        <label>dictionary</label><select id="bRel"></select>
      </span>
      <span id="topCtl" hidden>
        <button id="scopeAll" class="tab on">whole prompt</button>
        <button id="scopeTok" class="tab">at token</button>
      </span>
      <label>sort</label><select id="bSort">
        <option value="total">total score</option>
        <option value="effective">effective (g·a)</option>
        <option value="act">activation (a)</option>
        <option value="gate">gate (g)</option>
      </select>
    </div>
    <div class="split" id="browseSplit">
      <div><div class="note" id="compHead"></div>
           <div class="scroll"><table id="compTab"></table></div></div>
      <div><div class="note" id="featHead"></div>
           <div class="scroll"><table id="featTab"></table></div></div>
    </div>
  </section>

  <section class="card wide">
    <h2 id="detailTitle">component / feature</h2>
    <div id="detail" class="dim">click anything in the browser, or a q/k index in the QK table</div>
  </section>

  <section class="card wide">
    <h2>interaction <span class="dim" id="intMeta"></span></h2>
    <div class="ctl">
      <button id="tabStatic" class="tab on" title="the pair viewer's weight-only score"
        >static (weights)</button>
      <button id="tabLive" class="tab" title="the same pairing on this prompt's forward"
        >on this prompt</button>
      <label>suggested</label><select id="tpl"></select>
    </div>
    <div class="ctl">
      <label>A</label>
      <select id="aKind"><option value="module">module</option><option value="sae">SAE</option></select>
      <select id="aRole"></select><select id="aSite"></select><select id="aRel"></select>
      <select id="aLay"></select>
      <select id="aSide"><option>write</option><option>read</option></select>
      <button id="pSwap" title="exchange the two endpoints and re-run everything">&#8646;</button>
      <label>B</label>
      <select id="bKind"><option value="module">module</option><option value="sae">SAE</option></select>
      <select id="bRole2"></select><select id="bSite2"></select><select id="bRel2"></select>
      <select id="bLay"></select>
      <select id="bSide"><option>read</option><option>write</option></select>
    </div>
    <div class="ctl">
      <label>metric</label><select id="pMetric"></select>
      <label>mode</label><select id="pMode">
        <option value="auto">auto</option><option value="flat">flat</option>
        <option value="head">per-head</option></select>
      <label>head</label><select id="pHead"></select>
      <label>k</label><input id="pK" type="number" value="25" min="1" max="200">
      <label title="kappa is an unconditional mean, so it rewards a partner that simply fires often"
        >density &le;</label>
      <select id="pDens">
        <option value="5e-3">5e-3</option><option value="1e-3">1e-3</option>
        <option value="1e-2">1e-2</option><option value="0">off</option></select>
      <span id="posCtl" hidden>
        <label>dest</label><select id="pDest"></select>
        <label>src</label><select id="pSrc"></select>
      </span>
    </div>
    <div class="note" id="intNote"></div>
    <div class="split" id="staticBody">
      <div class="card" id="cardA"></div>
      <div class="card" id="cardB"></div>
    </div>
    <div id="liveBody" hidden>
      <div class="note" id="liveWhere"></div>
      <div class="scroll"><table id="inter"></table></div>
    </div>
    <div class="note" id="intCaveat"></div>
  </section>

  <section class="card wide">
    <h2>attribution <span class="dim">what feeds A &mdash; every component of B, every position</span></h2>
    <div class="ctl">
      <span class="dim" id="atpEnds">press run</span>
      <label title="the position A is read at; the contributors span every position">A at token</label><select id="atpPos"></select>
      <button id="atpRun">run attribution</button>
      <label>sort</label><select id="atpSort">
        <option value="abs">|score|</option><option value="signed">score</option></select>
      <label>show</label><select id="atpTop">
        <option value="30">top 30</option><option value="100">top 100</option>
        <option value="0">all</option></select>
      <button id="atpPrev">&#9664;</button><span id="atpPage" class="dim">&mdash;</span>
      <button id="atpNext">&#9654;</button>
      <span class="dim" id="atpMeta"></span>
    </div>
    <div class="scroll"><table id="atp"></table></div>
  </section>

  <section class="card wide">
    <h2>attention &mdash; QK decomposition</h2>
    <div class="ctl">
      <label>layer</label><select id="layer"></select>
      <label>head</label><select id="head"></select>
      <span class="dim" id="attnMeta"></span>
    </div>
    <div class="read" id="attnRead"></div>
    <div class="split">
      <div><div class="note">attention probability</div><div class="grid" id="attn"></div></div>
      <div><div class="note" id="pairHead">one pair's contribution &mdash; click a QK row</div>
           <div class="grid" id="pairGrid"></div>
           <div id="headStrip"></div></div>
    </div>
    <div class="ctl">
      <label>map</label><select id="qkView">
        <option value="z">contribution (logit)</option>
        <option value="p">&Delta; probability if removed</option></select>
      <span class="dim" id="pairScale"></span>
    </div>
    <div class="note" id="qkMeta"></div>
    <div class="ctl">
      <label>q</label><input id="qkFq" size="6" placeholder="any" title="query component index">
      <label>k</label><input id="qkFk" size="6" placeholder="any" title="key component index">
      <label>kind</label><select id="qkFkind">
        <option value="pair" selected>pair</option><option value="">any</option>
        <option value="bias">bias (any of the three)</option>
        <option value="q_bias">bias query</option><option value="k_bias">bias key</option>
        <option value="bias_bias">bias &times; bias</option><option value="error">error</option>
      </select>
      <label>rows</label><select id="qkRows">
        <option value="0" selected>all</option>
        <option value="30">top 30</option><option value="100">top 100</option></select>
      <label>sort</label><select id="qkSort">
        <option value="abs">|contribution|</option>
        <option value="desc">contribution &darr;</option>
        <option value="asc">contribution &uarr;</option></select>
      <button id="qkRecon" disabled>reconstruct from 0 picked</button>
      <label title="draw the picked rows TOGETHER with every term this cell's list does not contain, at every position: full - (listed - picked). Tick every row and you get the model's own pattern back exactly; untick one and the picture differs from the model by that row alone."><input type="checkbox" id="qkWhole"> complete (fill in every other position)</label>
      <label title="exactly the same matrix as ticking the `error` row -- a shortcut, since it is the one term you usually want beside a hand-picked set"><input type="checkbox" id="qkErr"> + error node</label>
      <button id="qkClear">clear</button>
      <span class="dim" id="qkSum"></span>
    </div>
    <div class="scroll" style="max-height:420px"><table id="qk"></table></div>
  </section>
</main>
<script>
const $ = s => document.querySelector(s);
const RUN = "__RUN__";
$("#run").textContent = RUN;
let STATE = {prompt:"", pos:null, layer:0, head:0, sel:null, meta:null, layers:[],
             tab:"layer", itab:"static", scope:"all", dest:null, src:null, pairSpace:null};

const fmt = (x, n=4) => (x>=0?"+":"") + x.toFixed(n);
const cls = x => x >= 0 ? "pos" : "neg";
const q = o => Object.entries(o).filter(([,v]) => v !== null && v !== undefined && v !== "")
                     .map(([k,v]) => k + "=" + encodeURIComponent(v)).join("&");
const esc = t => String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/\n/g,"⏎");
const dot = t => esc(t).replace(/ /g,"·");
// `+ score` means removing the component LOWERS the target. Stated once, in the header tooltip.
const SIGN = "+ means ablating this lowers the target";
// The browse panel's only score column. Same sign convention as the one-hop number the other
// panels serve, but every path instead of the residual one.
const TOTAL = "multi-hop: the first-order effect of ablating this component AT THIS TOKEN, "
  + "through every downstream path. + means ablating it lowers the target";
// A density of 0 means "no harvested row", which is DEAD, not "measured as never firing" -- the
// two are the same number and only one of them is a measurement, so `null` prints as an em dash.
const pct = v => v == null ? "—" : v >= 0.01 ? (100*v).toFixed(1) + "%"
                                             : (100*v).toFixed(3) + "%";

function targetArgs(){
  const t = $("#target").value;
  const o = {prompt: STATE.prompt, target: t};
  if(t === "logit_diff"){ o.correct = $("#correct").value; o.wrong = $("#wrong").value; }
  return o;
}
$("#target").onchange = () => {
  const ld = $("#target").value === "logit_diff";
  $("#correct").hidden = !ld; $("#wrong").hidden = !ld;
};

// Everything that goes wrong in the browser also goes to the server log, so the person reading
// the job output sees the same failure as the person at the screen.
const clientLog = (msg, where) => {
  try { fetch("/api/prompt/clientlog?" + q({msg: String(msg).slice(0, 500), where}),
              {method: "POST"}).catch(() => {}); } catch(e) {}
};
window.onerror = (msg, src, line, col) => {
  clientLog(`${msg} @${line}:${col}`, "onerror");
  $("#status").innerHTML = `<span class="err">JS error: ${esc(msg)} @${line}:${col}</span>`;
  return false;
};
window.onunhandledrejection = e => {
  const m = (e.reason && e.reason.message) || e.reason;
  clientLog(m, "unhandled rejection");
  $("#status").innerHTML = `<span class="err">failed: ${esc(m)}</span>`;
};

async function get(path, args){
  const r = await fetch(path + "?" + q(args));
  if(!r.ok){
    const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
    clientLog(`${r.status} ${detail}`, path);
    throw new Error(detail);
  }
  return r.json();
}
async function busy(msg, fn){
  $("#status").textContent = msg;
  try { return await fn(); }
  catch(e){ $("#status").textContent = "error: " + e.message; throw e; }
  finally { if($("#status").textContent === msg) $("#status").textContent = ""; }
}
// `cols` are [label, isNumeric, title?] so header and body cannot drift apart.
const thead = cols => "<tr>" + cols.map(c =>
  `<th class="${c[1]?"num":""}"${c[2]?` title="${c[2]}"`:""}>${c[0]}</th>`).join("") + "</tr>";
const fill = (el, items, keep) => {
  el.innerHTML = items.map(i => `<option value="${i.value}">${i.text}</option>`).join("");
  if(items.some(i => String(i.value) === String(keep))) el.value = keep;
};
const opts = vs => vs.map(v => ({value:v, text:v}));

// ---- token strip -------------------------------------------------------------
function drawStrip(d){
  const max = Math.max(...d.tokens.map(t => t.score)) || 1;
  $("#strip").innerHTML = "";
  for(const t of d.tokens){
    const el = document.createElement("span");
    el.className = "tok"; el.dataset.pos = t.pos;
    el.style.background = `rgba(${POS_RGB},${(0.04 + 0.45*t.score/max).toFixed(3)})`;
    el.innerHTML = `<i>${t.pos}</i>${dot(t.piece)}`;
    el.title = `|score| ${t.score.toFixed(3)} · error ${t.error.toFixed(3)}`;
    el.onclick = () => selectPos(t.pos);
    $("#strip").appendChild(el);
  }
  $("#tgt").textContent = `target = ${fmt(d.target)}`;
  $("#preds").innerHTML = "next: " + d.predictions.slice(0,8)
      .map(p => JSON.stringify(p.token)).join(" ")
    + `<br><span class="dim">scored: ${d.scored_roles.join(", ")} &nbsp;|&nbsp; `
    + `no gradient path from a logit target: ${d.unreachable_roles.join(", ")} `
    + `<span class="tag">click one anyway — it still shows what it did here</span></span>`;
}

// ---- browser -----------------------------------------------------------------
const moduleAt = (layer, role) =>
  (STATE.meta.modules.find(m => m.layer === layer && m.role === role) || {}).module;
const relLayers = key => ((STATE.meta.saes || []).find(s => s.key === key) || {layers:[]}).layers;
const saeSites = () => [...new Set((STATE.meta.saes || []).map(s => s.site))];
const relsAt = site => (STATE.meta.saes || []).filter(s => s.site === site);
function fillRel(siteSel, relSel){
  const rels = relsAt($(siteSel).value);
  fill($(relSel), rels.map(r => ({value:r.key, text:r.label})), $(relSel).value);
}

function setTab(tab){
  STATE.tab = tab;
  $("#tabLayer").classList.toggle("on", tab === "layer");
  $("#tabTop").classList.toggle("on", tab === "top");
  $("#layerCtl").hidden = tab !== "layer";
  $("#topCtl").hidden = tab !== "top";
  drawBrowse();
}

// Every browse fetch carries the generation it was issued in, and a response from an older one
// is DROPPED instead of drawn. Without this a slow dictionary (a cold 128k release measured 13.9 s
// against 0.4 s for a warm 32k one) lands after the next selection and the panel shows the layer
// you left, which reads as "it did not refresh".
let BROWSE_GEN = 0;
const stale = gen => gen !== BROWSE_GEN;

async function drawBrowse(){
  if(!STATE.meta) return;
  const gen = ++BROWSE_GEN;
  if(STATE.tab === "top") return drawTop(gen);
  $("#browseSplit").style.gridTemplateColumns = "minmax(0,1fr) minmax(0,1fr)";
  $("#featHead").parentElement.hidden = false;
  const layer = +$("#bLayer").value, role = $("#bRole").value, rel = $("#bRel").value;
  const mod = moduleAt(layer, role);
  // Not `Promise.all`: each table draws the moment ITS own request lands, so a slow dictionary
  // never holds up the component list beside it.
  drawComponents(mod, gen);
  drawFeatures(rel, layer, gen);
}

async function drawComponents(mod, gen){
  if(STATE.pos === null){
    $("#compHead").innerHTML = `<span class="dim">select a token</span>`;
    $("#compTab").innerHTML = ""; return;
  }
  const d = await busy("components…", () => get("/api/prompt/browse",
      {...targetArgs(), pos: STATE.pos, module: mod, k: 500}));
  if(stale(gen)) return;
  const by = $("#bSort").value;
  const rows = [...d.rows].sort((x,y) => Math.abs(y[by] ?? 0) - Math.abs(x[by] ?? 0));
  $("#compHead").innerHTML = `<b>${mod.replace("transformer.h.","h")}</b> · ${d.n_fired} fired at `
    + `${STATE.pos} ${JSON.stringify(d.piece)}`;
  $("#compTab").innerHTML = thead([["total score",true,TOTAL],["idx",true],
      ["g·a",true],["a",true],["g",true],["density",true,
       "fraction of ALL corpus tokens this fires on — a trace of one prompt cannot say"]])
    + rows.map(r => `<tr class="click" data-m="${r.module}" data-i="${r.idx}">
        <td class="num ${cls(r.total)}">${fmt(r.total)}</td>
        <td class="num">${r.idx}</td><td class="num ${cls(r.effective)}">${fmt(r.effective,3)}</td>
        <td class="num dim">${r.act.toFixed(3)}</td>
        <td class="num dim">${r.gate.toFixed(2)}</td>
        <td class="num dim">${pct(r.density)}</td></tr>`).join("");
  wire("#compTab");
}

async function drawFeatures(rel, layer, gen){
  const have = relLayers(rel).includes(layer);
  if(!rel){ $("#featHead").innerHTML = `<span class="dim">no dictionaries for this model</span>`;
            $("#featTab").innerHTML = ""; return; }
  // The run's layers and a release's layers are different sets. Say which one is missing rather
  // than silently showing a neighbouring layer's features beside this layer's components.
  if(!have){
    $("#featHead").innerHTML = `<span class="warn">${rel} has no layer ${layer}</span>`
      + ` <span class="dim">(has ${relLayers(rel).join(", ")})</span>`;
    $("#featTab").innerHTML = ""; return;
  }
  if(STATE.pos === null){
    $("#featHead").innerHTML = `<span class="dim">select a token</span>`;
    $("#featTab").innerHTML = ""; return;
  }
  const key = `${rel}:L${layer}`;
  $("#featHead").innerHTML = `<span class="dim">loading ${key}…</span>`;
  let d;
  try { d = await busy("features…", () => get("/api/prompt/features",
          {prompt: STATE.prompt, sae: key, pos: STATE.pos, k: 500})); }
  catch(e){ if(stale(gen)) return;
            $("#featHead").innerHTML = `<span class="err">${e.message}</span>`;
            $("#featTab").innerHTML = ""; return; }
  if(stale(gen)) return;
  $("#featHead").innerHTML = `<b>${key}</b> · ${d.n_live} live of ${d.d_sae} · ${d.site}`;
  $("#featTab").innerHTML = thead([["act",true],["feature",true],["density",true,
      "only this run's own components carry a harvested density; a dictionary's comes from "
      + "Neuronpedia one feature at a time — open the card for it"]])
    + d.rows.map(r => `<tr class="click" data-m="${key}" data-i="${r.idx}">
        <td class="num">${r.act.toFixed(3)}</td><td class="num">${r.idx}</td>
        <td class="num dim">${pct(r.density)}</td></tr>`).join("");
  wire("#featTab");
}

async function drawTop(gen){
  $("#browseSplit").style.gridTemplateColumns = "minmax(0,1fr)";
  $("#featHead").parentElement.hidden = true;
  const atTok = STATE.scope === "tok" && STATE.pos !== null;
  // At a token the multi-hop score exists, so this ranks by it. Over the WHOLE prompt it does
  // not: the score is per (module, position, component), and one list mixing positions would rank
  // a component's effect here against its effect somewhere else. That view keeps the one-hop score.
  const d = atTok
    ? await busy("fired…", () => get("/api/prompt/browse", {...targetArgs(), pos: STATE.pos, k: 120}))
    : await busy("top…", () => get("/api/prompt/trace", targetArgs()));
  if(stale(gen)) return;
  const rows = atTok ? d.rows : d.top_nodes;
  $("#compHead").innerHTML = atTok
    ? `every module, fired at ${STATE.pos} ${JSON.stringify(d.piece)} — ${d.n_fired} components`
    : `highest |score| over the whole prompt <span class="dim">— one hop; pick a token for the `
      + `multi-hop ranking</span>`;
  const cols = atTok ? [["total score",true,TOTAL]] : [["score",true,SIGN]];
  const num = r => atTok
    ? `<td class="num ${cls(r.total)}">${fmt(r.total)}</td>`
    : `<td class="num ${r.score===null?"dim":cls(r.score)}">`
      + `${r.score===null?"—":fmt(r.score)}</td>`;
  $("#compTab").innerHTML = thead(cols.concat([["module",false],["idx",true],["pos",false]]))
    + rows.map(r => `<tr class="click" data-m="${r.module}" data-i="${r.idx}">
        ${num(r)}
        <td>L${r.layer} ${r.role}</td><td class="num">${r.idx}</td>
        <td class="dim">${r.pos}${r.piece?" "+JSON.stringify(r.piece):""}</td></tr>`).join("");
  wire("#compTab");
}

function wire(sel){
  for(const tr of $(sel).querySelectorAll("tr.click"))
    tr.onclick = () => showComponent(tr.dataset.m, +tr.dataset.i);
}

// ---- detail: one component or feature ---------------------------------------
const POS_RGB = "31,95,196", NEG_RGB = "192,39,31";
const heat = (v,max) => `rgba(${v>=0?POS_RGB:NEG_RGB},`
  + `${(Math.min(1,Math.abs(v)/(max||1))*0.55).toFixed(3)})`;

function exHtml(ex){
  const s = ex.series.effective || ex.series.activation, [lo,hi] = ex.window;
  const max = Math.max(...s.map(Math.abs), 1e-9);
  let h = `<div class="ex"><span class="pk">peak ${ex.peak.toFixed(3)}</span>`;
  for(let i=lo;i<hi;i++)
    h += `<span class="tok${ex.firings[i]?" fire":""}" title="${s[i].toFixed(4)}"
           style="background:${heat(s[i],max)}">${esc(ex.tokens[i])}</span>`;
  return h + "</div>";
}

// This prompt's own trace for the selected component: the picture the harvest cannot give.
function hereHtml(n){
  const e = n.effective, max = Math.max(...e.map(Math.abs), 1e-9);
  let h = `<div class="note">on this prompt &mdash; heat = ${n.gate?"effective (g·a)":"activation"}`
        + `, underline = gate open</div><div class="ex">`;
  for(let i=0;i<e.length;i++){
    const open = n.gate ? n.gate[i] > 0 : e[i] !== 0;
    const bits = [`a ${n.act[i].toFixed(4)}`];
    if(n.gate) bits.push(`g ${n.gate[i].toFixed(3)}`, `e ${e[i].toFixed(4)}`);
    if(n.score) bits.push(`score ${fmt(n.score[i])}`);
    h += `<span class="tok${open?" fire":""}" data-pos="${i}" title="${i} · ${bits.join(" · ")}"
           style="background:${heat(e[i],max)}${i===STATE.pos?";outline:1px solid var(--hi)":""}"
           >${esc(n.pieces[i])}</span>`;
  }
  return h + "</div>";
}

function massHtml(mass){
  if(!mass) return "";
  const top = mass.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,4).filter(x=>x[0]>0.02);
  return `<div class="note">write mass by head &nbsp;` + top.map(([v,i]) =>
    `<span class="badge">h${i} ${(100*v).toFixed(0)}%</span>`).join("") + "</div>";
}

const pmiHtml = (rows,title) => !rows || !rows.length ? "" :
  `<div><b>${title}</b><br>` + rows.slice(0,12).map(([t,v]) =>
    `<span class="badge">${esc(t)} ${v.toFixed(2)}</span>`).join("") + "</div>";

// u_c through the final norm and the unembedding. Present for every component whose write
// direction is d_model wide; `reason` says which width disagreed when it is not.
function lensHtml(l){
  if(!l) return "";
  if(!l.available)
    return `<div class="note warn">no logit lens — ${esc(l.reason || "unavailable")}</div>`;
  const row = (rows,title) => `<div><b>${title}</b><br>` + rows.map(([t,v]) =>
    `<span class="badge">${esc(JSON.stringify(t).slice(1,-1))} ${v.toFixed(2)}</span>`).join("")
    + `</div>`;
  return `<div class="pmi">${row(l.promoted,"logit lens ↑")}${row(l.suppressed,"logit lens ↓")}</div>`
    + (l.in_residual_stream ? ""
       : `<div class="note warn">this write side is not the residual stream — the arithmetic is
          defined but the model never carries this vector to <code>ln_f</code> without putting it
          through the attention machinery first</div>`);
}

// One card, three places: the detail panel and both static-tab endpoints. Split out so the panels
// cannot drift into showing different things about the same component.
function navHtml(tag, idx, n, isSae, d){
  const lim = n - 1;
  return `<div class="ctl">`
    + `<button data-d="-1"${idx<=0?" disabled":""}>&#9664;</button>`
    + `<input class="idxin" type="number" value="${idx}" min="0" max="${lim}">`
    + `<button data-d="1"${idx>=lim?" disabled":""}>&#9654;</button>`
    + `<span class="dim">of ${n} ${isSae?"features":"components"}</span>`
    + (isSae && d.neuronpedia_url
        ? ` <a href="${d.neuronpedia_url}" target="_blank" style="color:var(--hi)">neuronpedia &#8599;</a>` : "")
    + (tag === "detail"
        ? ` <button data-as="a" title="point endpoint A at this">&rarr; A</button>`
        + `<button data-as="b" title="point endpoint B at this">&rarr; B</button>` : "")
    + `</div>`;
}

function cardBody(d, n){
  const isSae = n.kind === "sae";
  let h = `<div class="dim">L${n.layer} ${n.role}`
    + (d.read_space ? ` · read ${d.read_space.label} → write ${d.write_space.label}` : "") + `</div>`;
  const meta = [];
  if(d.firing_density != null) meta.push(`density ${(100*d.firing_density).toFixed(3)}%`);
  if(d.read_norm != null) meta.push(`|${isSae?"enc":"read"}| ${d.read_norm.toFixed(3)}`);
  if(d.write_norm != null) meta.push(`|${isSae?"dec":"write"}| ${d.write_norm.toFixed(3)}`);
  if(d.enc_dec_cosine != null) meta.push(`enc·dec ${d.enc_dec_cosine.toFixed(3)}`);
  if(d.max_activation != null) meta.push(`max act ${d.max_activation.toFixed(2)}`);
  if(d.n_batches_not_active != null) meta.push(`dead-clock ${d.n_batches_not_active}`);
  if(d.centred) meta.push(`<span class="warn">centred basis</span>`);
  if(d.label) h += `<div style="margin:4px 0"><span class="badge"
      style="border-color:var(--hi)">${esc(d.label.label ?? d.label)}</span></div>`;
  if(meta.length) h += `<div class="note">${meta.join(" · ")}</div>`;
  h += massHtml(n.head_mass);
  if(n.unreachable) h += `<div class="note warn">${n.reason
      || "no gradient path from a logit target to this module — score is null, not zero"}</div>`;
  h += hereHtml(n);
  h += lensHtml(d.logit_lens);
  if(d.input_pmi || d.output_pmi)
    h += `<div class="pmi">${pmiHtml(d.input_pmi,"input PMI")}${pmiHtml(d.output_pmi,"output PMI")}</div>`;
  if(d.pmi_available === false) h += `<div class="note warn">no token PMI in this harvest</div>`;
  if(d.explanations_dropped && d.explanations_dropped.length)
    h += `<div class="note dim">${d.explanations_dropped.length} duplicate autointerp`
       + ` description${d.explanations_dropped.length>1?"s":""} hidden</div>`;
  if(d.reason) h += `<div class="note warn">${esc(d.reason)}</div>`;
  if(d.examples && d.examples.length){
    h += `<div class="note">${d.examples.length} activating contexts from the harvest`
       + ` <span class="dim">(scroll sideways)</span></div>`
       + d.examples.slice(0,12).map(exHtml).join("");
  }
  return h;
}

async function showComponent(module, idx){
  clientLog(`${module}:${idx}`, "showComponent");
  $("#detailTitle").textContent = `${module.replace("transformer.h.","h")}:${idx}`;
  $("#detail").innerHTML = `<span class="dim">loading…</span>`;
  let d, n;
  try { [d, n] = await busy("component…", () => Promise.all([
    get("/api/component", {module, idx, window:12}),
    get("/api/prompt/node", {...targetArgs(), module, idx}),
  ])); }
  catch(e){ $("#detail").innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  STATE.sel = {module, idx, kind:n.kind, role:n.role, layer:n.layer};
  const lim = n.n_components - 1;
  $("#detail").innerHTML = navHtml("detail", idx, n.n_components, n.kind === "sae", d) + cardBody(d, n);
  for(const b of $("#detail").querySelectorAll("[data-d]"))
    b.onclick = () => showComponent(module, Math.max(0, Math.min(lim, idx + +b.dataset.d)));
  $("#detail").querySelector(".idxin").onchange = e =>
    showComponent(module, Math.max(0, Math.min(lim, +e.target.value)));
  for(const t of $("#detail").querySelectorAll(".ex .tok[data-pos]"))
    t.onclick = () => selectPos(+t.dataset.pos);
  // Opening a card USED to call `selectA`, which re-pointed endpoint A at whatever was clicked --
  // in the browse table, in the static ranking and in the prompt ranking alike. That silently
  // replaced the pairing the reader had set up, so the ranking they clicked into was no longer the
  // ranking they were reading. The card is now inert and the two buttons below are the only way to
  // move an endpoint.
  $("#detail").querySelector("[data-as=a]").onclick =
    () => selectEndpoint("a", module, idx, n.role, n.layer, n.kind);
  $("#detail").querySelector("[data-as=b]").onclick =
    () => selectEndpoint("b", module, idx, n.role, n.layer, n.kind);
}

async function selectPos(pos){
  STATE.pos = pos;
  if(STATE.dest === null || STATE.dest < pos) setPositions(pos, STATE.src);
  for(const el of document.querySelectorAll("#strip .tok"))
    el.classList.toggle("sel", +el.dataset.pos === pos);
  await drawBrowse();
  drawTab();
}

// ---- interaction: the pair viewer, plus the same pairing on this forward -----
// Two tabs over ONE pairing. `static` is weights only -- what could compose, ever. `on this
// prompt` is what actually did, and it obeys the forward pass: a side is read at the position the
// MODE says it is read at, and a side that did not fire there contributes exactly nothing.
const SEL = {A: 0, B: 0};
let LIVE = null;
let PS = null;

const epRole = w => $("#" + w + "Role" + (w === "b" ? "2" : "")).value;
const epKey = w => $("#" + w + "Kind").value === "sae"
  ? `${$("#" + w + "Rel" + (w === "b" ? "2" : "")).value}:L${$("#" + w + "Lay").value}`
  : moduleAt(+$("#" + w + "Lay").value, epRole(w));
const epSide = w => $("#" + w + "Side").value;
const isSaeEp = w => $("#" + w + "Kind").value === "sae";
const sfx = w => w === "b" ? "2" : "";

function syncEp(w){
  const sae = isSaeEp(w), x = sfx(w);
  $("#" + w + "Role" + x).hidden = sae;
  $("#" + w + "Site" + x).hidden = !sae;
  $("#" + w + "Rel" + x).hidden = !sae;
  fillEpLayers(w, +$("#" + w + "Lay").value);
}
function fillEpLayers(w, keep){
  const ls = isSaeEp(w) ? relLayers($("#" + w + "Rel" + sfx(w)).value) : STATE.layers;
  fill($("#" + w + "Lay"), opts(ls), keep);
}

// The pair viewer's own template list, so the two pages agree on what pairs with what.
const tplList = () => (STATE.meta.templates || []).map(t => ({...t, a: t.a_role, b: t.b_role}))
  .concat(STATE.meta.sae_templates || []);

function setEp(w, name, layer, side){
  const x = sfx(w), sae = String(name).startsWith("sae:");
  $("#" + w + "Kind").value = sae ? "sae" : "module";
  syncEp(w);
  if(sae){ $("#" + w + "Site" + x).value = name.slice(4); fillRel("#" + w + "Site" + x, "#" + w + "Rel" + x); }
  else { $("#" + w + "Role" + x).value = name; }
  fillEpLayers(w, layer);
  if(side) $("#" + w + "Side").value = side;
}

function applyTemplate(){
  const t = tplList().find(x => x.key === $("#tpl").value);
  if(!t) return refreshPair();
  const la = t.layer_mode === "same" ? t.layers.same : t.layers.a;
  const lb = t.layer_mode === "same" ? t.layers.same : t.layers.b;
  const pa = la.includes(+$("#aLay").value) ? +$("#aLay").value : la[0];
  const pb = t.layer_mode === "same" ? pa : (lb.includes(+$("#bLay").value) ? +$("#bLay").value : lb[0]);
  setEp("a", t.a, pa, t.a_side);
  setEp("b", t.b, pb, t.b_side);
  refreshPair();
}

// Point ONE endpoint at a component or feature -- endpoint and index together. Only `a` pulls a
// template across (so B follows what you are looking at); pointing B somewhere must leave A alone,
// which is the whole reason this takes a slot instead of always meaning A.
async function selectEndpoint(slot, key, idx, role, layer, kind){
  const sae = kind === "sae", name = sae ? "sae:" + role : role;
  const t = slot !== "a" ? null
          : (tplList().find(x => x.a === name && x.a_side === $("#aSide").value)
             || tplList().find(x => x.a === name));
  if(t){
    $("#tpl").value = t.key;
    const la = t.layer_mode === "same" ? t.layers.same : t.layers.a;
    const lb = t.layer_mode === "same" ? t.layers.same : t.layers.b;
    const pa = la.includes(layer) ? layer : la[0];
    setEp("a", t.a, pa, t.a_side);
    setEp("b", t.b, t.layer_mode === "same" ? pa : lb[0], t.b_side);
  } else {
    setEp(slot, name, layer, sae ? "write" : $("#" + slot + "Side").value);
  }
  SEL[slot.toUpperCase()] = idx;
  await refreshPair();
}

// Changing an index redraws the CARD for that side -- always, on either tab, which is what the
// app did before this function existed -- and then the prompt table too when that tab is showing.
// Making these exclusive was the regression: on the prompt tab the card, its activation examples
// and its ranking all stopped updating, and only the "did not fire" line moved.
function selChanged(which){
  clientLog(`${which || "-"} A=${SEL.A} B=${SEL.B} itab=${STATE.itab}`, "selChanged");
  if(which) drawSide(which); else drawStatic();
  if(STATE.itab === "live") drawLive();
}

function liveMeta(){
  if(!LIVE || LIVE.error || LIVE.reason) return;
  $("#intMeta").innerHTML = `A:${SEL.A} &rarr; B:${SEL.B}`
    + ` &middot; ${LIVE.n_live_b} live partners at ${LIVE.pos} ${JSON.stringify(LIVE.piece)}`;
}

// Exchange the two endpoints, indices included, and re-run. The swap is done on the CONTROLS, so
// whatever `refreshPair` makes of the new pairing -- link, mode, per-head availability -- is
// derived fresh rather than transposed from the old answer.
async function swapEndpoints(){
  const read = w => ({kind: $("#" + w + "Kind").value, role: $("#" + w + "Role" + sfx(w)).value,
                      site: $("#" + w + "Site" + sfx(w)).value, rel: $("#" + w + "Rel" + sfx(w)).value,
                      lay: $("#" + w + "Lay").value, side: $("#" + w + "Side").value});
  const a = read("a"), b = read("b");
  const put = (w, v) => {
    const x = sfx(w);
    $("#" + w + "Kind").value = v.kind;
    syncEp(w);
    if(v.kind === "sae"){
      $("#" + w + "Site" + x).value = v.site;
      fillRel("#" + w + "Site" + x, "#" + w + "Rel" + x);
      $("#" + w + "Rel" + x).value = v.rel;
    } else {
      $("#" + w + "Role" + x).value = v.role;
    }
    fillEpLayers(w, +v.lay);
    $("#" + w + "Lay").value = v.lay;
    $("#" + w + "Side").value = v.side;
  };
  put("a", b); put("b", a);
  [SEL.A, SEL.B] = [SEL.B, SEL.A];
  $("#tpl").value = "";
  await refreshPair();
}

function setPositions(dest, src){
  const n = STATE.pieces ? STATE.pieces.length : 0;
  if(!n) return;
  STATE.dest = dest === null || dest === undefined ? n - 1 : Math.max(0, Math.min(n - 1, dest));
  const items = STATE.pieces.map((p,i) => ({value:i, text:`${i} ${dot(p).slice(0,10)}`}));
  fill($("#pDest"), items, STATE.dest);
  // src <= dest, always: attention is causal, so a later source is not a quantity the model forms.
  STATE.src = (src === null || src === undefined || src > STATE.dest) ? null : src;
  fill($("#pSrc"), [{value:"", text:"— pick one —"}].concat(items.slice(0, STATE.dest + 1)),
       STATE.src === null ? "" : STATE.src);
}

// Mirrors `interact._mode` exactly. Which side is read where depends on it, and so does whether a
// source position exists at all. Only the two pairings that ARE attention route through a pattern;
// an o:write × v:read across layers, or any v × o that is not the one-layer OV template, is a
// residual composition at one position and is drawn as `same_pos` rather than refused.
function pairMode(ps){
  if(ps.link === "bilinear_form") return "bilinear";
  const rs = ps.a.role + ":" + ps.a.side, ss = ps.b.role + ":" + ps.b.side;
  const ov = (rs === "attn.v:write" && ss === "attn.o:read")
          || (rs === "attn.o:read" && ss === "attn.v:write");
  if(ov && ps.a.layer === ps.b.layer) return "routed";
  return "same_pos";
}
// Mirrors `interact._free_side`: the side read at `src` is the KEY for bilinear and the VALUE for
// routed, whichever endpoint holds that role -- not whichever was named first.
function readsAt(ps, mode){
  if(mode === "same_pos") return {a: "dest", b: "dest"};
  const want = mode === "bilinear" ? "attn.k" : "attn.v";
  return ps.a.role === want ? {a: "src", b: "dest"} : {a: "dest", b: "src"};
}

async function refreshPair(){
  const pair = {a_module: epKey("a"), a_side: epSide("a"),
                b_module: epKey("b"), b_side: epSide("b")};
  if(!pair.a_module || !pair.b_module){
    $("#intNote").innerHTML = `<span class="warn">this run has no such module at that layer</span>`;
    return;
  }
  try { PS = await get("/api/pairspace", pair); }
  catch(e){ $("#intNote").innerHTML = `<span class="err">${e.message}</span>`; return; }
  const c = PS.compat;
  fill($("#pMetric"), PS.metrics.map(m => ({value:m.key, text:m.label + (m.available?"":" — no data")})),
       $("#pMetric").value);
  if(!$("#pMetric").value || !(PS.metrics.find(m => m.key === $("#pMetric").value) || {}).available){
    const first = PS.metrics.find(m => m.available); if(first) $("#pMetric").value = first.key;
  }
  const perHead = $("#pMode").value === "head" || ($("#pMode").value === "auto" &&
      ((PS.link === "bilinear_form" && c.per_head) || (!c.flat && c.per_head)));
  $("#pHead").disabled = !perHead;
  fill($("#pHead"), [{value:"", text: perHead ? "best of " + c.n_head_pairs : "no head"}].concat(
      Array.from({length: perHead ? c.n_head_pairs : 0}, (_,i) => ({value:i, text:"head "+i}))),
      $("#pHead").value);
  SEL.A = Math.min(SEL.A, PS.a.n_components - 1);
  SEL.B = Math.min(SEL.B, PS.b.n_components - 1);
  let note = `<b>${PS.a.space.label}</b> → <b>${PS.b.space.label}</b> · link `
    + (PS.link_label || `<span class="warn">unclassified</span>`)
    + ` · flat ${c.flat?"yes":"no"} · per-head ${c.per_head? c.n_head_pairs+" pairs":"no"}`
;
  if(PS.basis && PS.basis.corrections && PS.basis.corrections.length)
    note += `<br><span class="badge" style="border-color:var(--hi)">basis</span> `
          + PS.basis.corrections.join(" · ");
  if(PS.link === "bilinear_form" && !perHead && c.flat)
    note += `<br><span class="err">a flat score on a QK pair sums the heads' logits, which the `
          + `model never forms — switch mode to per-head.</span>`;
  if(!c.flat && !c.per_head) note += `<br><span class="err">${c.reason}</span>`;
  $("#intNote").innerHTML = note;
  $("#intCaveat").textContent = PS.caveat || "";
  drawTab();
}

function setITab(tab){
  STATE.itab = tab;
  $("#tabStatic").classList.toggle("on", tab === "static");
  $("#tabLive").classList.toggle("on", tab === "live");
  $("#staticBody").hidden = tab !== "static";
  $("#liveBody").hidden = tab !== "live";
  $("#posCtl").hidden = tab !== "live";
  drawTab();
}
const drawTab = () => STATE.itab === "live" ? drawLive() : drawStatic();

// ---- static tab: the pair viewer, both cards and both rankings ---------------
async function drawStatic(){
  if(!PS) return;
  await Promise.all([drawSide("A"), drawSide("B")]);
}

async function drawSide(which){
  const side = which === "A" ? PS.a : PS.b, el = $("#card" + which);
  const n = side.n_components;
  el.innerHTML = `<h2>${which} · ${side.label || side.module} · side ${side.side}</h2>`
               + `<div class="note">loading…</div>`;
  let d, node;
  try { [d, node] = await Promise.all([
    get("/api/component", {module: side.module, idx: SEL[which], window: 12}),
    get("/api/prompt/node", {...targetArgs(), module: side.module, idx: SEL[which]}),
  ]); }
  catch(e){ el.innerHTML = `<h2>${which}</h2><div class="err">${e.message}</div>`; return; }
  el.innerHTML = `<h2>${which} · ${side.label || side.module} · `
    + `<span style="color:var(--fg)">side ${side.side}</span></h2>`
    + navHtml(which, SEL[which], n, node.kind === "sae", d)
    + cardBody(d, node)
    + `<div id="rank${which}"><div class="note">ranking…</div></div>`;
  for(const b of el.querySelectorAll("[data-d]"))
    b.onclick = () => { SEL[which] = Math.max(0, Math.min(n-1, SEL[which] + +b.dataset.d));
                        selChanged(which); };
  el.querySelector(".idxin").onchange = e => {
    SEL[which] = Math.max(0, Math.min(n-1, +e.target.value)); selChanged(which); };
  for(const t of el.querySelectorAll(".ex .tok[data-pos]"))
    t.onclick = () => selectPos(+t.dataset.pos);
  drawRank(which);
}

async function drawRank(which){
  const from = which, to = which === "A" ? "B" : "A";
  const e = {a_module: epKey("a"), a_side: epSide("a"), b_module: epKey("b"), b_side: epSide("b")};
  const args = from === "A" ? e
    : {a_module: e.b_module, a_side: e.b_side, b_module: e.a_module, b_side: e.a_side};
  const el = $("#rank" + which); if(!el) return;
  let r;
  try { r = await get("/api/rank", {...args, idx: SEL[from], metric: $("#pMetric").value,
        mode: $("#pMode").value, head: $("#pHead").value, k: $("#pK").value,
        max_density: $("#pDens").value}); }
  catch(err){ el.innerHTML = `<div class="err">${err.message}</div>`; return; }
  // dot_coact is a product with kappa ~1e-5: a fixed 4-decimal format prints every score as 0.0000.
  const big = [...r.top, ...r.bottom].some(x => Math.abs(x.score) >= 1e-3);
  const sf = v => big ? fmt(v) : (v>=0?"+":"") + v.toExponential(2);
  const hd = thead([[to + " idx", true], ["score", true], ["z(null)", true, "in units of the null sd, 1/√d_eff"],
                    ["z(row)", true, "in sd of this row"], ["head", true]]);
  const body = l => l.map(x => `<tr class="click" data-to="${to}" data-i="${x.idx}">
      <td class="num">${x.idx}</td><td class="num ${cls(x.score)}">${sf(x.score)}</td>
      <td class="num dim">${x.z_theory!=null?x.z_theory.toFixed(1)+"σ":"—"}</td>
      <td class="num dim">${x.z_empirical!=null?x.z_empirical.toFixed(1):"—"}</td>
      <td class="num dim">${x.head!==undefined&&x.head!==null?"h"+x.head:"—"}</td></tr>`).join("");
  let foot = `<div class="note">partners in ${to} · ${r.source}`
    + (r.per_head ? ` · per-head (d=${r.d_eff})` : ` · flat (d=${r.d_eff})`)
    + ` · row mean ${r.row_mean!=null?sf(r.row_mean):"—"} sd ${r.row_std!=null?sf(r.row_std):"—"}`;
  if(r.n_covered != null && r.n_components != null && r.n_covered < r.n_components)
    foot += ` · <span class="warn">ranked over ${r.n_covered} of ${r.n_components}`
          + ` (excluded, not scored 0)</span>`;
  if(r.density_note) foot += ` · <span class="warn">${r.density_note}</span>`;
  foot += "</div>";
  if(r.shared_gate) foot += `<div class="note warn">shared encoder — both modules read one gate on `
    + `this arm, so kappa here measures gate sharing as much as composition.</div>`;
  el.innerHTML = foot + `<div class="split"><div><table>${hd}${body(r.top)}</table></div>`
    + `<div><table>${hd}${body(r.bottom)}</table></div></div>`;
  for(const tr of el.querySelectorAll("tr[data-i]"))
    tr.onclick = () => { SEL[tr.dataset.to] = +tr.dataset.i; selChanged(tr.dataset.to); };
}

// ---- prompt tab: only what this forward actually composes --------------------
async function drawLive(){
  if(!PS) return;
  const mode = pairMode(PS), reads = readsAt(PS, mode);
  // An MLP or residual pairing reads both sides at ONE position. There is no source to choose, so
  // the control is pinned to the destination rather than offering a choice with no meaning.
  if(mode === "same_pos"){
    if(STATE.src !== STATE.dest) setPositions(STATE.dest, STATE.dest);
    $("#pSrc").disabled = true;
  } else {
    $("#pSrc").disabled = false;
  }
  const usingHead = $("#pHead").value !== "" ? $("#pHead").value : STATE.head;
  const where = `<span class="badge">mode ${mode}</span> `
    + `A is read at the <b>${reads.a}</b> token, B at the <b>${reads.b}</b> token`
    + (mode === "same_pos" ? ` — one position, so src is pinned to dest`
                           : ` &middot; head <b>h${usingHead}</b>`);
  if(mode !== "same_pos" && STATE.src === null){
    $("#liveWhere").innerHTML = where;
    $("#inter").innerHTML = "";
    $("#intMeta").innerHTML = `<span class="warn">pick a source token — or click a cell in the `
      + `attention pattern, which sets both ends at once</span>`;
    return;
  }
  // A per-head pairing has no flat answer -- the model forms one logit per head. "best of 12" is a
  // weight-only notion, so on this prompt the head falls back to the one the pattern panel is
  // showing rather than 422-ing on a missing argument.
  const head = $("#pHead").value !== "" ? $("#pHead").value
             : (mode === "same_pos" ? "" : String(STATE.head));
  LIVE = null;
  const live = await busy("interaction…", () => get("/api/prompt/interact",
      {prompt: STATE.prompt, a: epKey("a"), a_side: epSide("a"),
       b: epKey("b"), b_side: epSide("b"), a_idx: SEL.A, pos: STATE.dest,
       src: STATE.src === null ? "" : STATE.src,
       head, k: 200}).catch(e => ({error: e.message})));
  LIVE = live.error || live.reason ? null : live;
  $("#liveWhere").innerHTML = where
    + (live.factors ? ` &middot; factors a=<code>${live.factors.a}</code> `
                    + `b=<code>${live.factors.b}</code>` : "");
  if(live.error){
    $("#inter").innerHTML = ""; $("#intMeta").innerHTML = `<span class="err">${live.error}</span>`;
    return;
  }
  // An empty table always says why: the component did not fire where this pairing reads it, or
  // there is no cross-position term at all.
  if(live.reason){
    $("#inter").innerHTML = "";
    $("#intMeta").innerHTML = `<span class="warn">${live.reason}</span>`;
    return;
  }
  liveMeta();
  $("#inter").innerHTML = thead([["B idx", true], ["score", true,
        "e_a · ⟨dir_a, dir_b⟩ · factor_b — what this pairing did on this forward"],
      ["geometry", true, "the weight-only dot, the same number the static tab ranks by"],
      ["src", true, "position the source side was read at"]])
    + live.rows.map(r => `<tr class="click" data-i="${r.b_idx}">
        <td class="num">${r.b_idx}</td>
        <td class="num ${cls(r.score)}">${fmt(r.score)}</td>
        <td class="num dim">${fmt(r.geometry)}</td>
        <td class="num dim">${r.src != null ? r.src : "—"}</td></tr>`).join("");
  // A row here is a partner in B, so clicking it moves B -- exactly what the static ranking does.
  // It used to call `showComponent`, which re-pointed endpoint A at the row and destroyed the
  // pairing that produced the row in the first place.
  for(const tr of $("#inter").querySelectorAll("tr.click"))
    tr.onclick = () => { SEL.B = +tr.dataset.i; liveMeta();
                         drawSide("B");                      // the card, examples and ranking
                         showComponent(epKey("b"), SEL.B); };
}

// ---- attribution ---------------------------------------------------------------
// Self-contained on purpose. Nothing in the interaction panel calls into this, and this calls
// nothing there except to READ the two endpoint selects when the button is pressed. An earlier
// version hooked into `refreshPair`, so a fault here took the whole interaction panel down with
// it -- which is the one thing this panel must never be able to do.
let ATP = null;
const atpLimit = () => +$("#atpTop").value || 200;

function drawAtp(){
  const meta = $("#atpMeta"), tbl = $("#atp"), pg = $("#atpPage");
  if(!ATP){ tbl.innerHTML = ""; meta.innerHTML = ""; pg.textContent = "—"; return; }
  if(ATP.reason){
    tbl.innerHTML = ""; pg.textContent = "—";
    meta.innerHTML = `<span class="warn">${esc(ATP.reason)}</span>`; return;
  }
  const lim = atpLimit(), pages = Math.max(1, Math.ceil(ATP.n_total / lim));
  pg.textContent = `${Math.floor(ATP.offset / lim) + 1} / ${pages}`;
  meta.innerHTML = `${ATP.n_total} of ${ATP.n_components}×${ATP.n_pos} (idx, pos) reached`;
  tbl.innerHTML = thead([["B idx", true], ["pos", true], ["token"],
      ["score", true, "∂a_A/∂a_b · a_b — grad × act of the contributor"],
      ["a", true], ["g", true], ["e = g·a", true],
      ["density", true, "share of this prompt's positions where its gate is open"]])
    + ATP.rows.map(r => `<tr class="click" data-i="${r.idx}">
        <td class="num">${r.idx}</td><td class="num dim">${r.pos}</td>
        <td>${esc(dot(r.piece))}</td><td class="num ${cls(r.score)}">${fmt(r.score)}</td>
        <td class="num dim">${fmt(r.act)}</td><td class="num dim">${fmt(r.gate)}</td>
        <td class="num dim">${fmt(r.effective)}</td>
        <td class="num dim">${(100*r.density).toFixed(1)}%</td></tr>`).join("");
  // A row is a component of B, so clicking it moves B -- the same thing a prompt-match row does.
  for(const tr of $("#atp").querySelectorAll("tr.click"))
    tr.onclick = () => { SEL.B = +tr.dataset.i; drawSide("B");
                         showComponent(epKey("b"), SEL.B); };
}

async function runAtp(offset){
  if(!STATE.pieces || !STATE.pieces.length) return;
  if($("#atpPos").options.length !== STATE.pieces.length)
    fill($("#atpPos"), STATE.pieces.map((p,i) => ({value:i, text:`${i} ${dot(p).slice(0,10)}`})),
         STATE.dest == null ? STATE.pieces.length - 1 : STATE.dest);
  const a = epKey("a"), b = epKey("b");
  if(!a || !b){ ATP = {reason: "pick two modules in the interaction panel first"};
                drawAtp(); return; }
  $("#atpEnds").textContent = `backward from ${a.replace("transformer.h.","h")}:${SEL.A}`
    + ` → contributors in ${b.replace("transformer.h.","h")}`;
  ATP = await busy("attribution…", () => get("/api/prompt/atp",
      {prompt: STATE.prompt, a, b, a_idx: SEL.A, pos: +$("#atpPos").value,
       sort: $("#atpSort").value, offset: offset || 0, limit: atpLimit()})
    .catch(e => ({reason: e.message})));
  drawAtp();
}

// ---- attention ---------------------------------------------------------------
let ATTN = null;
const HINT = `<span class="dim">hover a cell to read it; click to decompose that pair</span>`;

function drawAttn(d){
  ATTN = d;
  const n = d.pieces.length, a = d.attn;
  let h = `<table><tr class="axis"><th></th>`
        + d.pieces.map((p,i)=>`<th class="c">${i} ${dot(p).slice(0,8)}</th>`).join("") + "</tr>";
  for(let t=0;t<n;t++){
    h += `<tr><th class="r">${t} ${dot(d.pieces[t]).slice(0,9)}</th>`;
    for(let k=0;k<n;k++)
      h += k>t ? `<td style="background:var(--mask)"></td>`
               : `<td data-t="${t}" data-k="${k}"
                    style="background:rgba(${POS_RGB},${(0.85*a[t][k]).toFixed(3)})"></td>`;
    h += "</tr>";
  }
  $("#attn").innerHTML = h + "</table>";
  for(const td of $("#attn").querySelectorAll("td[data-t]")){
    // A grid cell IS a (dest, src) pair, so one click sets both ends and the prompt tab has
    // everything it needs -- which is the only way to get a source for an attention pairing
    // without hunting through the token dropdown.
    td.onclick = () => { setPositions(+td.dataset.t, +td.dataset.k);
                         if(STATE.itab === "live") drawTab();
                         loadQK(+td.dataset.t, +td.dataset.k); };
    td.onmouseenter = () => readout(+td.dataset.t, +td.dataset.k);
  }
  $("#attn").onmouseleave = () => { $("#attnRead").innerHTML = HINT; };
  $("#attnRead").innerHTML = HINT;
  const bad = d.recon_error > 1e-4;
  $("#attnMeta").innerHTML =
    `residual ${(100*d.residual).toFixed(1)}% <span class="tag">error node's share of the pattern</span>` +
    (bad ? ` <span class="warn">recon error ${d.recon_error.toExponential(1)}</span>` : "");
}

// `logit` is spelled out because `z(null)` / `z(row)` in the interaction panel are z-SCORES, and
// one page carrying both would invite reading this pre-softmax score as a number of sigmas.
function readout(t, k){
  const d = ATTN;
  $("#attnRead").innerHTML =
      `<b>${t}</b> ${JSON.stringify(d.pieces[t])} <span class="dim">attends to</span> `
    + `<b>${k}</b> ${JSON.stringify(d.pieces[k])}`
    + ` &nbsp;·&nbsp; p = <b>${d.attn[t][k].toFixed(4)}</b>`
    + ` &nbsp;·&nbsp; logit = <b class="${cls(d.z[t][k])}">${fmt(d.z[t][k], 3)}</b>`
    + ` <span class="dim">(pre-softmax, q·k/&radic;d_h)</span>`;
}

// The pair's own [query, key] map, drawn to the SAME shape as the pattern beside it so the two can
// be read against each other: where this pair pushes, and where the head actually attends.
// One grid renderer for both halves of the attention row, so a subset's pattern and the model's
// are drawn to the same scale and can be compared by eye rather than by tooltip.
function heatGrid(sel, pieces, m, max, tip){
  const n = pieces.length;
  let h = `<table><tr class="axis"><th></th>`
        + pieces.map((p,i)=>`<th class="c">${i} ${dot(p).slice(0,8)}</th>`).join("") + "</tr>";
  for(let t=0;t<n;t++){
    h += `<tr><th class="r">${t} ${dot(pieces[t]).slice(0,9)}</th>`;
    for(let k=0;k<n;k++)
      h += k>t ? `<td style="background:var(--mask)"></td>`
               : `<td title="${t}→${k}  ${m[t][k].toExponential(3)}${tip ? "  " + tip(t,k) : ""}"
                    style="background:${heat(m[t][k],max)}"></td>`;
    h += "</tr>";
  }
  $(sel).innerHTML = h + "</table>";
}

const pairName = r =>
    r.kind === "pair" ? `q:${r.q_idx} × k:${r.k_idx}`
  : r.kind === "k_bias" ? `q:${r.q_idx} × key bias`
  : r.kind === "q_bias" ? `query bias × k:${r.k_idx}`
  : r.kind === "bias_bias" ? "query bias × key bias" : "error node";

// THE map is drawn against the HEAD's own |Z|, never against its own maximum. A pair's map is
// rank-1 -- `scale · ⟨u_q[c]_h, u_k[c']_h⟩ · outer(e_q, e_k)` -- so the head enters only as that
// leading scalar, and normalising each map to itself divides out the single quantity that differs
// between heads. Measured: L4 q5740×k341 is +3.82 on head 11 and −0.06 on head 10, the same
// picture rescaled by −62.8. On the head scale head 10 correctly renders as nearly blank.
// Δp is drawn on a FIXED 0..1 scale and its tooltip carries both probabilities. A bare difference
// is unreadable: where the head ignores a key, p = 0.000 and p_off = 0.700 gives −0.700, which
// looks like a probability pushed below zero until the two numbers sit side by side. The sign
// convention, stated on screen rather than left to be inferred: this is p(with) − p(without), so
// NEGATIVE means the head would attend there MORE if this row were removed.
function drawPairGrid(d, r){
  const useP = $("#qkView").value === "p";
  const m = useP ? d.attn_true.map((row,t) => row.map((p,k) => p - d.attn_off[t][k])) : d.matrix;
  $("#pairHead").innerHTML =
      `${pairName(r)} — ${useP ? "effect on the pattern, p(with) − p(without this row)"
                               : "contribution to the logit, un-summed"}`
    + `, rows = query, cols = key`
    + (KIND_NOTE[r.kind] ? `<br><span class="warn">${KIND_NOTE[r.kind]}</span>` : "");
  const peakP = Math.max(...m.flat().map(Math.abs), 0);
  $("#pairScale").innerHTML = useP
    ? `largest |Δp| <b>${peakP.toFixed(3)}</b>, drawn on the full 0–1 probability scale`
      + ` <span class="warn">— negative means the head would attend there MORE without this row;`
      + ` hover for both probabilities</span>`
    : `peak |contribution| <b>${d.peak.toExponential(2)}</b> drawn against this head's`
      + ` max |Z| <b>${d.z_max.toExponential(2)}</b>`
      + ` <span class="dim">— same scale on every head, so a weak pair looks weak</span>`;
  if(useP) heatGrid("#pairGrid", d.pieces, m, 1,
                    (t,k) => `p ${d.attn_true[t][k].toFixed(4)} → ${d.attn_off[t][k].toFixed(4)} without`);
  else heatGrid("#pairGrid", d.pieces, m, d.z_max);
  drawHeadStrip(d, r);
}

// Which head does this row belong to. The map cannot say -- it is identical on all of them -- so
// this is the contribution to Z[t, t_key] head by head, and clicking a cell moves there.
function drawHeadStrip(d, r){
  const el = $("#headStrip");
  if(!d.per_head){ el.innerHTML = ""; return; }
  const max = Math.max(...d.per_head.map(Math.abs), 1e-12);
  el.className = "hstrip";
  el.innerHTML = `<b title="this row's contribution to the logit at the picked cell, on every head`
    + ` of this layer">at [${d.at.t},${d.at.t_key}] per head</b>`
    + d.per_head.map((v,h) => `<span class="${h===d.head?"here":""}" data-h="${h}"
         style="background:${heat(v,max)}" title="head ${h}: ${v.toExponential(3)}">${h}<i
         >${fmt(v,2)}</i></span>`).join("");
  for(const sp of el.querySelectorAll("span"))
    sp.onclick = () => { if(+sp.dataset.h === STATE.head) return;
      STATE.head = +sp.dataset.h; $("#head").value = STATE.head;
      loadAttn().then(() => loadQK(d.at.t, d.at.t_key)); };
}

// Every row kind has a map, bias rows included -- those are exactly the rows whose picture
// settles what they do, and they had a number and no picture before.
async function showPair(r){
  PAIR = null; SUBSET = null;
  const d = await busy("pair…", () => get("/api/prompt/qk_pair",
      {prompt: STATE.prompt, layer: STATE.layer, head: STATE.head,
       kind: r.kind, c_q: r.q_idx, c_k: r.k_idx,
       t: QK ? QK.t : null, t_key: QK ? QK.t_key : null}));
  PAIR = {d, r};
  drawPairGrid(d, r);
}

async function loadAttn(){
  const d = await busy("attention…", () => get("/api/prompt/attn",
      {prompt: STATE.prompt, layer: STATE.layer, head: STATE.head}));
  drawAttn(d);
  QK = null; PAIR = null; SUBSET = null;
  $("#qk").innerHTML = ""; $("#qkSum").textContent = "";
  $("#qkMeta").textContent = "click a cell in the pattern for its decomposition";
  $("#pairGrid").innerHTML = ""; $("#headStrip").innerHTML = ""; $("#pairScale").textContent = "";
  $("#pairHead").textContent = "one pair's contribution — click a QK row";
}

// `b_out` is a learned BIAS, not a measured mean. It is approximately the corpus mean of the
// module's output, but naming it "mean" states a claim nothing here measures.
const KIND = {pair:"", q_bias:"bias query", k_bias:"bias key", bias_bias:"bias × bias",
              error:"error"};
const KIND_NOTE = {
  k_bias: "constant along the key axis — softmax ignores a constant there, so this moves no probability",
  q_bias: "constant along the query axis; the key dependence here is the positional / BOS prior",
  bias_bias: "constant everywhere — it shifts the logit and changes no pattern",
  error: "what the decomposition could not explain",
};
// Inside a fixed head, a component's mass in some OTHER head is not part of this row. The number
// that belongs here is how much of it lives in the head being decomposed.
const inHead = m => m ? `${(100*m[STATE.head]).toFixed(0)}%` : "";

let QK = null;
// The row whose map is on screen, so the view switch can redraw it without another fetch.
let PAIR = null;
// The last reconstruction, for the same reason.
let SUBSET = null;
// `<kind>:<c_q>:<c_k>`, the same spelling `/api/prompt/subset` parses, so the selection IS the
// request and nothing has to be translated between the table and the route.
const PICKED = new Set();
const rowKey = r => `${r.kind}:${r.q_idx ?? ""}:${r.k_idx ?? ""}`;

function syncPicked(){
  const n = PICKED.size;
  $("#qkRecon").textContent = `reconstruct from ${n} picked`;
  $("#qkRecon").disabled = n === 0;
}

// Two ways to look at a chosen set, and the `map` switch picks between them.
//
// BUILD-UP (`z`) sums the picked terms and softmaxes them. It reconstructs ONE logit exactly --
// the cell the rows came from, since a component that did not fire there contributes 0 there --
// and no part of the pattern. Away from that cell the same argument fails: the rows are keyed to
// the components that fired at [t0,k0], and cc there runs over ALL of them. Measured on L4H11
// [13,12], at the cell [13,0]: cc −57.94, bias +52.39, error −0.97, summing to the model's −6.52,
// while every picked pair together supplies +0.07 -- 0.12% of cc. So the build-up lands on
// key 13 (pairs only, p 0.90) or key 0 (with the whole bias group, p 1.0000) where the model is
// on key 12. The `gap` readout says so; the Δ-probability view is the way to ask this question.
//
// ABLATION (`Δ probability`) is `softmax(Z) − softmax(Z − Σ picked)`: the model's own score with
// the chosen terms taken out. A real counterfactual at every cell, whatever was picked -- which is
// why a reconstruction opens here rather than in the build-up.
let SHOWN_ABLATION = false;
async function reconstruct(){
  if(!QK || !PICKED.size) return;
  // Open the FIRST reconstruction in the ablation view, once per page, then leave the choice
  // alone -- flipping a control the reader has just set is worse than a bad default.
  if(!SHOWN_ABLATION){ $("#qkView").value = "p"; SHOWN_ABLATION = true; }
  SUBSET = await busy("reconstructing…", () => get("/api/prompt/subset",
      {prompt: STATE.prompt, layer: STATE.layer, head: STATE.head,
       rows: [...PICKED].join(","), t: QK.t, t_key: QK.t_key,
       with_error: $("#qkErr").checked, complete: $("#qkWhole").checked}));
  drawSubset();
}

function drawSubset(){
  const d = SUBSET; if(!d) return;
  // A subset is not one row, so the per-head strip does not apply.
  PAIR = null; $("#headStrip").innerHTML = "";
  const n = `${d.n_rows} picked row${d.n_rows>1?"s":""}`;
  const nodes = d.complete ? ` <span class="badge">+ every term outside this list</span>`
              : (d.with_error ? ` <span class="badge">+ error node</span>` : "")
                + (d.with_error ? "" : ` <span class="badge">components only</span>`);
  if($("#qkView").value === "p"){
    const m = d.attn_true.map((row,t) => row.map((p,k) => p - d.attn_off[t][k]));
    heatGrid("#pairGrid", d.pieces, m, 1,
             (t,k) => `p ${d.attn_true[t][k].toFixed(4)} → ${d.attn_off[t][k].toFixed(4)} without`);
    $("#pairHead").innerHTML = `<b>effect of ${n} on the pattern</b>`
      + `<br><span class="dim">softmax(Z) − softmax(Z − Σ picked), rows = query, cols = key</span>`;
    $("#pairScale").innerHTML = `largest |Δp| <b>`
      + `${Math.max(...m.flat().map(Math.abs), 0).toFixed(3)}</b>, on the full 0–1 scale`
      + ` <span class="dim">— an ablation of the model's own score, so this one is a real`
      + ` counterfactual at every cell, not only at [${d.at.t},${d.at.t_key}]. The switches change`
      + ` WHAT is removed: pairs alone is a different experiment from pairs + bias + error.</span>`;
    return;
  }
  heatGrid("#pairGrid", d.pieces, d.attn, 1);
  $("#pairHead").innerHTML = `<b>pattern ${d.complete ? "with everything else filled in" : "built"} `
    + `from ${n}</b>${nodes}`
    + `<br><span class="dim">at [${d.at.t},${d.at.t_key}]: logit `
    + `<b>${fmt(d.at.z,3)}</b> of <b>${fmt(d.at.z_true,3)}</b> · the rows you did NOT pick sum to `
    + `<b class="${cls(d.at.omitted)}">${fmt(d.at.omitted,3)}</b></span>`;
  $("#pairScale").innerHTML = d.complete
    ? `largest gap from the real pattern <b>${d.gap.toExponential(2)}</b>`
      + ` <span class="dim">— <code>full − (listed − picked)</code>: the model's own score with your`
      + ` omissions removed, so this is exact at every cell. Tick every row and the gap is fp noise;`
      + ` what you see is what the rows you left out were holding up.</span>`
    : `largest gap from the real pattern <b>${d.gap.toFixed(3)}</b>`
    + ` <span class="warn">— a build-up rebuilds the LOGIT at [${d.at.t},${d.at.t_key}] and no part`
    + ` of the pattern. The rows are that cell's decomposition, so elsewhere the pair terms are`
    + ` missing whatever fired at their own positions while the bias and error nodes arrive whole;`
    + ` the switches make this worse away from the cell, not better. Switch the map to`
    + ` Δ probability for the ablation, which is sound everywhere.</span>`;
}

async function loadQK(t, k){
  // The complete set is ~1.3k rows and sums to the logit EXACTLY, so it is fetched whole and
  // truncated here: a server-side top-k cannot report what the rows it dropped add up to.
  QK = await busy("qk…", () => get("/api/prompt/qk",
      {prompt: STATE.prompt, layer: STATE.layer, head: STATE.head, t, t_key: k, k: 5000}));
  drawQK();
  const first = QK.rows.find(r => r.kind === "pair");
  if(first) showPair(first);
}

// Three explicit fields rather than one text box with a syntax. The kinds are a DROPDOWN because
// the bias rows are the ones you cannot guess the spelling of -- `bias x bias` failed against the
// label `bias × bias`, and a filter that silently matches nothing is worse than no filter.
function qkFilter(){
  return {q: $("#qkFq").value.trim(), k: $("#qkFk").value.trim(), kind: $("#qkFkind").value};
}
const qkFiltered = f => !!(f.q || f.k || f.kind);
function qkMatch(r, f){
  // "bias" groups the three; with select-all that is the whole bias node, built out of rows the
  // reader can see and count -- which is what a hidden "+ bias nodes" switch could never be.
  if(f.kind === "bias" ? !r.kind.includes("bias") : f.kind && r.kind !== f.kind) return false;
  if(f.q && r.q_idx !== +f.q) return false;
  if(f.k && r.k_idx !== +f.k) return false;
  return true;
}

function drawQK(){
  const d = QK; if(!d) return;
  const want = +$("#qkRows").value, how = $("#qkSort").value, find = qkFilter();
  let all = [...d.rows].sort(how === "abs" ? (x,y) => Math.abs(y.contribution) - Math.abs(x.contribution)
                            : how === "desc" ? (x,y) => y.contribution - x.contribution
                                             : (x,y) => x.contribution - y.contribution);
  if(qkFiltered(find)) all = all.filter(r => qkMatch(r, find));
  const rows = want ? all.slice(0, want) : all;
  const shown = rows.reduce((a,r) => a + r.contribution, 0);
  const total = all.reduce((a,r) => a + r.contribution, 0);
  $("#qkMeta").innerHTML =
      `logit[${d.t},${d.t_key}] = <b>${fmt(d.z)}</b> &rarr; p = <b>${d.attn.toFixed(4)}</b> &nbsp;`
    + `<span class="dim">${JSON.stringify(d.q_piece)} attending to ${JSON.stringify(d.k_piece)}</span>`;
  // Contributions are SIGNED and cancel: |contribution| over all rows runs many times the logit
  // itself, so there is no percentage to read here and a truncated list is not a share of anything.
  const gross = all.reduce((a,r) => a + Math.abs(r.contribution), 0);
  $("#qkSum").innerHTML = `${rows.length} of ${all.length}`
    + (qkFiltered(find) ? ` matching (of ${d.rows.length})` : "") + ` rows &middot; shown sum `
    + `<b class="${cls(shown)}">${fmt(shown,3)}</b> &middot; all rows sum `
    + `<b class="${cls(total)}">${fmt(total,3)}</b> = the logit`
    + ` &middot; <span class="warn">Σ|contribution| = ${gross.toFixed(1)}</span>`
    + ` <span class="dim">— terms are signed and cancel, so these are not percentages</span>`;
  const qm = d.q_module, km = d.k_module;
  // The header box picks exactly the rows on screen, so the filter IS the selector: `q5740` then
  // tick-all picks that component's rows and nothing else, and an empty filter with rows=all picks
  // the whole decomposition.
  const allPicked = rows.length > 0 && rows.every(r => PICKED.has(rowKey(r)));
  $("#qk").innerHTML = thead([[`<input type="checkbox" id="qkAll" ${allPicked?"checked":""}>`,
                               false,"pick every row shown below"],
                              ["map",false,"draw this row's [query, key] contribution"],
                              ["contrib",true,"e_q · e_k · ⟨u_q, u_k⟩ / √d_h — these sum to the logit"],
                              ["kind",false],["q",true],["k",true],
                              ["in h"+STATE.head,true,
                               "share of the component's squared write norm inside THIS head"]])
    + rows.map((r,i) => `<tr class="click" data-i="${i}">
        <td><input type="checkbox" class="pick" data-key="${rowKey(r)}"
              ${PICKED.has(rowKey(r)) ? "checked" : ""}></td>
        <td class="dim" style="color:var(--hi)">&#9635;</td>
        <td class="num ${cls(r.contribution)}">${fmt(r.contribution)}</td>
        <td class="dim">${KIND[r.kind]}</td>
        <td class="num${r.q_idx!=null?" hit":""}" data-m="${qm}" data-i2="${r.q_idx ?? ""}"
            >${r.q_idx ?? "—"}</td>
        <td class="num${r.k_idx!=null?" hit":""}" data-m="${km}" data-i2="${r.k_idx ?? ""}"
            >${r.k_idx ?? "—"}</td>
        <td class="num dim">${inHead(r.q_head_mass) || inHead(r.k_head_mass)}</td></tr>`).join("");
  // EVERY row draws its map, bias and error rows included. The q/k index cells open that
  // component instead, and the checkbox picks without drawing, so no two clicks fight over a pixel.
  $("#qkAll").onclick = () => {
    const on = $("#qkAll").checked;
    for(const r of rows){ if(on) PICKED.add(rowKey(r)); else PICKED.delete(rowKey(r)); }
    drawQK();
  };
  for(const tr of $("#qk").querySelectorAll("tr.click"))
    tr.onclick = () => showPair(rows[+tr.dataset.i]);
  for(const cb of $("#qk").querySelectorAll("input.pick"))
    cb.onclick = e => { e.stopPropagation();
      if(cb.checked) PICKED.add(cb.dataset.key); else PICKED.delete(cb.dataset.key);
      syncPicked(); };
  syncPicked();
  for(const td of $("#qk").querySelectorAll("td[data-i2]"))
    if(td.dataset.i2 !== "")
      td.onclick = e => { e.stopPropagation(); showComponent(td.dataset.m, +td.dataset.i2); };
}

// ---- boot --------------------------------------------------------------------
async function run(){
  STATE.prompt = $("#prompt").value;
  const d = await busy("tracing… (first call loads the model, ~80s)",
                       () => get("/api/prompt/trace", targetArgs()));
  STATE.pieces = d.tokens.map(t => t.piece);
  STATE.layers = d.layers;
  drawStrip(d);
  // Offer the layers the run actually decomposed. A partial run (L3-10) has no layer 0, so a
  // 0-based list would put unusable options in the dropdown and 422 on the first one.
  fill($("#layer"), opts(d.layers), STATE.layer);
  fill($("#head"), opts(Array.from({length:d.n_heads}, (_,i)=>i)), STATE.head);
  STATE.layer = +$("#layer").value; STATE.head = +$("#head").value;
  if(!STATE.meta){
    const M = STATE.meta = await get("/api/meta", {});
    const roles = [...new Set(M.modules.map(m => m.role))];
    const siteOpts = saeSites().map(x => ({value:x, text:(M.sae_sites||{})[x] || x}));
    fill($("#bLayer"), opts(d.layers), STATE.layer);
    fill($("#bRole"), opts(roles), "attn.o");
    fill($("#bSite"), siteOpts); fillRel("#bSite", "#bRel");
    fill($("#aRole"), opts(roles), "attn.q");
    fill($("#bRole2"), opts(roles), "attn.k");
    fill($("#aSite"), siteOpts); fillRel("#aSite", "#aRel");
    fill($("#bSite2"), siteOpts); fillRel("#bSite2", "#bRel2");
    fill($("#aLay"), opts(d.layers), STATE.layer);
    fill($("#bLay"), opts(d.layers), STATE.layer);
    fill($("#tpl"), tplList().map(t => ({value:t.key, text:t.label})), "attn_qk");
    syncEp("a"); syncEp("b");
  }
  setPositions(null, null);
  applyTemplate();
  await Promise.all([drawBrowse(), loadAttn()]);
  if(d.top_nodes.length) showComponent(d.top_nodes[0].module, d.top_nodes[0].idx);
}

$("#go").onclick = run;
$("#prompt").onkeydown = e => { if(e.key === "Enter") run(); };
$("#tabLayer").onclick = () => setTab("layer");
$("#tabTop").onclick = () => setTab("top");
$("#tabStatic").onclick = () => setITab("static");
$("#tabLive").onclick = () => setITab("live");
$("#scopeAll").onclick = () => { STATE.scope = "all"; $("#scopeAll").classList.add("on");
                                 $("#scopeTok").classList.remove("on"); drawBrowse(); };
$("#scopeTok").onclick = () => { STATE.scope = "tok"; $("#scopeTok").classList.add("on");
                                 $("#scopeAll").classList.remove("on"); drawBrowse(); };
$("#bSite").onchange = () => { fillRel("#bSite", "#bRel"); drawBrowse(); };
for(const id of ["#bLayer","#bRole","#bRel","#bSort"]) $(id).onchange = () => drawBrowse();
$("#tpl").onchange = applyTemplate;
$("#atpRun").onclick = () => runAtp(0);
for(const id of ["#atpSort","#atpTop"]) $(id).onchange = () => { if(ATP && !ATP.reason) runAtp(0); };
$("#atpPrev").onclick = () => { if(ATP && !ATP.reason && ATP.offset > 0)
                                  runAtp(Math.max(0, ATP.offset - atpLimit())); };
$("#atpNext").onclick = () => { if(ATP && !ATP.reason && ATP.offset + atpLimit() < ATP.n_total)
                                  runAtp(ATP.offset + atpLimit()); };
$("#pSwap").onclick = swapEndpoints;
for(const w of ["a","b"]){
  const x = w === "b" ? "2" : "";
  $("#" + w + "Kind").onchange = () => { syncEp(w); refreshPair(); };
  $("#" + w + "Site" + x).onchange = () => { fillRel("#" + w + "Site" + x, "#" + w + "Rel" + x);
                                            fillEpLayers(w, +$("#" + w + "Lay").value); refreshPair(); };
  $("#" + w + "Rel" + x).onchange = () => { fillEpLayers(w, +$("#" + w + "Lay").value); refreshPair(); };
  $("#" + w + "Role" + x).onchange = refreshPair;
  $("#" + w + "Lay").onchange = refreshPair;
  $("#" + w + "Side").onchange = refreshPair;
}
for(const id of ["#pMetric","#pMode","#pHead","#pK","#pDens"]) $(id).onchange = () => refreshPair();
$("#pDest").onchange = () => { setPositions(+$("#pDest").value, STATE.src); drawTab(); };
$("#pSrc").onchange = () => { const v = $("#pSrc").value;
  setPositions(STATE.dest, v === "" ? null : +v); drawTab(); };
for(const id of ["#qkRows","#qkSort"]) $(id).onchange = () => drawQK();
for(const id of ["#qkFq","#qkFk"]) $(id).oninput = () => drawQK();
$("#qkFkind").onchange = () => drawQK();
$("#qkView").onchange = () => { if(PAIR) drawPairGrid(PAIR.d, PAIR.r); else drawSubset(); };
$("#qkRecon").onclick = reconstruct;
for(const id of ["#qkErr","#qkWhole"]) $(id).onchange = () => { if(PICKED.size) reconstruct(); };
$("#qkClear").onclick = () => { PICKED.clear(); drawQK(); PAIR = null; SUBSET = null;
  $("#pairGrid").innerHTML = ""; $("#headStrip").innerHTML = ""; $("#pairScale").textContent = "";
  $("#pairHead").textContent = "one pair's contribution — click a QK row"; };
for(const id of ["#layer","#head"]) $(id).onchange = () => {
  STATE.layer = +$("#layer").value; STATE.head = +$("#head").value; loadAttn();
};
run();
</script></body></html>
'''
