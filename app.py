"""Tiny browser UI for open-loop LeWM rollouts on Moving-MNIST."""

import argparse
import base64
import csv
import io
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate

from stable_pretraining.backbone.decoders import build_image_decoder
from utils import LeWM


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeWM · Latent World Model</title><style>
:root{--ink:#f6f4ff;--muted:#aaa6bd;--panel:rgba(20,18,35,.72);--line:rgba(255,255,255,.1);--violet:#9b7bff;--cyan:#52e7da}
*{box-sizing:border-box}html{color-scheme:dark}body{min-height:100vh;margin:0;color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif;background:#090812;overflow-x:hidden}
body:before{content:"";position:fixed;inset:-30%;z-index:-2;background:radial-gradient(circle at 25% 25%,#4f2a8a 0,transparent 27%),radial-gradient(circle at 75% 35%,#075d67 0,transparent 25%),radial-gradient(circle at 50% 90%,#301b55 0,transparent 28%);filter:blur(40px)}
body:after{content:"";position:fixed;inset:0;z-index:-1;opacity:.13;background-image:linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px);background-size:32px 32px;mask-image:linear-gradient(to bottom,black,transparent)}
.shell{width:min(1120px,calc(100% - 32px));margin:auto;padding:38px 0 60px}header{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:24px}.eyebrow{color:var(--cyan);font-size:11px;font-weight:800;letter-spacing:.18em;text-transform:uppercase}
h1{font-size:clamp(34px,6vw,64px);line-height:.95;margin:10px 0 14px;letter-spacing:-.055em}h1 span{color:transparent;background:linear-gradient(100deg,var(--violet),var(--cyan));background-clip:text;-webkit-background-clip:text}.lede{max-width:650px;margin:0;color:var(--muted);font-size:16px}
.badge{white-space:nowrap;padding:8px 12px;border:1px solid rgba(82,231,218,.35);border-radius:999px;color:var(--cyan);background:rgba(82,231,218,.07);font-size:12px}.badge:before{content:"";display:inline-block;width:7px;height:7px;margin-right:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 14px var(--cyan)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:18px}.card{border:1px solid var(--line);background:var(--panel);backdrop-filter:blur(18px);box-shadow:0 22px 70px rgba(0,0,0,.32);border-radius:22px}.viewer{padding:18px}.viewer-head,.actions-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.label{color:var(--muted);font-size:11px;font-weight:750;letter-spacing:.12em;text-transform:uppercase}#stepCount{font-variant-numeric:tabular-nums;color:var(--cyan)}
.comparison-labels{display:grid;grid-template-columns:1fr 1fr;margin:13px 0 7px;text-align:center;color:var(--muted);font-size:11px;font-weight:750;letter-spacing:.12em;text-transform:uppercase}.comparison-labels span:first-child{border-right:1px solid var(--line)}.screen{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:16px;background:#000;aspect-ratio:2/1}.screen:after{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--line);pointer-events:none}#current{display:block;width:100%;height:100%;object-fit:fill}#loader{position:absolute;inset:0;display:grid;place-items:center;background:rgba(5,4,12,.7);opacity:0;pointer-events:none;transition:.2s;color:var(--cyan);font-size:12px;letter-spacing:.12em}body.busy #loader{opacity:1}
.side{padding:18px;display:flex;flex-direction:column;gap:18px}.reset{display:grid;grid-template-columns:1fr auto;gap:8px;margin-top:8px}input{min-width:0;width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:11px;color:var(--ink);background:rgba(255,255,255,.05);outline:none}input:focus{border-color:var(--violet);box-shadow:0 0 0 3px rgba(155,123,255,.12)}
button{border:1px solid var(--line);color:var(--ink);background:rgba(255,255,255,.055);cursor:pointer;transition:transform .14s,background .14s,border-color .14s,box-shadow .14s}button:hover{background:rgba(155,123,255,.16);border-color:rgba(155,123,255,.55);box-shadow:0 8px 24px rgba(88,55,170,.18);transform:translateY(-2px)}button:active,.key-hit{transform:scale(.94)!important;background:rgba(82,231,218,.18)!important}button:disabled{opacity:.45;cursor:wait}.reset button{padding:0 14px;border-radius:11px;font-weight:700}
.controls{width:190px;margin:4px auto;display:grid;grid-template-columns:repeat(3,58px);grid-template-rows:repeat(2,58px);gap:8px}.move{border-radius:16px;font-size:25px;font-weight:400}#up{grid-column:2}#left{grid-column:1;grid-row:2}#down{grid-column:2;grid-row:2}#right{grid-column:3;grid-row:2}.rotation-controls{display:flex;justify-content:center;gap:8px;margin:10px auto}.rotation-controls button{width:91px;height:42px;border-radius:13px;font-size:21px}.hint{margin:0;text-align:center;color:var(--muted);font-size:12px}kbd{display:inline-grid;min-width:23px;height:22px;place-items:center;margin:0 2px;border:1px solid var(--line);border-bottom-color:rgba(255,255,255,.28);border-radius:5px;background:rgba(255,255,255,.06);color:#ddd}
.actions-card{grid-column:1/-1;padding:16px 18px;overflow:hidden}#actions{display:flex;align-items:center;gap:7px;overflow-x:auto;padding:11px 2px 2px;scrollbar-color:rgba(155,123,255,.35) transparent}.action{flex:0 0 auto;display:flex;align-items:center;gap:7px;padding:6px 8px 6px 6px;border:1px solid var(--line);border-radius:999px;color:#ded9ed;background:rgba(255,255,255,.04)}.action b{display:grid;width:30px;height:30px;place-items:center;border-radius:50%;color:#0b0a12;background:linear-gradient(135deg,var(--violet),var(--cyan));font-size:17px}.action small{color:#837e96;font-variant-numeric:tabular-nums}.connector{flex:0 0 13px;height:1px;background:linear-gradient(90deg,var(--violet),var(--cyan));opacity:.5}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.chart-card{padding:18px;overflow:hidden}.chart-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}.chart-title strong{font-size:15px}.legend{display:flex;gap:12px;color:var(--muted);font-size:11px}.legend i{display:inline-block;width:9px;height:9px;margin-right:5px;border-radius:50%}.legend .linear{background:var(--violet)}.legend .nonlinear{background:var(--cyan)}.chart{width:100%;height:auto;display:block}.chart text{font:10px system-ui;fill:#858097}.chart .gridline{stroke:rgba(255,255,255,.08);stroke-width:1}.chart .axis{stroke:rgba(255,255,255,.2);stroke-width:1}.chart-empty{height:210px;display:grid;place-items:center;color:var(--muted);font-size:12px}
@media(max-width:760px){header{display:block}.badge{display:inline-block;margin-top:16px}.grid,.charts{grid-template-columns:1fr}.side{grid-row:3}.actions-card{grid-column:1}.shell{width:min(100% - 20px,1120px);padding-top:24px}}
</style></head><body><main class="shell"><header><div><div class="eyebrow">Action-conditioned latent world model</div><h1><span>Watch it imagine.</span></h1><p class="lede">Choose a digit, then steer the model's recursive imagination. After reset, no real frame enters the model again.</p></div><div class="badge">OPEN LOOP</div></header>
<section class="grid"><div class="card actions-card"><div class="actions-head"><span class="label">Past actions</span><span class="label" id="actionCount">0 actions</span></div><div id="actions"></div></div><div class="card viewer"><div class="viewer-head"><span class="label">Current world state</span><span id="stepCount">STEP 00</span></div><div class="comparison-labels"><span>Truth</span><span>Imagined</span></div><div class="screen"><img id="current"><div id="loader">IMAGINING…</div></div></div>
<aside class="card side"><div><span class="label">Choose a digit class</span><div class="reset"><input id="digit" aria-label="Digit class" type="number" min="0" max="9" value="7"><button onclick="reset()">New world</button></div></div><div><span class="label">Apply action</span><div class="controls"><button class="move" id="up" aria-label="Move up" onclick="step('up')">↑</button><button class="move" id="left" aria-label="Move left" onclick="step('left')">←</button><button class="move" id="down" aria-label="Move down" onclick="step('down')">↓</button><button class="move" id="right" aria-label="Move right" onclick="step('right')">→</button></div><div class="rotation-controls" id="rotationControls"><button id="rotate_left" onclick="step('rotate_left')">↶</button><button id="rotate_right" onclick="step('rotate_right')">↷</button></div><p class="hint">Keyboard <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd></p></div></aside></section>
<section class="charts"><div class="card chart-card"><div class="chart-title"><strong>Digit probe accuracy</strong><span class="legend"><span><i class="linear"></i>Linear</span><span><i class="nonlinear"></i>Nonlinear</span></span></div><div id="digitChart"></div></div><div class="card chart-card"><div class="chart-title"><strong>Position probe R²</strong><span class="legend"><span><i class="linear"></i>Linear</span><span><i class="nonlinear"></i>Nonlinear</span></span></div><div id="positionChart"></div></div></section></main>
<script>
const session=(typeof crypto!=='undefined'&&crypto.randomUUID?crypto.randomUUID():'s-'+Date.now()+Math.random().toString(36).slice(2)),arrows={up:'↑',down:'↓',left:'←',right:'→',rotate_left:'↶',rotate_right:'↷'};let busy=false;
function render(x){document.querySelector('#current').src=x.current;document.querySelector('#rotationControls').style.display=x.rotation_enabled?'flex':'none';const actions=x.actions||[],n=actions.length;document.querySelector('#stepCount').textContent=`STEP ${String(n).padStart(2,'0')}`;document.querySelector('#actionCount').textContent=`${n} action${n===1?'':'s'}`;document.querySelector('#actions').innerHTML=n?actions.map((a,i)=>`${i?'<i class="connector"></i>':''}<span class="action"><b>${arrows[a]}</b><small>#${i+1}</small></span>`).join(''):'<span class="hint">No actions yet</span>';const el=document.querySelector('#actions');el.scrollLeft=el.scrollWidth}
async function call(path,body){if(busy)return;busy=true;document.body.classList.add('busy');document.querySelectorAll('button').forEach(b=>b.disabled=true);try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,session})});if(!r.ok)throw new Error(await r.text());render(await r.json())}catch(e){alert(e.message)}finally{busy=false;document.body.classList.remove('busy');document.querySelectorAll('button').forEach(b=>b.disabled=false)}}
function drawChart(id,series){const root=document.querySelector('#'+id),all=series.flatMap(s=>s.values);if(!all.length){root.innerHTML='<div class="chart-empty">No probe metrics in this checkpoint</div>';return}const W=600,H=230,L=42,R=12,T=12,B=30,maxX=Math.max(...all.map(p=>p[0]),1),minY=Math.min(0,...all.map(p=>p[1])),maxY=Math.max(1,...all.map(p=>p[1])),x=v=>L+v/maxX*(W-L-R),y=v=>T+(maxY-v)/(maxY-minY)*(H-T-B);let svg=`<svg class="chart" viewBox="0 0 ${W} ${H}">`;for(let i=0;i<=4;i++){const value=minY+(maxY-minY)*i/4,yy=y(value);svg+=`<line class="gridline" x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}"/><text x="${L-7}" y="${yy+3}" text-anchor="end">${value.toFixed(2)}</text>`}for(let i=0;i<=4;i++){const value=Math.round(maxX*i/4),xx=x(value);svg+=`<line class="gridline" x1="${xx}" y1="${T}" x2="${xx}" y2="${H-B}"/><text x="${xx}" y="${H-10}" text-anchor="middle">${value}</text>`}svg+=`<line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/><text x="${(L+W-R)/2}" y="${H-1}" text-anchor="middle">epoch</text>`;for(const s of series){if(s.values.length)svg+=`<polyline fill="none" stroke="${s.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" points="${s.values.map(p=>x(p[0])+','+y(p[1])).join(' ')}"/>`}root.innerHTML=svg+'</svg>'}
async function loadMetrics(){try{const r=await fetch('/metrics'),m=await r.json();drawChart('digitChart',[{color:'#9b7bff',values:m.digit_linear},{color:'#52e7da',values:m.digit_mlp}]);drawChart('positionChart',[{color:'#9b7bff',values:m.position_linear},{color:'#52e7da',values:m.position_mlp}])}catch(e){console.warn(e)}}
function reset(){call('/reset',{digit:Number(document.querySelector('#digit').value)})}function step(action){call('/step',{action})}document.addEventListener('keydown',e=>{const keys={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right'},a=keys[e.key];if(a){e.preventDefault();const b=document.querySelector('#'+a);b.classList.add('key-hit');setTimeout(()=>b.classList.remove('key-hit'),150);step(a)}});reset();
loadMetrics();
</script></body></html>"""


class Engine:
    directions = {"up": (0, -4), "down": (0, 4), "left": (-4, 0), "right": (4, 0)}

    def __init__(self, checkpoint, data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = saved["config"]
        self.model = LeWM(
            cfg["latent_dim"],
            lamb=cfg["lamb"],
            slices=cfg["slices"],
            rollout_weight=cfg.get("rollout_weight", 0.0),
            sigreg_mode=cfg.get("sigreg_mode", "pooled"),
            action_dim=cfg.get("action_dim", 2),
        )
        self.action_dim = cfg.get("action_dim", 2)
        self.rotation_step = cfg.get("rotation_step", 0)
        state = {
            key.replace("encoder._orig_mod.", "encoder.").replace(
                "predictor._orig_mod.", "predictor."
            ): value
            for key, value in saved["model"].items()
        }
        self.model.load_state_dict(state)
        self.decoder = build_image_decoder(
            cfg["latent_dim"],
            (1, 64, 64),
            decoder_kwargs={"base_channels": 128, "min_channels": 16, "num_res_blocks": 1},
        )
        self.decoder.load_state_dict(saved["decoder"])
        self.model.to(self.device).eval()
        self.decoder.to(self.device).eval()
        self.mnist = MNIST(data, train=False, download=True)
        self.sessions, self.lock = {}, threading.Lock()

    @staticmethod
    def frame(digit, pos, angle=0):
        frame = torch.zeros(1, 64, 64)
        x, y = pos.tolist()
        visible_digit = rotate(
            digit, angle=float(angle), interpolation=InterpolationMode.BILINEAR
        )
        frame[:, y : y + 24, x : x + 24] = visible_digit
        return frame

    @staticmethod
    def panel(truth, imagined):
        pixels = torch.cat((truth, imagined.cpu().clamp(0, 1)), -1).squeeze()
        image = Image.fromarray((pixels * 255).byte().numpy()).convert("RGB")
        image = image.resize((1024, 512), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="PNG")
        return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()

    @torch.inference_mode()
    def reset(self, session, digit_class):
        digit_class = max(0, min(9, digit_class))
        index = int((self.mnist.targets == digit_class).nonzero()[0])
        digit = self.mnist.data[index].float()[None, None].div(255)
        digit = torch.nn.functional.interpolate(
            digit, (24, 24), mode="bilinear", align_corners=False
        )[0]
        pos = torch.tensor([20, 20])
        truth = self.frame(digit, pos, 0)
        latent = self.model.encoder(truth[None].to(self.device))
        imagined = self.decoder(latent)[0].float()
        panel = self.panel(truth, imagined)
        self.sessions[session] = {
            "digit": digit,
            "pos": pos,
            "latent": latent,
            "actions": [],
            "angle": 0,
        }
        return {
            "current": panel,
            "actions": [],
            "rotation_enabled": self.action_dim == 3,
        }

    @torch.inference_mode()
    def step(self, session, direction):
        state = self.sessions.get(session)
        if state is None:
            return self.reset(session, 7)
        if direction in self.directions:
            requested = torch.tensor(self.directions[direction])
            rotation_action = 0
        elif direction in ("rotate_left", "rotate_right") and self.action_dim == 3:
            requested = torch.zeros(2, dtype=torch.long)
            rotation_action = -1 if direction == "rotate_left" else 1
        else:
            raise ValueError(f"unknown action: {direction}")
        new_pos = (state["pos"] + requested).clamp(0, 40)
        action = (new_pos - state["pos"]).float() / 4
        if self.action_dim == 3:
            action = torch.cat((action, torch.tensor([rotation_action], dtype=torch.float)))
            state["angle"] = (
                state["angle"] + rotation_action * self.rotation_step
            ) % 360
        action = action[None].to(self.device)
        state["pos"] = new_pos
        state["latent"] = self.model.step(state["latent"], action)
        truth = self.frame(state["digit"], new_pos, state["angle"])
        imagined = self.decoder(state["latent"])[0].float()
        panel = self.panel(truth, imagined)
        state["actions"] = (state["actions"] + [direction])[-31:]
        return {
            "current": panel,
            "actions": state["actions"],
            "rotation_enabled": self.action_dim == 3,
        }


def latest_checkpoint(root="runs"):
    selection = Path(__file__).with_name("default_checkpoint.txt")
    if selection.exists():
        checkpoint = Path(selection.read_text().strip()).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).parent / checkpoint
        if checkpoint.is_file():
            return checkpoint
    checkpoints = list(Path(root).rglob("lewm.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"no lewm.pt found below {Path(root).resolve()}; run `python mmnist.py` first"
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def probe_history(checkpoint):
    metrics = Path(checkpoint).with_name("metrics.csv")
    names = {
        "digit_linear": "eval/digit_probe_acc_epoch",
        "digit_mlp": "eval/digit_mlp_probe_acc_epoch",
        "position_linear": "eval/position_probe_r2_epoch",
        "position_mlp": "eval/position_mlp_probe_r2_epoch",
    }
    result = {name: [] for name in names}
    if not metrics.exists():
        return result
    by_epoch = {}
    with metrics.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("epoch"):
                continue
            epoch = int(float(row["epoch"])) + 1
            values = by_epoch.setdefault(epoch, {})
            for name, column in names.items():
                if row.get(column):
                    values[name] = float(row[column])
    for epoch, values in sorted(by_epoch.items()):
        for name, value in values.items():
            result[name].append([epoch, value])
    return result


def serve(args):
    checkpoint = args.checkpoint or latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint.resolve()}")
    engine = Engine(checkpoint, args.data)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                body = json.dumps(probe_history(checkpoint)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/":
                self.send_error(404)
                return
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(size))
                session = request.get("session") or str(uuid.uuid4())
                with engine.lock:
                    if self.path == "/reset":
                        result = engine.reset(session, int(request.get("digit", 7)))
                    elif self.path == "/step":
                        result = engine.step(session, request.get("action"))
                    else:
                        raise ValueError("unknown endpoint")
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as error:
                self.send_error(400, str(error))

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LeWM app: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        help="LeWM checkpoint (default: selected release, then newest runs/**/lewm.pt)",
    )
    parser.add_argument("--data", default="data")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    serve(parser.parse_args())
