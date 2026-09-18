import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.dirname(APP_DIR) # the frontend files live one level up

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# PULL FROM WEIGHTS NPZ

_w = np.load(os.path.join(APP_DIR, "weights.npz"))
VOCAB = list(_w["vocab"])
SOURCE_NAMES = [str(s) for s in _w["source_names"]] #alpha vector in column order
C2I = {ch: i for i, ch in enumerate(VOCAB)}
I2C = {i: ch for ch, i in C2I.items()}
V, N_EMBD = _w["wte"].shape
BLOCK_SIZE, _ = _w["wpe"].shape
K = len(SOURCE_NAMES)

N_LAYER = sum (1 for k in _w.files if k.startswith("Wqkv_"))

wte, wpe, lm_head = _w["wte"], _w["wpe"], _w["lm_head"]
M_embd = _w["M_embd"]
Wqkv = [_w[f"Wqkv_{li}"] for li in range(N_LAYER)]
Wo = [_w[f"Wo_{li}"] for li in range(N_LAYER)]
Win = [_w[f"Win_{li}"] for li in range(N_LAYER)]
Wout = [_w[f"Wout_{li}"] for li in range(N_LAYER)]
M_head = [_w[f"Mhead_{li}"] for li in range(N_LAYER)]
M_inner = [_w[f"Minner_{li}"] for li in range(N_LAYER)]

N_HEAD = M_head[0].shape[0]
HEAD_DIM = N_EMBD // N_HEAD
EPS = 1e-5

# FORWARD PASS MATH
# `layernorm_fwd`, `attn_branch_fwd`, `mlp_branch_fwd`, `forward`, 
# `generate`), with the backward-pass and training-only pieces dropped
# since `backend.py` only ever infers

def layernorm(x):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    return xc / np.sqrt(var + EPS)

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

def gate(alpha, M):
    return M @ alpha

def gelu(x):
    t = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1 + t)

def attn_branch(x, li, alpha):
    T = x.shape[0]
    n = layernorm(x)
    qkv = n @ Wqkv[li]
    q, k, v = np.split(qkv, 3, axis=-1)
    def to_heads(z): return z.reshape(T, N_HEAD, HEAD_DIM).transpose(1, 0, 2)
    qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)
    scores = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(HEAD_DIM)
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)     # can't listen ahead
    scores = np.where(mask[None, :, :], -1e9, scores)
    att = softmax(scores)
    y = att @ vh
    y = y * gate(alpha, M_head[li])[:, None, None]         # mic check #1 (each head's own chart)
    y_flat = y.transpose(1, 0, 2).reshape(T, N_EMBD)
    return (y_flat @ Wo[li]) * gate(alpha, M_embd)[None, :]  # mic check #2 (shared chart)

def mlp_branch(x, li, alpha):
    n = layernorm(x)
    h = gelu(n @ Win[li])
    h = h * gate(alpha, M_inner[li])[None, :]              # mic check #1 (own chart)
    return (h @ Wout[li]) * gate(alpha, M_embd)[None, :]    # mic check #2 (shared chart)

def forward(char_idx, alpha):
    T = len(char_idx)
    g_embd = gate(alpha, M_embd)
    x = (wte[char_idx] + wpe[:T]) * g_embd[None, :]        # raw cue: checked once
    for li in range(N_LAYER):
        x = x + attn_branch(x, li, alpha)                  # voice 1: listen back
        x = x + mlp_branch(x, li, alpha)                   # voice 2: own interpretation
    xf = layernorm(x) * g_embd[None, :]                    # final re-level, checked once more
    return xf @ lm_head

def generate(alpha, length=60, start_char=" ", seed=0, temperature=0.8):
    g = np.random.default_rng(seed)
    history = [C2I[start_char]]
    out = [start_char]
    for _ in range(length):
        ctx = np.array(history[-BLOCK_SIZE:])              # attention only ever saw this much context in training
        logits = forward(ctx, alpha)
        probs = softmax(logits[-1] / temperature)
        nxt = g.choice(V, p=probs)
        history.append(nxt)
        out.append(I2C[nxt])
    return "".join(out)

# ROUTING LOGIC
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.get_json(silent=True) or {}
    dials = body.get("dials")
    if not isinstance(dials, dict) or any(n not in dials for n in SOURCE_NAMES):
        return jsonify({"error": f"body must be {{'dials': {{{', '.join(SOURCE_NAMES)}}}}}"}), 400
    try:
        alpha = np.array([min(1.0, max(0.0, float(dials[n]))) for n in SOURCE_NAMES])
    except (TypeError, ValueError):
        return jsonify({"error": "dial values must be numbers"}), 400
    seed = body.get("seed", 0)
    text = generate(alpha, seed=seed)
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(port=5058, debug=True)