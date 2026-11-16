// In-browser port of app/backend.py (forward pass + generate only).
// Same math, same names, line for line where possible, so the two can be
// read side by side. Weights come from weights.js (window.ABC_WEIGHTS).
//
// One known difference: the random sampler. numpy's default_rng and this
// file's mulberry32 give different streams, so a given seed won't produce
// the same characters as the Flask version, only the same behavior.

const ABC = (() => {
    const W = window.ABC_WEIGHTS;
    const VOCAB = W.vocab;
    const SOURCE_NAMES = W.source_names;           // alpha vector in column order
    const C2I = Object.fromEntries(VOCAB.map((ch, i) => [ch, i]));
    const V = W.wte.length, N_EMBD = W.wte[0].length;
    const BLOCK_SIZE = W.wpe.length;
    const N_LAYER = Object.keys(W).filter(k => k.startsWith('Wqkv_')).length;
    const N_HEAD = W.Mhead_0.length;
    const HEAD_DIM = N_EMBD / N_HEAD;
    const EPS = 1e-5;

    // SMALL MATRIX HELPERS (arrays of rows)
    const matmul = (A, B) => A.map(row => {
        const out = new Array(B[0].length).fill(0);
        for (let i = 0; i < row.length; i++) {
            const a = row[i], Bi = B[i];
            for (let j = 0; j < out.length; j++) out[j] += a * Bi[j];
        }
        return out;
    });
    const matvec = (M, v) => M.map(row => row.reduce((s, m, i) => s + m * v[i], 0));
    const scaleCols = (X, g) => X.map(row => row.map((x, j) => x * g[j]));
    const addRows = (A, B) => A.map((row, i) => row.map((x, j) => x + B[i][j]));

    // FORWARD PASS MATH
    function layernorm(X) {
        return X.map(row => {
            const mu = row.reduce((s, x) => s + x, 0) / row.length;
            const xc = row.map(x => x - mu);
            const variance = xc.reduce((s, x) => s + x * x, 0) / row.length;
            return xc.map(x => x / Math.sqrt(variance + EPS));
        });
    }

    function softmax(row) {
        const m = Math.max(...row);
        const e = row.map(x => Math.exp(x - m));
        const s = e.reduce((a, b) => a + b, 0);
        return e.map(x => x / s);
    }

    const gate = (alpha, M) => matvec(M, alpha);

    const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

    function attnBranch(X, li, alpha) {
        const T = X.length;
        const qkv = matmul(layernorm(X), W[`Wqkv_${li}`]);
        const gHead = gate(alpha, W[`Mhead_${li}`]);
        const yFlat = Array.from({ length: T }, () => new Array(N_EMBD).fill(0));
        for (let h = 0; h < N_HEAD; h++) {
            const q0 = h * HEAD_DIM, k0 = N_EMBD + h * HEAD_DIM, v0 = 2 * N_EMBD + h * HEAD_DIM;
            for (let t = 0; t < T; t++) {
                const scores = [];
                for (let s = 0; s < T; s++) {
                    if (s > t) { scores.push(-1e9); continue; }   // can't listen ahead
                    let dot = 0;
                    for (let d = 0; d < HEAD_DIM; d++) dot += qkv[t][q0 + d] * qkv[s][k0 + d];
                    scores.push(dot / Math.sqrt(HEAD_DIM));
                }
                const att = softmax(scores);
                for (let d = 0; d < HEAD_DIM; d++) {
                    let y = 0;
                    for (let s = 0; s < T; s++) y += att[s] * qkv[s][v0 + d];
                    yFlat[t][h * HEAD_DIM + d] = y * gHead[h];   // mic check #1 (each head's own chart)
                }
            }
        }
        return scaleCols(matmul(yFlat, W[`Wo_${li}`]), gate(alpha, W.M_embd)); // mic check #2 (shared chart)
    }

    function mlpBranch(X, li, alpha) {
        const gInner = gate(alpha, W[`Minner_${li}`]);
        const H = matmul(layernorm(X), W[`Win_${li}`]).map(row => row.map((x, j) => gelu(x) * gInner[j])); // mic check #1
        return scaleCols(matmul(H, W[`Wout_${li}`]), gate(alpha, W.M_embd));                           // mic check #2
    }

    function forward(charIdx, alpha) {
        const gEmbd = gate(alpha, W.M_embd);
        let X = charIdx.map((c, t) => W.wte[c].map((x, j) => (x + W.wpe[t][j]) * gEmbd[j])); // raw cue: checked once
        for (let li = 0; li < N_LAYER; li++) {
            X = addRows(X, attnBranch(X, li, alpha));   // voice 1: listen back
            X = addRows(X, mlpBranch(X, li, alpha));    // voice 2: own interpretation
        }
        const xf = scaleCols(layernorm(X), gEmbd);      // final re-level, checked once more
        return matmul(xf, W.lm_head);
    }

    // Seeded random numbers (mulberry32), standing in for np.random.default_rng
    function mulberry32(seed) {
        let a = seed >>> 0;
        return () => {
            a = (a + 0x6D2B79F5) >>> 0;
            let t = a;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    function generate(alpha, { length = 60, startChar = ' ', seed = 0, temperature = 0.8 } = {}) {
        const rand = mulberry32(seed);
        const history = [C2I[startChar]];
        let out = startChar;
        for (let i = 0; i < length; i++) {
            const ctx = history.slice(-BLOCK_SIZE);     // attention only ever saw this much context in training
            const logits = forward(ctx, alpha);
            const probs = softmax(logits[logits.length - 1].map(x => x / temperature));
            let r = rand(), nxt = V - 1;
            for (let j = 0; j < V; j++) { r -= probs[j]; if (r < 0) { nxt = j; break; } }
            history.push(nxt);
            out += VOCAB[nxt];
        }
        return out;
    }

    // dials = { cat, dog, bird } -> text, same contract as POST /api/generate
    function generateFromDials(dials, seed = 0) {
        const alpha = SOURCE_NAMES.map(n => Math.min(1, Math.max(0, Number(dials[n]) || 0)));
        return generate(alpha, { seed });
    }

    return { forward, generate, generateFromDials, SOURCE_NAMES, C2I };
})();
