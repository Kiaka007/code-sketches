// Principle behind spider-graph:
// An axis is an angle, a dial is a radius (0 to 1) along that angle
// Each radius represents a unit between 0 and 1 in alpha
// Future work will plug in the name, label, and descriptor

const AXES = [
    {name: 'cat', label: 'cat', fullName: 'Cat', descriptor: 'Cat Words', angleDeg: -90 },
    {name: 'dog', label: 'dog', fullName: 'Dog', descriptor: 'Dog Words', angleDeg: 30 },
    {name: 'bird', label: 'bir', fullName: 'Bird', descriptor: 'Bird Words', angleDeg: 150 },
];

//Resting state
const dials = { cat: 0, dog: 0, bird: 0 };

//Center X (CX), Center Y (CY) along max Radius (R)
// R = 85 to give 15px of padding for label and drag elements
const CX = 100, CY = 100, R = 85;

function axisPoint(angleDeg, radius01) {
    const rad = angleDeg * Math.PI / 180;
    return {
        x: CX + R * radius01 * Math.cos(rad),
        y: CY + R * radius01 * Math.sin(rad),
    };
}

// SVG Rendering of the graph
const svg = document.getElementById('radar');
const SVG_NS = 'http://www.w3.org/2000/svg';

// Set SVG attrs
function el(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
}

// Guide rings at intervals 0, 0.25, 0.5, 0.75, 1.0
// These tie to alpha values
for (const r of [0.25, 0.5, 0.75, 1.0]) {
    svg.appendChild(el('circle', { class: 'radar-ring', cx: CX, cy: CY, r: R * r}));
}

// Spokes + Axis Labels
// LABEL_TRIM pulls the vertical-pointing labels' extra gap back in. Text is
// never rotated to match its axis, so the same radial offset reads as a big
// visual gap on a straight-up/straight-down axis but almost none on a
// sideways one. Scaling by sin(angle) keeps horizontal axes (sin = 0) untouched 
// and caps out at LABEL_TRIM for vertical ones, instead of dividing by sin 
// (which would blow up at 0°/180°).
const LABEL_TRIM = 8;

for (const axis of AXES) {
    const rim = axisPoint(axis.angleDeg, 1);
    svg.appendChild(el('line', { class: 'radar-spoke', x1: CX, y1: CY, x2: rim.x, y2: rim.y }));

    const labelPt = axisPoint(axis.angleDeg, 1.3); // Placement of angle outside of rim
    const rad = axis.angleDeg * Math.PI / 180;
    const verticalNudge = -LABEL_TRIM * Math.sin(rad);
    const text = el('text', {class: 'radar-label', x:labelPt.x, y: labelPt.y, dy: verticalNudge });
    text.textContent = axis.label;
    svg.appendChild(text);
}

svg.appendChild(el('circle', { class: 'radar-center', cx: CX, cy: CY, r: 3 }));

const polygon = el('polygon', { class: 'radar-polygon' });
svg.appendChild(polygon);

const handles = {};
for (const axis of AXES) {
    const h = el('circle', { class: 'radar-handle', r: 6 });
    svg.appendChild(h);
    handles[axis.name] = h;
}

// ALPHA DEPTH KEYMAP
//A second non-radial view of the alpha states
// x is categorical not relational, listed in AXES fixed order
// y is the source's alpha value moving down in depth as the alpha value increases
// This view should show that at a quarter size an equilateral shape's samples are 
// more shallow than a vector drawn uniformly at alpha 1.0
const depthKeySvg = document.getElementById('depth-key-svg');
const KEY_W = 104, KEY_H = 40, KEY_Y_TOP = 6, KEY_Y_BOTTOM = 34;
const KEY_X = AXES.map((_, i) => 15 + i * ((KEY_W - 30) / (AXES.length - 1)));

function depthKeyY(dial) {
    return KEY_Y_TOP + dial * (KEY_Y_BOTTOM - KEY_Y_TOP);
}

// Reference grid -- 0/25/50/75/100 lines, static, matches the real Figma asset
for (const frac of [0, 0.25, 0.5, 0.75, 1.0]) {
    const y = KEY_Y_TOP + frac * (KEY_Y_BOTTOM - KEY_Y_TOP);
    depthKeySvg.appendChild(el('line', { class: 'depth-key-grid', x1: 6, y1: y, x2: KEY_W - 6, y2: y }));
}

// The Fill between dots is a scanning aid only, needed due to the
// small surface area of the graphic and to better convey depth
// the relationship between the dots is arbitrary
const depthKeyFill = el('polygon', { class: 'depth-key-fill' });
depthKeySvg.appendChild(depthKeyFill);

const depthKeyDots = AXES.map((axis, i) => {
    const dot = el('circle', { class: 'depth-key-dot', r: 3, cx: KEY_X[i] });
    depthKeySvg.appendChild(dot);
    return dot;
});

function updateDepthKey() {
    const dotYs = AXES.map(axis => depthKeyY(dials[axis.name]));
    depthKeyDots.forEach((dot, i) => dot.setAttribute('cy', dotYs[i]));

    const last = AXES.length - 1;
    const fillPts = AXES.map((axis, i) => `${KEY_X[i]},${dotYs[i]}`);
    fillPts.push(`${KEY_X[last]},${KEY_Y_TOP}`, `${KEY_X[0]},${KEY_Y_TOP}`);
    depthKeyFill.setAttribute('points', fillPts.join(' '));
}


//RENDER GRAPH
function render() {
    const pts = AXES.map(axis => axisPoint(axis.angleDeg, dials[axis.name]));
    polygon.setAttribute('points', pts.map(p => `${p.x}, ${p.y}`).join(' '));
    for (const axis of AXES) {
        const p = axisPoint(axis.angleDeg, dials[axis.name]);
        handles[axis.name].setAttribute('cx', p.x);
        handles[axis.name].setAttribute('cy', p.y);
    }
    updateDepthKey();
}

render();

// DRAG INTERACTION
// always recomputes its position from `axisPoint(angle, dial)`, 
// never from the raw pointer

// Convert axis angle into a path/direction
function unitVector(angleDeg) {
    const rad = angleDeg * Math.PI / 180;
    return { x: Math.cos(rad), y: Math.sin(rad) };
}

// Converts pointer event coordinates to viewbox coordinates
function clientToSvgPoint(evt) {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
}

// When a drag starts within this many units of center, every handle is
// stacked on the exact same point (dial 0 = radius 0 for every axis), so
// there's nothing distinct to click. 

// This makes it to where the drag's direction decides which axis
// gets grabbed instead of all handles being fixed in the z-index
const CENTER_PICK_RADIUS = 10;
// How far the pointer has to move before that direction "locks in"
const DIRECTION_LOCK_THRESHOLD = 6;

// Picks whichever axis's own direction the drag (dx, dy) lines up with best
function pickAxisByDirection(dx, dy) {
    let best = AXES[0], bestDot = -Infinity;
    for (const candidate of AXES) {
        const u = unitVector(candidate.angleDeg);
        const dot = dx * u.x + dy * u.y;
        if (dot > bestDot) {
            bestDot = dot;
            best = candidate;
        }
    }
    return best;
}

// Return a pointerdown handler so each handle gets its own drag session
function startDrag(axis) {
    return function onPointerDown(evt) {
        evt.target.setPointerCapture(evt.pointerId);

        const downPt = clientToSvgPoint(evt);
        const distFromCenter = Math.hypot(downPt.x - CX, downPt.y - CY);

        // Clicked a handle already out on its spoke: no ambiguity, drag it.
        // Clicked at/near center: hold off — resolve which axis by direction
        // once the pointer has actually moved somewhere.
        let activeAxis = axis;
        let resolved = distFromCenter > CENTER_PICK_RADIUS;

        function applyDial(p) {
            const unit = unitVector(activeAxis.angleDeg);
            const vx = p.x - CX, vy = p.y - CY;
            const projected = (vx * unit.x + vy * unit.y) / R; // bounds to this axis so no sideways drifting
            dials[activeAxis.name] = Math.min(1, Math.max(0, projected));
            render();
            updateStats();
        }

        function onMove(moveEvt) {
            const p = clientToSvgPoint(moveEvt);

            if (!resolved) {
                const dx = p.x - downPt.x, dy = p.y - downPt.y;
                if (Math.hypot(dx, dy) < DIRECTION_LOCK_THRESHOLD) return; // not enough movement to read direction yet
                activeAxis = pickAxisByDirection(dx, dy);
                resolved = true;
            }

            applyDial(p);
        }

        //Cleans listener slate
        function onUp(upEvt) {
            evt.target.releasePointerCapture(evt.pointerId);
            evt.target.removeEventListener('pointermove', onMove);
            svg.appendChild(handles[activeAxis.name]); // bring the dragged handle to front, now that the drag has ended
            evt.target.removeEventListener('pointerup', onUp);
            requestGeneration();
        }

        evt.target.addEventListener('pointermove', onMove);
        evt.target.addEventListener('pointerup', onUp);
    };
}

// HANDLES
for (const axis of AXES) {
    handles[axis.name].addEventListener('pointerdown', startDrag(axis));
}

// DYNAMIC UPDATES TO THE STATS LIST
function updateStats() {
    for (const axis of AXES) {
        const valueEl = document.querySelector(`.value[data-axis="${axis.name}"]`);
        if (valueEl === document.activeElement) continue;
        valueEl.value = dials[axis.name].toFixed(2);
    }
    updateNpcTitle();
}

for (const axis of AXES) {
    const valueEl = document.querySelector(`.value[data-axis="${axis.name}"]`);

    valueEl.addEventListener('input', () => {
        const parsed = parseFloat(valueEl.value);
        if (Number.isNaN(parsed)) return; // mid-edit wait for input
        dials[axis.name] = Math.min(1, Math.max(0, parsed));
        render();
        updateStats();
    });

    valueEl.addEventListener('blur', () => {
        valueEl.value = dials[axis.name].toFixed(2); // snap back to formatting
        requestGeneration();
    });

    valueEl.addEventListener('keydown', (evt) => {
        if (evt.key === 'Enter') valueEl.blur();
    });
}

updateStats(); //sync the printed numbers to the resting state on load


// MAIN CHAT TITLES
// npc-title and descriptors are built from 
// whichever axes are actullay present (alpha > 0)
// rebuilt on every alpha change, same trigger points as updateStats()

function buildCombinator(container, items) {
    container.replaceChildren();
    items.forEach((text, i) => {
        if (i > 0) {
            const joiner = document.createElement('p');
            joiner.className = 'joiner';
            joiner.textContent = '+';
            container.appendChild(joiner);
        }
        const item = document.createElement('p');
        item.className = 'item';
        item.textContent = text;
        container.appendChild(item);
    });
}

function updateNpcTitle() {
    const npcTitle = document.querySelector('.npc-title');
    const combinator = document.querySelector('.npc-combinator');
    const descriptors = document.querySelector('.descriptor-combinator');
    const present = AXES.filter(axis => dials[axis.name] > 0);

    npcTitle.classList.toggle('no-sources', present.length === 0);
    if (present.length === 0) {
        buildCombinator(combinator, ['No sources selected']);
        descriptors.replaceChildren();
        return;
    }
    buildCombinator(combinator, present.map(axis => axis.fullName));
    buildCombinator(descriptors, present.map(axis => axis.descriptor));
}

// FETCHING THE OUTPUT
const GENERATE_URL = "/api/generate";
let requestSeq = 0;   // stale-response guard, a response is only ever
// displayed if nothing newer has started since it was sent

async function requestGeneration() {
  const myRequest = ++requestSeq;
  const outputCard = document.querySelector('.output');
  const outputText = document.querySelector('.output-text');
  outputCard.classList.remove('is-error');

  // Don't fetch if all axes are at 0
  const anyPresent = AXES.some(axis => dials[axis.name] > 0);
  if (!anyPresent) {
    outputCard.classList.remove('is-loading');
    outputText.textContent = "No sources selected — nothing to generate.";
    return;
  }

  outputCard.classList.add('is-loading');
  try {
    const res = await fetch(GENERATE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dials }),
    });
    if (myRequest !== requestSeq) return;   // a newer drag/blur already superseded this one
    if (!res.ok) throw new Error(`server responded ${res.status}`);
    const data = await res.json();
    outputText.textContent = data.text;
  } catch (err) {
    if (myRequest !== requestSeq) return;
    outputCard.classList.add('is-error');
    outputText.textContent = "Couldn't reach the model server — is app/backend.py running?";
  } finally {
    if (myRequest === requestSeq) outputCard.classList.remove('is-loading');
  }
}

// REMAP INTERACTIVITY
 
document.querySelector('.remap-btn').addEventListener('click', () => {
  // TODO: wire to real source-remapping once the backend is connected.
  console.log('Remap Sources clicked — no-op for now.');
});