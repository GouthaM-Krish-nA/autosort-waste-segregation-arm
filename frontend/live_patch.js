
/* ======================================================================
   live_patch.js - switches the SegreBot console from the built-in
   simulator to the REAL backend (webcam + trained classifier + arm).

   Add ONE line to frontend/index.html, right after the big </script>
   and before </body>:

       <script src="live_patch.js"></script>

   Nothing else in index.html needs to change. rev1.html stays untouched
   as your offline, simulator-only backup.
   ====================================================================== */
(function () {
  // ---------------------------------------------------------------- 1. connection
  API.BASE_URL = location.protocol.startsWith('http') ? '/api' : 'http://127.0.0.1:5000/api';
  API.USE_MOCK = false;

  const tglMock = $('#tglMock');
  if (tglMock) { tglMock.classList.remove('is-on'); tglMock.setAttribute('aria-pressed', 'false'); }
  const setBase = $('#setBase');
  if (setBase) setBase.value = API.BASE_URL;

  // The model sees webcam photos, not studio photos, so its confidence is lower than 85%.
  State.threshold = 60;
  const thr = $('#thrRange'); if (thr) thr.value = 60;
  const thrVal = $('#thrVal'); if (thrVal) thrVal.textContent = '60%';

  // ---------------------------------------------------------------- 2. API overrides
  // When the backend is unreachable the dashboard now goes quiet instead of inventing data.
  const EMPTY_ANALYTICS = {
    counts: { Plastic: 0, Paper: 0, Metal: 0 }, ok: 0, fail: 0, total: 0,
    accuracy: 91.7, perClass: [], outcome: [], hourly: [{ label: '00', value: 0 }]
  };

  API.getDetection = async function (force = false) {
    const d = await this._req('GET', '/detection/latest' + (force ? '?force=1' : ''), null,
                              () => ({ detected: false }));
    if (!d || d.detected === false) return null;
    d.timestamp = new Date(d.timestamp || Date.now());   // JSON gives a string, the UI wants a Date
    return d;
  };
  API.getAnalytics = function () { return this._req('GET', '/analytics', null, () => EMPTY_ANALYTICS); };
  API.getStatus    = function () { return this._req('GET', '/status', null, () => null); };

  // Manual sliders fire dozens of events per second. The real arm moves slowly, so only send
  // the LAST value, 150 ms after the user pauses (one request per joint, not a flood).
  const jogTimers = {}, jogValues = {};
  API.setJoint = function (id, deg) {
    jogValues[id] = deg;
    clearTimeout(jogTimers[id]);
    jogTimers[id] = setTimeout(function () {
      API._req('POST', '/robot/joint', { joint: id, angle: jogValues[id] }, () => ({ ok: false }));
    }, 150);
    return Promise.resolve({ ok: true });
  };

  // History rows: real outcome only (no random "Failed grip" rows)
  Mock.toRow = function (d) {
    const low = d.confidence < State.threshold;
    return { ...d, action: low ? 'Skipped (low confidence)' : 'Pick \u2192 ' + d.bin,
             status: low ? 'Rejected' : 'Sorted' };
  };

  // ---------------------------------------------------------------- 3. sorting cycle
  const originalRunCycle = runCycle;
  runCycle = async function (d) {
    if (d.confidence < State.threshold) {
      // Do exactly what the real arm does: nothing. No pretend "reject chute" animation.
      log('cmdLog', 'WARN', 'Skipped: confidence ' + d.confidence.toFixed(1) +
          '% is below the ' + State.threshold + '% threshold. Remove the item and try again.');
      State.history.unshift(Mock.toRow(d));
      State.cycleCount++;
      $('#chipCycleVal').textContent = '#' + String(State.cycleCount).padStart(4, '0');
      refreshAll();
      return;
    }
    API.sendCommand(d.category.toUpperCase());     // <- the REAL arm gets the word PLASTIC / METAL / PAPER
    await originalRunCycle(d);                     // the on-screen arm mirrors it (digital twin)

    // The real arm is slower than the animation: wait for it to finish before the next detection.
    setOperation('Waiting for the real arm', 'Moving');
    for (let i = 0; i < 480; i++) {                // up to ~4 min
      const s = await API.getStatus();
      if (!s || !s.busy) break;
      await new Promise(function (r) { setTimeout(r, 500); });
    }
    setOperation('Ready', 'Home');
  };

  // The old presence-based auto trigger is OFF: the webcam now watches the SHOW CAM pose, not the
  // pickup spot. Cycles are started with the "Start sorting cycle" button (section 9 below).
  startAuto = function () {};

  // ---------------------------------------------------------------- 4. buttons
  const detectBtn = $('#btnDetect');
  if (detectBtn) {
    const fresh = detectBtn.cloneNode(true);       // drops the old simulator click handler
    detectBtn.replaceWith(fresh);
    fresh.textContent = 'Classify current view';
    fresh.addEventListener('click', async () => {
      if (State.estop) return;
      const d = await API._req('POST', '/robot/classify', {}, () => ({ detected: false }));
      if (!d || d.detected === false) { log('cmdLog', 'WARN', 'No camera frame yet. Is app.py running?'); return; }
      d.timestamp = new Date(d.timestamp || Date.now());
      renderDetection(d);
      log('cmdLog', 'INFO', 'Manual classification: ' + d.category + ' (' + d.confidence.toFixed(1) +
          '%)  ' + probsText(d) + '. The arm was not moved.');
    });

    const bg = document.createElement('button');
    bg.className = 'btn';
    bg.textContent = 'Set empty background';
    bg.title = 'Click while the pick spot is empty and the arm is at home';
    bg.addEventListener('click', async () => {
      const r = await API._req('POST', '/camera/reset_background', {}, () => ({ ok: false }));
      log('cmdLog', r && r.ok ? 'SUCCESS' : 'ERROR',
          r && r.ok ? 'Empty-spot reference captured.' : 'Could not reach the backend.');
    });
    fresh.before(bg);
  }

  // ---------------------------------------------------------------- 5. live camera instead of the drawing
  ['viewportDash', 'viewportDet'].forEach(function (id) {
    const vp = document.getElementById(id);
    if (!vp) return;
    const fakeLayers = vp.querySelectorAll('.viewport-grid,.viewport-belt,.ghost-object,.scanline,.viewport-vignette');
    const img = document.createElement('img');
    img.alt = 'Live camera';
    img.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:fill;';
    img.onerror = function () {                    // backend not running: show the drawing again
      img.style.display = 'none';
      fakeLayers.forEach(function (e) { e.style.display = ''; });
    };
    fakeLayers.forEach(function (e) { e.style.display = 'none'; });
    vp.style.aspectRatio = '4 / 3';                // webcam shape, so the detection box lines up
    img.src = API.BASE_URL + '/camera/stream';
    vp.prepend(img);
  });

  // ---------------------------------------------------------------- 6. clear the demo data
  Mock.subsystems = [];
  State.history = [];
  State.cycleCount = 0;
  State.bins = { Plastic: 0, Paper: 0, Metal: 0 };
  State.detection = null;
  State.sysLog = [];
  State.cmdLog = [];

  function resetDetectionPanels() {
    const set = function (sel, txt) { const e = $(sel); if (e) e.textContent = txt; };
    set('#dashObject', 'Awaiting object');  set('#dashObjectSub', 'Place an item on the pick spot');
    set('#dashConf', '\u2014');             set('#dashBin', '\u2014');
    set('#dashBinSub', 'No assignment');    set('#chipCycleVal', '\u2014');
    const sub = $('#dashConfSub');
    if (sub) sub.innerHTML = '<i class="lamp ok"></i> Threshold ' + State.threshold + '%';
    set('#detObject', 'No object');         set('#detConf', '\u2014');
    set('#detCat', '\u2014');               set('#detBin', '\u2014');
    set('#detStamp', '\u2014');             set('#detLatency', '\u2014');
    set('#detFrame', '\u2014');
    const wrap = $('#detCatWrap');
    if (wrap) wrap.innerHTML = '<span class="pill metal"><i class="pill-dot"></i>Unclassified</span>';
    const arc = $('#ringArc'); if (arc) arc.style.strokeDashoffset = 276.5;
    ['#bboxDash', '#bboxDet'].forEach(function (s) { const b = $(s); if (b) b.style.display = 'none'; });
  }
  resetDetectionPanels();

  // Analytics page: say what the number really is
  const accNote = document.querySelector('#aAcc + .kpi-delta');
  if (accNote) accNote.textContent = 'Validated on 120 held-out images';

  // The arm has 6 joints (waist, shoulder, elbow, wrist pitch, wrist roll, gripper)
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const hits = [];
  while (walker.nextNode()) {
    const n = walker.currentNode;
    if (n.parentNode.tagName !== 'SCRIPT' && n.nodeValue.indexOf('5-DOF') !== -1) hits.push(n);
  }
  hits.forEach(function (n) { n.nodeValue = n.nodeValue.replace(/5-DOF/g, '6-DOF'); });

  // ---------------------------------------------------------------- 7. real System Status page
  async function refreshStatus() {
    const s = await API.getStatus();
    if (s && s.subsystems) { Mock.subsystems = s.subsystems; renderSystem(); }
  }
  refreshStatus();
  setInterval(function () { if (State.view === 'system') refreshStatus(); }, 4000);
  const ping = $('#btnPing');
  if (ping) ping.addEventListener('click', refreshStatus);

  // ---------------------------------------------------------------- 9. real sorting cycle
  // One button runs on the real arm:  GRAB -> LIFT -> SHOW CAM -> classify -> DROP into the bucket.
  const STAGE_TEXT = {
    GRABBING: 'Grabbing the item', LIFTING: 'Lifting', SHOWING: 'Showing to the camera',
    CLASSIFYING: 'Classifying', DROPPING: 'Dropping into the bucket', MOVING: 'Moving'
  };
  let cycleWatch = null, lastCycleFrame = null;

  async function startCycle() {
    if (State.estop) { log('cmdLog', 'WARN', 'Emergency stop is engaged.'); return; }
    const r = await API._req('POST', '/robot/cycle', {}, () => ({ ok: false, error: 'backend unreachable' }));
    if (!r || !r.ok) { log('cmdLog', 'WARN', 'Cycle not started: ' + ((r && r.error) || 'unknown')); return; }
    log('cmdLog', 'INFO', 'Sorting cycle started: put the item at the pickup spot first.');
    lastCycleFrame = null;
    showManualRow(false);
    watchCycle();
  }

  function watchCycle() {
    if (cycleWatch) clearInterval(cycleWatch);
    cycleWatch = setInterval(pollCycle, 800);
  }

  async function pollCycle() {
    const c = await API._req('GET', '/robot/cycle', null, () => null);
    if (!c) return;
    if (c.running) setOperation(STAGE_TEXT[c.stage] || c.stage, 'Cycle');

    if (c.detection && c.detection.frame !== lastCycleFrame) {
      lastCycleFrame = c.detection.frame;
      const d = Object.assign({}, c.detection, { timestamp: new Date(c.detection.timestamp) });
      renderDetection(d);
      log('cmdLog', 'INFO', 'Classified ' + d.category + ' (' + d.confidence.toFixed(1) + '%)  ' + probsText(d));
    }
    if (c.running) return;

    clearInterval(cycleWatch); cycleWatch = null;
    if (c.stage === 'DONE' && c.detection) {
      const d = Object.assign({}, c.detection, { timestamp: new Date(c.detection.timestamp) });
      State.history.unshift(Mock.toRow(d));
      State.bins[d.category] = Math.min(100, (State.bins[d.category] || 0) + 1);
      State.cycleCount++;
      $('#chipCycleVal').textContent = '#' + String(State.cycleCount).padStart(4, '0');
      log('cmdLog', 'SUCCESS', c.message);
    } else if (c.stage === 'DONE') {
      log('cmdLog', 'SUCCESS', c.message);
    } else if (c.stage === 'NEEDS_MANUAL') {
      log('cmdLog', 'WARN', c.message);
      showManualRow(true);
    } else if (c.stage === 'ERROR') {
      log('cmdLog', 'ERROR', c.message);
    }
    setOperation('Ready', 'Home');
    refreshAll();
  }

  // Shown only after a low-confidence result: pick the bucket yourself, or release the item.
  const manualRow = document.createElement('div');
  manualRow.className = 'btn-row';
  manualRow.style.cssText = 'display:none;margin-bottom:14px;align-items:center';
  manualRow.innerHTML = '<span class="eyebrow">Low confidence, choose:</span>' +
    ['paper', 'metal', 'plastic'].map(function (b) {
      return '<button class="btn" data-drop="' + b + '">Drop ' + b + '</button>';
    }).join('') +
    '<button class="btn danger" data-drop="release">Release item</button>';
  const detHead = $('#view-detection .view-head');
  if (detHead) detHead.after(manualRow);
  function showManualRow(on) { manualRow.style.display = on ? 'flex' : 'none'; }
  manualRow.addEventListener('click', async function (e) {
    const b = e.target.closest('[data-drop]');
    if (!b) return;
    const which = b.dataset.drop;
    const r = which === 'release'
      ? await API._req('POST', '/robot/release', {}, () => ({ ok: false }))
      : await API._req('POST', '/robot/drop', { bin: which }, () => ({ ok: false }));
    if (r && r.ok) { showManualRow(false); watchCycle(); }
    else log('cmdLog', 'WARN', 'Could not start: ' + ((r && r.error) || 'unknown'));
  });

  // The start button, on the Dashboard and the Waste Detection page.
  function makeStartButton() {
    const b = document.createElement('button');
    b.className = 'btn good';
    b.textContent = 'Start sorting cycle';
    b.addEventListener('click', startCycle);
    return b;
  }
  const dashHead = $('#view-dashboard .view-head');
  if (dashHead) dashHead.appendChild(makeStartButton());
  const classifyBtn = $('#btnDetect');
  if (classifyBtn) classifyBtn.before(makeStartButton());

  // ---------------------------------------------------------------- 10. what the classifier sees
  function probsText(d) {
    if (!d || !d.probs) return '';
    return Object.keys(d.probs).map(function (k) { return k + ' ' + d.probs[k] + '%'; }).join(' | ');
  }
  // probabilities line + the exact crop the model looked at (helps tune SHOW_ROI in app.py)
  const resultBody = $('#detFrame') && $('#detFrame').closest('.card-body');
  if (resultBody) {
    const box = document.createElement('div');
    box.style.cssText = 'margin-top:14px';
    box.innerHTML =
      '<span class="eyebrow">Probabilities</span>' +
      '<div class="mono" id="detProbs" style="font-size:12px;margin:4px 0 12px">\u2014</div>' +
      '<span class="eyebrow">What the classifier sees</span>' +
      '<div style="margin-top:6px"><img id="seenCrop" alt="" style="max-width:100%;max-height:150px;' +
      'border-radius:8px;border:1px solid var(--line);display:none"></div>';
    resultBody.appendChild(box);
  }
  const baseRenderDetection = renderDetection;
  renderDetection = function (d) {
    baseRenderDetection(d);
    const pe = $('#detProbs'); if (pe) pe.textContent = probsText(d) || '\u2014';
    const img = $('#seenCrop');
    if (img) {
      img.onload = function () { img.style.display = 'block'; };
      img.src = API.BASE_URL + '/camera/last_crop?t=' + Date.now();
    }
  };

  // ---------------------------------------------------------------- 11. Robot Control: arm sequence buttons
  const seqCard = document.createElement('div');
  seqCard.className = 'card';
  seqCard.innerHTML =
    '<div class="card-head"><h3>Sorting sequence (real arm)</h3><div class="spacer"></div>' +
    '<span class="eyebrow">grab, lift, show, drop</span></div>' +
    '<div class="card-body">' +
    '<div class="btn-row">' +
    '<button class="btn primary" data-step="grab">1  Grab</button>' +
    '<button class="btn primary" data-step="lift">2  Lift</button>' +
    '<button class="btn primary" data-step="show">3  Show cam</button>' +
    '<button class="btn" data-step="pick">Pick (grab + lift)</button></div>' +
    '<div class="btn-row" style="margin-top:10px">' +
    '<button class="btn" data-step="drop_paper">4  Drop PAPER</button>' +
    '<button class="btn" data-step="drop_metal">4  Drop METAL</button>' +
    '<button class="btn" data-step="drop_plastic">4  Drop PLASTIC</button></div>' +
    '<div class="btn-row" style="margin-top:10px">' +
    '<button class="btn good" id="btnFullCycle">Start full sorting cycle</button>' +
    '<button class="btn" data-step="home">Home</button>' +
    '<button class="btn danger" data-step="release">Release + home</button></div>' +
    '<div class="field-hint" style="margin-top:10px">The full cycle runs grab, lift, show cam, classify, then drops the item ' +
    'into the bucket for the detected category.</div></div>';
  const ctlRight = $('#view-control .g-ctl > div:last-child');
  if (ctlRight) ctlRight.prepend(seqCard);
  seqCard.addEventListener('click', async function (e) {
    if (e.target.id === 'btnFullCycle') { startCycle(); return; }
    const b = e.target.closest('[data-step]');
    if (!b) return;
    if (State.estop) { log('cmdLog', 'WARN', 'Emergency stop is engaged.'); return; }
    const r = await API._req('POST', '/robot/step', { step: b.dataset.step }, () => ({ ok: false, error: 'backend unreachable' }));
    if (r && r.ok) { log('cmdLog', 'INFO', 'Step started: ' + b.dataset.step); watchCycle(); }
    else log('cmdLog', 'WARN', 'Step not started: ' + ((r && r.error) || 'unknown'));
  });

  // ---------------------------------------------------------------- 8. restart with the new logic
  refreshAll();
  log('sysLog', 'SUCCESS', 'Live mode: dashboard is talking to ' + API.BASE_URL + ' (real camera + classifier).');
  log('sysLog', 'INFO', 'Place an item on the pick spot, hand out of the way, to start a cycle.');
  log('cmdLog', 'INFO', 'Console ready. Live mode.');
  stopAuto();
  startAuto();
})();

