# SWE Report: Fill-in Guide for the Highlighted Placeholders

Each item is tagged:
- **[Have]** the information comes from the work done in this project, so you can paste it.
- **[Confirm]** I can suggest text, but you must check it against your final system.
- **[Missing]** I have no data. Only you can supply it. I explain how to get it.

---

## 0. Fix these first (consistency problems in the draft)

**1. The classifier is named YOLOv8n-cls (ONNX), but everything built in this project uses MobileNetV2 in Keras.**
The training script, `waste_classifier.h5`, the Flask backend and the dashboard all use MobileNetV2 with transfer learning. I have no record of a YOLOv8 model being trained. The same draft says 91.67% validation accuracy. That number came from the MobileNetV2 training run (110 of 120 validation images).
Check and correct these places if your final system uses the Keras model:
- Table 9.1, row 1: "with YOLOv8n-cls (ONNX)"
- Section 6.2, M1: "YOLOv8n-cls model in ONNX format"
- Listing 6.3.1 title: "Python, ONNX Runtime"
- Figure 8.1 placeholder: "training curves from YOLOv8 training"

If you really did train a separate YOLOv8n-cls model, use its own numbers and ignore my figures.

**2. "Flask backend commands ESP32 over Wi-Fi/HTTP" is not something we built here.**
The backend in this project sends one word (`PLASTIC`, `METAL`, `PAPER`, `HOME`) over **USB serial**. Only tick "Met" and describe Wi-Fi if you have run it on your final system. The design makes the switch easy, because only the `ArmLink.send()` function talks to the arm. The routes stay the same.

**3. The report says `DataService`. The dashboard code calls it `API`.** Rename one of them so the code listing matches the text.

**4. About 91.67%.** It is 110 correct out of 120 validation images. With that sample size the 95% confidence range is roughly 87% to 97%. Saying "met the 90% target on the validation set" is fair. Also state that the images came from the same dataset as the training images.

---

## 1. Table 3.3: component cost breakdown

- **[Have]** Your DA-1 progress report (the DOCX with 24BRS1216 and 24BRS1185 in its name) already has a corrected component table that totals ₹2,568.40. Copy the quantities and prices from it.
- **[Confirm]** The BenchRobotics design this arm is based on uses 3 MG996R and 3 SG90 servos, plus 1 ESP32 and 1 PCA9685. Check that against your invoices.
- **[Missing]** I have no prices. Use the miscellaneous row (power supply, wiring, filament) as the balancing figure so the rows add up to ₹2,568.40. The laptop webcam costs nothing, so say that in a note.

---

## 2. Section 6.2: module description

**M1: dataset placeholder [Have, for the Keras model]**
> The model was trained on 600 images from the Kaggle Garbage Classification dataset: 200 images for each of plastic, metal and paper (160 for training and 40 for validation per class).

Also change the model name in M1 to "a MobileNetV2 transfer-learning model (Keras)". Images are resized to 160 × 160. The model rescales pixel values to the range −1 to 1 inside the network.

**M2: endpoint list [Have, for the backend in this project]**

| Endpoint | Purpose |
|---|---|
| `GET /api/detection/latest` | Detects a new item, classifies it, returns category, confidence and bounding box |
| `GET /api/camera/stream` | Live MJPEG video for the dashboard |
| `POST /api/camera/reset_background` | Captures the empty-spot reference ("Set empty background") |
| `POST /api/robot/command` | Sends `PLASTIC`, `METAL`, `PAPER` or `HOME` to the arm |
| `POST /api/robot/joint`, `/mode`, `/estop` | Manual joint control, mode switch, emergency stop |
| `GET /api/status` | Health of camera, model, arm link and server |
| `GET /api/analytics`, `/api/history` | Counts, outcomes, hourly series, event log |
| `POST /api/settings` | Saves dashboard settings |

If your final backend uses other routes such as `/api/classify` or `/api/sort`, list those instead.

**M4: "[Credit the teammate who built this page]" [Missing]**
I do not know the name. Suggested wording: "The on-board calibration page was built by [teammate name] and is used for jogging joints and saving the six named positions."

---

## 3. Section 6.3: code listings (15 to 30 lines each)

**6.3.1 Classifier inference** (retitle it "Python, Keras")
```python
CATEGORIES = ["plastic", "metal", "paper"]   # same order as training
IMG_SIZE = (160, 160)

def load_classifier():
    return keras.models.load_model(MODEL_FILE)

def classify(image_bgr):
    """Returns (category, confidence_percent, latency_ms)."""
    img = cv2.cvtColor(cv2.resize(image_bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(img.astype("float32"), axis=0)   # rescaling is built into the model
    t0 = time.time()
    with model_lock:
        preds = model.predict(batch, verbose=0)[0]
    latency = (time.time() - t0) * 1000.0
    idx = int(np.argmax(preds))
    return CATEGORIES[idx], float(preds[idx]) * 100.0, latency
```

**6.3.2 Flask sort endpoint.** In this backend, classification is in `/api/detection/latest` and sending the command is in `/api/robot/command`. This listing is the command route. It uses USB serial. If your final route talks HTTP to the ESP32, paste your own code.
```python
ARM_COMMANDS = {"PLASTIC", "METAL", "PAPER", "HOME"}

def _run_arm_command(cmd):
    event = state.get("last_event")
    try:
        reply = arm.send(cmd)                 # one word out, waits for "DONE"
    finally:
        state["busy"] = False
    if event and event["status"] == "Detected" and event["category"].upper() == cmd:
        event["status"] = "Sorted" if reply == "DONE" else "Failed"

@app.post("/api/robot/command")
def robot_command():
    cmd = str((request.get_json(silent=True) or {}).get("command", "")).strip()
    word = cmd.upper()
    if word not in ARM_COMMANDS:
        return jsonify({"ok": True, "forwarded": False, "command": cmd})
    if state["estop"]:
        return jsonify({"ok": False, "forwarded": False, "error": "emergency stop is engaged"})
    if state["busy"]:
        return jsonify({"ok": False, "forwarded": False, "error": "arm is busy"})
    state["busy"] = True
    threading.Thread(target=_run_arm_command, args=(word,), daemon=True).start()
    return jsonify({"ok": True, "forwarded": True, "command": word})
```

**6.3.3 ESP32 servo write.** The `pwm` lines below come from the wiring-check sketch. The `server.on(...)` handler lives in your teammate's firmware, so paste it from there.
```cpp
const int SERVO_MIN = 150;   // pulse "ticks" for 0 degrees
const int SERVO_MAX = 550;   // pulse "ticks" for 180 degrees

int angleToTicks(int angle) {
  return SERVO_MIN + (long)(SERVO_MAX - SERVO_MIN) * angle / 180;
}

// in setup():
pwm.begin();
pwm.setOscillatorFrequency(27000000);
pwm.setPWMFreq(50);                       // standard servo frequency

// to move a joint:
pwm.setPWM(TEST_CHANNEL, 0, angleToTicks(90));
```

**6.3.4 Dashboard mock versus fetch.** In the shared version the object is called `API`.
```javascript
const API = {
  BASE_URL: 'http://192.168.1.42:5000/api',
  USE_MOCK: true,

  async _req(method, path, body, fallback){
    if(this.USE_MOCK) return fallback();           // simulator
    try{
      const opts = method === 'GET' ? {}
        : {method:'POST', headers:{'Content-Type':'application/json'},
           body:JSON.stringify(body)};
      const res = await fetch(this.BASE_URL + path, opts);   // live backend
      if(!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      Net.recover();
      return data;
    }catch(err){
      Net.drop(path, err);
      return fallback();
    }
  },

  getDetection()   { return this._req('GET',  '/detection/latest', null, Mock.detection); },
  sendCommand(cmd) { return this._req('POST', '/robot/command', {command:cmd}, () => Mock.ack({cmd})); }
};
```

---

## 4. Section 7: tools

**7.3 Git and GitHub [Missing]**
I cannot see your repository. Get the numbers in the project folder:
- Commit count: `git rev-list --count HEAD`
- Branches: `git branch -a`
- Recent history for the screenshot: `git log --oneline -15`

**7.4 VS Code screenshot [Have]**
Run `python backend/app.py` and capture the terminal showing the line "SegreBot is starting. Open http://127.0.0.1:5000".

**7.5 Arduino IDE screenshot [Have]**
Upload `pca9685_check.ino` and capture the Serial Monitor at 115200 showing the I2C scan ("device found at 0x40") and the servo messages.

**Summary table last row [Confirm].** Postman was not used in this project. Colab was not used either, since training ran locally on a laptop CPU. Rows that are true:

| Tool | Purpose | Phase | Usage |
|---|---|---|---|
| Kaggle | Dataset source | Data preparation | Garbage Classification images for plastic, metal and paper |
| Python venv and pip | Environment management | Implementation | Isolated environment for Flask, OpenCV and Keras |

---

## 5. Section 8: testing

**Test cases TC02 to TC17.** I have no recorded outputs from your hardware runs, so do not invent them. This table shows what I can support and what you still have to run.

| TC | Support available | You still need to |
|---|---|---|
| TC04 empty spot | Automated check: an empty spot gives no detection. Low-confidence items are skipped and logged. | Run on the live system with an unknown object |
| TC05, TC06 sort commands | Automated check (simulated arm): the command is forwarded, and a second command is refused while busy. Demo video (21 Oct): the arm picked a can, carried it over the metal bin and released it, but the can landed on the table beside the bin, so do not mark TC05 as Pass from that run | Re-run after fixing the bin pose and record the result |
| TC07 slider limit | Slider ranges and step buttons clamp to each joint's limits in the dashboard code | Confirm the firmware also clamps |
| TC08 home | `home` is translated to `HOME` (automated check) | Run on the arm |
| TC10 full cycle | Automated check: the analytics count rises by one | Check the History page on live hardware |
| TC11 threshold | The new threshold applies to the next detection (dashboard code) | Run once and record |
| TC15 emergency stop | In this backend the stop blocks new commands only | See the warning below |
| TC02, TC03, TC09, TC12 to TC14, TC17 | None | Run and record |

**Warning on TC15.** The expected output says "All motion halts immediately". In the backend written here, the stop prevents new commands but cannot interrupt a move already in progress. Only mark it Pass if your final firmware has a stop command and you tested it. Otherwise reword the expected output to "no new commands are accepted and the on-screen motion halts".

**Section 8.3 results text [Have]**
> The classifier achieved 91.67% validation accuracy (110 of 120 images) on the held-out set. The validation images come from the same dataset as the training images, so accuracy on live webcam frames is lower.

For per-class results and the confusion matrix **[Missing]**: I can write a small script that runs your 120 validation images through `waste_classifier.h5` and saves a confusion matrix image and per-class accuracy. Training curves were not saved, so a one-minute retrain with the history saved would produce them.

**Bug log (Table 8.2) [Have, choose 4 of these 5]**

| ID | Description | Severity | Fix | Status |
|---|---|---|---|---|
| B01 | Saved classifier failed to reload with "Unknown layer: TrueDivide" | High | Replaced the raw TensorFlow preprocessing operation with a built-in Rescaling layer and retrained | Fixed |
| B02 | Dataset downloads failed (3.6 GB TrashNet error, and an "Invalid string class label" error on a second dataset) | Medium | Switched to the 82 MB Kaggle zip downloaded by browser and wrote a script to split it 160/40 | Fixed |
| B03 | Arm did not move after rewiring because the PCA9685 V+ (servo power) wire was unconnected | High | Connected V+ for the test, then moved to an external 5V supply | Fixed (confirm final supply) |
| B04 | Servos shook and behaved erratically on laptop USB or a power bank (voltage dips) | High | Dedicated 5V supply of 3A or more on the PCA9685 terminals with a shared ground | Confirm: open or mitigated |
| B05 | Backend crashed on start with "Keras cannot be imported" | Medium | Imported Keras first and loaded the model through `keras.models.load_model`, bypassing `tf.keras` | Confirm after your run |
| B06 | In the demo video the can was released beside the metal bin, not inside it. The gripper held the can sideways and past the bin's far edge | Medium | Move the metal-bin pose forward and lower, and rotate the wrist so the can drops into the opening. Re-test | Open |

---

## 6. Section 9: results

**Table 9.1**
- **Row 1** [Confirm]: "91.67% validation accuracy (110 of 120 images) with MobileNetV2 transfer learning (Keras)".
- **Row 2, success rate** [Have, partly]: the demo video shows the pick, lift and carry working, but the can landed beside the metal bin. Mark this row "Partially met". For a number, run 10 cycles per bin, count the items that land in the right bin, and report X of 30.
- **Row 4, "live vs mock"** [Have, for the dashboard built here]:
  > Waste Detection, Robot Control, System Status and Analytics read live backend data. Activity History is kept in the browser session, and bin fill levels are counters, not sensor readings.

**The paragraph under Table 9.1**
- **Errors observed** [Have]. Replace the examples with these real ones:
  - Early webcam tests gave low confidence (0.42 to 0.52 on some items), because the training photos were studio images. The item is now cropped before classification and the confidence threshold is 60%.
  - Servo shaking when powered from USB or a power bank (bug B04).
- **The 1137 ms latency** [Confirm]. The first prediction after startup is slow because TensorFlow warms up. Add one line after the model loads, then measure the average of 20 predictions:
  ```python
  model.predict(np.zeros((1, 160, 160, 3), dtype="float32"), verbose=0)   # warm-up
  ```
  Quote the steady-state number next to the 1.5 s target.
- **Added latency "[X ms]"** [Missing]: time 20 command round trips (send, then wait for the reply) and report the average.

**Figures**
- **9.1 and 9.2** [Have, from the demo video]: use the images in the `figures` folder: `fig_9_1_assembled_arm_three_bins.png`, `fig_9_2_arm_mid_cycle_over_metal_bin.png` and the four-step `fig_9_2_sort_sequence_4_steps.png`. A phone photo will be sharper than a video frame if you have one.
- **9.4 Analytics** [Missing]: do 8 to 10 sorts first so the charts have data, with the green connection chip showing.
- **9.5 History and System Status** [Missing]: capture both pages in live mode.

---

## 7. What only you can supply

Photos, GitHub numbers, test-case outputs for TC02 to TC17, the sorting success rate, the Wi-Fi latency, per-class results and the confusion matrix, the teammate's name, and component prices.
