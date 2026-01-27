#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Raspberry Pi 4 â€“ 2-Motor Bot (REMOTE + AUTO), 3x Ultrasonic obstacle sensing,

DHT11 + MAX30100 sensors, Web UI (Flask) + SSE, optional Bluetooth SPP (/dev/rfcomm0).



Author: you + ChatGPT

"""



import os
import sys
import time
import json
import math
import queue
import signal
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
# ---- GPIO & Sensors ----
import RPi.GPIO as GPIO
import adafruit_dht
import board
# I2C for MAX30100

try:

    from smbus2 import SMBus

except:

    from smbus import SMBus  # fallback



# ---- Web (Flask) ----

from flask import Flask, request, Response, jsonify, make_response

# ---- API Client ----
from api_client import WheelchairAPIClient, get_sensor_snapshot



# ====================== USER CONFIG ======================

# API Configuration (for remote Next.js server)
API_URL = "https://your-app.vercel.app"  # Replace with your Vercel URL
DEVICE_ID = "wheelchair-rpi-001"  # Unique identifier for this device
API_UPLOAD_ENABLED = True  # Set to False to disable API uploads

HTTP_PORT = 8080



# Motor pins (BCM)

M_L_IN1, M_L_IN2 = 17, 27

M_R_IN1, M_R_IN2 = 22, 23



# Ultrasonics (BCM)

US_FRONT_TRIG, US_FRONT_ECHO = 5, 6

US_LEFT_TRIG,  US_LEFT_ECHO  = 13, 19

US_RIGHT_TRIG, US_RIGHT_ECHO = 26, 21



# DHT11

DHT_PIN = 4

DHT_TYPE = adafruit_dht.DHT11



# MAX30100

MAX30100_I2C_ADDR = 0x57



# Safety / behavior

OBSTACLE_CM = 30.0        # stop below this distance

ULTRA_HZ = 10             # polling rate of ultrasonic thread

SENSORS_HZ = 1            # DHT + MAX30100 update rate

AUTO_LOOP_DELAY = 0.01    # loop granularity in AUTO

REMOTE_IDLE_BRAKE = True  # stop motors when no command



# Optional Bluetooth SPP device

BT_SERIAL_DEV = "/dev/rfcomm0"   # set to None to disable

BT_BAUD = 9600



# =========================================================



GPIO.setmode(GPIO.BCM)

GPIO.setwarnings(False)



# ---------------------- Motor Driver ----------------------

class MotorDriver:

    def __init__(self, l_in1, l_in2, r_in1, r_in2):

        self.l_in1, self.l_in2 = l_in1, l_in2

        self.r_in1, self.r_in2 = r_in1, r_in2

        for p in (l_in1, l_in2, r_in1, r_in2):

            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)

        self._lock = threading.Lock()

        self._last_cmd = "S"



    def _set(self, l1, l2, r1, r2):

        GPIO.output(self.l_in1, GPIO.HIGH if l1 else GPIO.LOW)

        GPIO.output(self.l_in2, GPIO.HIGH if l2 else GPIO.LOW)

        GPIO.output(self.r_in1, GPIO.HIGH if r1 else GPIO.LOW)

        GPIO.output(self.r_in2, GPIO.HIGH if r2 else GPIO.LOW)



    def stop(self):

        with self._lock:

            self._set(0,0,0,0)

            self._last_cmd = "S"



    def forward(self):

        with self._lock:

            self._set(1,0,1,0)

            self._last_cmd = "F"



    def back(self):

        with self._lock:

            self._set(0,1,0,1)

            self._last_cmd = "B"



    def left(self):

        with self._lock:

            self._set(0,1,1,0)

            self._last_cmd = "L"



    def right(self):

        with self._lock:

            self._set(1,0,0,1)

            self._last_cmd = "R"



    def last(self):

        with self._lock:

            return self._last_cmd



# ---------------------- Ultrasonic ------------------------

class Ultra3:

    def __init__(self, pins: List[Tuple[int,int]]):

        self.pins = pins  # [(trig, echo), ...] order: Front, Left, Right

        for trig, echo in pins:

            GPIO.setup(trig, GPIO.OUT, initial=GPIO.LOW)

            GPIO.setup(echo, GPIO.IN)

        self.distances = {"front": None, "left": None, "right": None}

        self._stop = threading.Event()



    @staticmethod

    def _measure_pair(trig, echo, timeout=0.03):

        # trigger

        GPIO.output(trig, GPIO.HIGH)

        time.sleep(10e-6)

        GPIO.output(trig, GPIO.LOW)



        # wait for echo HIGH

        start = time.time()

        while GPIO.input(echo) == 0:

            if time.time() - start > timeout:

                return None

        t0 = time.time()



        # wait for echo LOW

        while GPIO.input(echo) == 1:

            if time.time() - t0 > timeout:

                return None

        t1 = time.time()



        dt = t1 - t0

        # distance in cm = (time * speed_of_sound)/2

        return (dt * 34300.0) / 2.0



    def loop(self, hz=10):

        period = 1.0 / max(1, hz)

        labels = ["front", "left", "right"]

        while not self._stop.is_set():

            for i, (trig, echo) in enumerate(self.pins):

                d = self._measure_pair(trig, echo)

                self.distances[labels[i]] = d

                time.sleep(0.005)

            time.sleep(max(0.0, period - 3*0.005))



    def stop(self):

        self._stop.set()



# ---------------------- DHT11 -----------------------------

class DHT11Reader:

    def __init__(self, pin_bcm):

        # Map BCM pin to a board pin object for CircuitPython

        bcm_to_board = {

            4: board.D4, 17: board.D17, 27: board.D27, 22: board.D22,

            5: board.D5, 6: board.D6, 13: board.D13, 19: board.D19,

            26: board.D26, 21: board.D21, 20: board.D20, 16: board.D16,

            12: board.D12, 25: board.D25, 24: board.D24, 23: board.D23,

            18: board.D18

        }

        self.temp_c = None

        self.hum = None

        self._stop = threading.Event()

        pin_obj = bcm_to_board.get(pin_bcm)

        if pin_obj is None:

            raise RuntimeError(f"Unsupported DHT pin (BCM {pin_bcm}) for CircuitPython map")

        # DHT11 device

        self._dht = adafruit_dht.DHT11(pin_obj)  # use_pulseio=False works well on Pi



    def loop(self, hz=1):

        period = 1.0 / max(1, hz)

        while not self._stop.is_set():

            try:

                t = self._dht.temperature
                print(t)
                h = self._dht.humidity
                print(h)
                if (t is not None) and (h is not None):

                    self.temp_c = float(t)

                    self.hum = float(h)

            except RuntimeError:

                # DHTs are noisy; just try again next tick

                pass

            except Exception:

                # Hard failure ï¿½ re-init to be safe

                try:

                    self._dht.exit()

                except Exception:

                    pass

                time.sleep(0.2)

                # No need to remap; object already set

                pass

            time.sleep(period)



    def stop(self):

        self._stop.set()

        try:

            self._dht.exit()

        except Exception:

            pass



# ---------------------- MAX30100 (basic) ------------------

class MAX30100:

    """

    Minimal MAX30100 reader: configures SPO2 mode and reads raw IR/RED averages.

    (Full HR/SpO2 algorithms need filtering; here we expose raw values and a very rough HR.)

    """

    REG_INT_STATUS     = 0x00

    REG_INT_ENABLE     = 0x01

    REG_FIFO_WR_PTR    = 0x02

    REG_OVF_CTR        = 0x03

    REG_FIFO_RD_PTR    = 0x04

    REG_FIFO_DATA      = 0x05

    REG_MODE_CONFIG    = 0x06

    REG_SPO2_CONFIG    = 0x07

    REG_LED_CONFIG     = 0x09

    REG_TEMP_INT       = 0x16

    REG_TEMP_FRAC      = 0x17

    REG_REV_ID         = 0xFE

    REG_PART_ID        = 0xFF



    MODE_HRONLY = 0x02

    MODE_SPO2   = 0x03



    def __init__(self, busno=1, addr=MAX30100_I2C_ADDR):

        self.addr = addr

        self.bus = SMBus(busno)

        self.present = False

        self.ir = None

        self.red = None

        self.hr = None

        self.spo2 = None

        try:

            part = self.bus.read_byte_data(self.addr, self.REG_PART_ID)

            if part in (0x11, 0x00, 0xff):  # some clones return 0x11

                self.present = True

                self._init_sensor()

        except Exception:

            self.present = False



    def _write(self, reg, val):

        self.bus.write_byte_data(self.addr, reg, val)



    def _read(self, reg):

        return self.bus.read_byte_data(self.addr, reg)



    def _init_sensor(self):
        # Reset FIFO
        self._write(self.REG_FIFO_WR_PTR, 0x00)
        self._write(self.REG_OVF_CTR,     0x00)
        self._write(self.REG_FIFO_RD_PTR, 0x00)

        # SPO2 mode
        self._write(self.REG_MODE_CONFIG, self.MODE_SPO2)

        # SPO2 config: HI_RES + SR=100Hz + PW=1600us
        self._write(self.REG_SPO2_CONFIG, 0x27)

        # LED config (1 byte!): upper nibble=RED, lower nibble=IR (each 0..0xF)
        # 0x4 ~= 12.6 mA. If you get zeros, try 0x66 or 0x77.
        self._write(self.REG_LED_CONFIG, 0x44)



    def read_average(self, nsamples=8):

        if not self.present:

            return None, None

        ir_vals, red_vals = [], []

        for _ in range(nsamples):

            try:

                # Each sample: IR(16-bit), RED(16-bit)

                data = self.bus.read_i2c_block_data(self.addr, self.REG_FIFO_DATA, 4)

                ir = (data[0] << 8) | data[1]

                red = (data[2] << 8) | data[3]

                ir_vals.append(ir)

                red_vals.append(red)

            except Exception:

                break

            time.sleep(0.01)

        if not ir_vals:

            return None, None

        return sum(ir_vals)/len(ir_vals), sum(red_vals)/len(red_vals)



# ---------------------- State / Control -------------------

@dataclass

class AutoStep:

    cmd: str      # "F","B","L","R","S"

    secs: float



@dataclass

class BotState:

    mode: str = "REMOTE"                 # "REMOTE" or "AUTO"

    auto_plan: List[AutoStep] = field(default_factory=list)

    auto_index: int = 0

    auto_started_at: Optional[float] = None

    obstacle_hit: Optional[str] = None   # "front"/"left"/"right" or None

    last_bt_cmd: Optional[str] = None



# ---------------------- App Wiring ------------------------

motors = MotorDriver(M_L_IN1, M_L_IN2, M_R_IN1, M_R_IN2)

ultra = Ultra3([(US_FRONT_TRIG,US_FRONT_ECHO),

                (US_LEFT_TRIG,US_LEFT_ECHO),

                (US_RIGHT_TRIG,US_RIGHT_ECHO)])

dht = DHT11Reader(DHT_PIN)

max30 = None

try:

    max30 = MAX30100()

except Exception:

    max30 = None



state = BotState()

state_lock = threading.Lock()



# ---------------------- Worker Threads --------------------

def ultrasonic_thread():

    ultra.loop(hz=ULTRA_HZ)



def sensors_thread():

    global max30

    while True:

        # DHT is updated inside its own loop

        # MAX30100

        if max30 and max30.present:

            ir, red = max30.read_average(nsamples=8)

            if ir is not None:

                max30.ir, max30.red = ir, red

                # Naive heart-rate placeholder (NOT medical): just detect pulse-ish ratio

                # If you need real HR/SpO2, use a proper library/algorithm.

                max30.hr = None

                max30.spo2 = None

        time.sleep(1.0 / max(1, SENSORS_HZ))



def dht_thread():

    dht.loop(hz=SENSORS_HZ)



def safety_watchdog_thread():

    # monitors distances and applies emergency stop

    while True:

        f = ultra.distances.get("front")

        l = ultra.distances.get("left")

        r = ultra.distances.get("right")

        hit = None

        if f is not None and f < OBSTACLE_CM:

            hit = "front"

        elif l is not None and l < OBSTACLE_CM:

            hit = "left"

        elif r is not None and r < OBSTACLE_CM:

            hit = "right"



        with state_lock:

            state.obstacle_hit = hit

        if hit:

            motors.stop()

        time.sleep(0.02)



def auto_runner_thread():

    # executes the time-based plan when in AUTO mode

    while True:

        with state_lock:

            is_auto = (state.mode == "AUTO")

            plan = list(state.auto_plan)

            idx = state.auto_index

            started = state.auto_started_at

            obstacle = state.obstacle_hit



        if not is_auto or not plan:

            time.sleep(0.05)

            continue



        if obstacle:

            motors.stop()

            time.sleep(0.05)

            continue



        if idx >= len(plan):

            motors.stop()

            # Finished plan -> remain in AUTO but idle

            time.sleep(0.1)

            continue



        step = plan[idx]

        # Kick off step if not started

        if started is None:

            # issue command

            if   step.cmd == "F": motors.forward()

            elif step.cmd == "B": motors.back()

            elif step.cmd == "L": motors.left()

            elif step.cmd == "R": motors.right()

            else:                 motors.stop()

            with state_lock:

                state.auto_started_at = time.time()

        else:

            elapsed = time.time() - started

            if elapsed >= step.secs:

                motors.stop()

                with state_lock:

                    state.auto_index += 1

                    state.auto_started_at = None



        time.sleep(AUTO_LOOP_DELAY)



def bluetooth_reader_thread():

    if not BT_SERIAL_DEV:

        return

    import serial

    while True:

        try:

            with serial.Serial(BT_SERIAL_DEV, BT_BAUD, timeout=1) as ser:

                while True:

                    line = ser.readline().decode(errors='ignore').strip().upper()

                    if not line:

                        continue

                    # Expected simple commands: F,B,L,R,S or JSON {"cmd":"F","secs":2}

                    try:

                        if line.startswith("{"):

                            msg = json.loads(line)

                            c = msg.get("cmd","S").upper()

                            secs = float(msg.get("secs", 0))

                        else:

                            parts = line.split()

                            c = parts[0]

                            secs = float(parts[1]) if len(parts) > 1 else 0

                    except Exception:

                        c, secs = "S", 0



                    with state_lock:

                        state.last_bt_cmd = f"{c} {secs}".strip()



                    if c in ("F","B","L","R","S"):

                        if secs > 0:

                            # timed single-shot

                            if   c == "F": motors.forward()

                            elif c == "B": motors.back()

                            elif c == "L": motors.left()

                            elif c == "R": motors.right()

                            else:          motors.stop()

                            t0 = time.time()

                            while time.time() - t0 < secs:

                                if state.obstacle_hit:

                                    break

                                time.sleep(0.02)

                            motors.stop()

                        else:

                            if   c == "F": motors.forward()

                            elif c == "B": motors.back()

                            elif c == "L": motors.left()

                            elif c == "R": motors.right()

                            else:          motors.stop()

        except Exception:

            # wait and retry open

            time.sleep(2.0)



def api_upload_thread():
    """Thread to periodically upload sensor data to Next.js API"""
    if not API_UPLOAD_ENABLED:
        print("API uploads disabled")
        return
    
    # Initialize API client
    api_client = WheelchairAPIClient(API_URL, DEVICE_ID)
    
    print(f"🌐 API Client initialized: {API_URL}")
    print(f"📱 Device ID: {DEVICE_ID}")
    
    # Check initial status
    status = api_client.check_status()
    if status.get('killswitch'):
        print("⚠️  WARNING: API Killswitch is ENABLED!")
    else:
        print("✅ API Status: Active")
    
    while True:
        try:
            # Collect sensor data
            sensor_data = get_sensor_snapshot(motors, ultra, dht, max30, state, state_lock)
            
            # Upload to API
            success = api_client.upload_sensor_data(sensor_data)
            
            # Periodic status check (every 30 uploads)
            if int(time.time()) % 30 == 0:
                status = api_client.check_status()
                if status.get('killswitch'):
                    print("⚠️  API Killswitch ENABLED - uploads blocked")
            
        except Exception as e:
            print(f"API upload error: {e}")
        
        time.sleep(1.0)  # Upload every second


# ---------------------- Flask Web -------------------------

app = Flask(__name__, static_folder=None)



INDEX_HTML = """<!doctype html>

<html>

<head>

  <meta charset="utf-8">

  <title>RPi Bot â€“ Remote & Auto</title>

  <meta name="viewport" content="width=device-width, initial-scale=1">

  <style>

    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; margin:20px; background:#0b1220; color:#e7eefb}

    .row{display:flex; gap:12px; flex-wrap:wrap}

    .card{background:#111a2e; border-radius:14px; padding:16px; box-shadow:0 8px 28px rgba(0,0,0,.25)}

    button{padding:10px 16px; border:0; border-radius:12px; font-weight:600; cursor:pointer}

    .btn{background:#1f6feb; color:white}

    .btn:active{transform:scale(.98)}

    .grid{display:grid; grid-template-columns:repeat(3, minmax(120px,1fr)); gap:12px}

    .lcd{font-family:monospace; background:#071126; padding:10px; border-radius:8px}

    .muted{opacity:.8}

    .danger{color:#ff6b6b}

    .ok{color:#16c798}

    .pill{display:inline-block; padding:3px 10px; border-radius:999px; background:#0f1b36; margin-left:8px}

    input,textarea{background:#0f1b36; border:1px solid #23335c; color:#e7eefb; border-radius:8px; padding:8px}

    textarea{width:100%; height:120px}

    .small{font-size:12px; color:#9fb4d1}

  </style>

</head>

<body>

  <h2>RPi Bot <span id="mode-pill" class="pill">mode: â€¦</span></h2>



  <div class="row">

    <div class="card" style="flex:1">

      <h3>Remote Controls</h3>

      <div class="grid">

        <button class="btn" onclick="send('F')">Forward</button>

        <button class="btn" onclick="send('S')">Stop</button>

        <button class="btn" onclick="send('B')">Back</button>

        <button class="btn" onclick="send('L')">Left</button>

        <button class="btn" onclick="send('R')">Right</button>

        <button class="btn" onclick="setMode('REMOTE')">Mode: REMOTE</button>

      </div>

      <p class="small muted">Tip: In REMOTE mode, obstacle detection still stops the bot if an object is too close.</p>

    </div>



    <div class="card" style="flex:1">

      <h3>Auto Plan</h3>

      <p class="small">JSON steps: e.g. <code>[{"cmd":"F","secs":10},{"cmd":"L","secs":10},{"cmd":"R","secs":10}]</code></p>

      <textarea id="plan">[{"cmd":"F","secs":5},{"cmd":"L","secs":2},{"cmd":"F","secs":5},{"cmd":"R","secs":2},{"cmd":"B","secs":3}]</textarea>

      <div class="row" style="margin-top:8px">

        <button class="btn" onclick="uploadPlan()">Upload Plan</button>

        <button class="btn" onclick="setMode('AUTO')">Mode: AUTO</button>

        <button class="btn" onclick="resetAuto()">Reset Auto</button>

      </div>

    </div>

  </div>



  <div class="row" style="margin-top:16px">

    <div class="card" style="flex:1">

      <h3>Sensors</h3>

      <div class="grid">

        <div class="lcd">DHT11<br/>Temp: <span id="t">â€“</span> Â°C<br/>Hum: <span id="h">â€“</span> %</div>

        <div class="lcd">Ultrasonic<br/>Front: <span id="uf">â€“</span> cm<br/>Left: <span id="ul">â€“</span> cm<br/>Right: <span id="ur">â€“</span> cm</div>

        <div class="lcd">MAX30100<br/>IR: <span id="ir">â€“</span><br/>RED: <span id="red">â€“</span><br/><span class="small muted">HR/SpOâ‚‚ demo only</span></div>

      </div>

      <p>Obstacle: <span id="obs" class="pill">none</span></p>

    </div>



    <div class="card" style="flex:1">

      <h3>Status</h3>

      <p>Mode: <b id="mode">â€¦</b></p>

      <p>Last motor cmd: <b id="last">S</b></p>

      <p>Auto index: <b id="aidx">0</b></p>

      <p class="muted small">Bluetooth last cmd: <span id="bt">â€“</span></p>

    </div>

  </div>



<script>

function send(c){

  fetch('/cmd?c='+encodeURIComponent(c));

}

function setMode(m){

  fetch('/mode?m='+encodeURIComponent(m));

}

function uploadPlan(){

  let txt = document.getElementById('plan').value;

  try{

    let plan = JSON.parse(txt);

    fetch('/auto/plan', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(plan)});

  }catch(e){ alert('Invalid JSON'); }

}

function resetAuto(){

  fetch('/auto/reset', {method:'POST'});

}

const ev = new EventSource('/stream');

ev.onmessage = (e)=>{

  let s = JSON.parse(e.data);

  document.getElementById('t').textContent = s.dht_temp ?? 'â€“';

  document.getElementById('h').textContent = s.dht_hum  ?? 'â€“';

  document.getElementById('uf').textContent = s.us_front ?? 'â€“';

  document.getElementById('ul').textContent = s.us_left  ?? 'â€“';

  document.getElementById('ur').textContent = s.us_right ?? 'â€“';

  document.getElementById('ir').textContent = s.ir ?? 'â€“';

  document.getElementById('red').textContent = s.red ?? 'â€“';

  document.getElementById('mode').textContent = s.mode;

  document.getElementById('last').textContent = s.last;

  document.getElementById('aidx').textContent = s.auto_index;

  document.getElementById('bt').textContent = s.last_bt || 'â€“';

  let obs = document.getElementById('obs');

  obs.textContent = s.obstacle || 'none';

  obs.className = 'pill ' + (s.obstacle ? 'danger' : 'ok');

  document.getElementById('mode-pill').textContent = 'mode: ' + s.mode;

};

</script>

</body>

</html>

"""



@app.route("/")

def index():

    resp = make_response(INDEX_HTML)

    resp.headers["Content-Type"] = "text/html; charset=utf-8"

    return resp



@app.route("/cmd")

def http_cmd():

    c = (request.args.get("c","S") or "S").upper()

    # Always force REMOTE for direct commands

    with state_lock:

        state.mode = "REMOTE"

    if   c == "F": motors.forward()

    elif c == "B": motors.back()

    elif c == "L": motors.left()

    elif c == "R": motors.right()

    else:          motors.stop()

    return "OK"



@app.route("/mode")

def http_mode():

    m = (request.args.get("m","REMOTE") or "REMOTE").upper()

    if m not in ("REMOTE","AUTO"):

        m = "REMOTE"

    with state_lock:

        state.mode = m

        # reset auto timing if switching into AUTO with no plan

        if m == "AUTO" and not state.auto_plan:

            state.auto_index = 0

            state.auto_started_at = None

    return "OK"



@app.route("/auto/plan", methods=["POST"])

def http_auto_plan():

    try:

        arr = request.get_json(force=True)

        plan = []

        for step in arr:

            cmd = str(step.get("cmd","S")).upper()

            secs = float(step.get("secs", 0))

            if cmd not in ("F","B","L","R","S"): cmd = "S"

            if secs < 0: secs = 0

            plan.append(AutoStep(cmd, secs))

        with state_lock:

            state.auto_plan = plan

            state.auto_index = 0

            state.auto_started_at = None

    except Exception as e:

        return jsonify({"ok": False, "err": str(e)}), 400

    return jsonify({"ok": True, "count": len(state.auto_plan)})



@app.route("/auto/reset", methods=["POST"])

def http_auto_reset():

    with state_lock:

        state.auto_index = 0

        state.auto_started_at = None

    motors.stop()

    return "OK"



@app.route("/stream")

def stream():

    def gen():

        while True:

            with state_lock:

                data = {

                    "mode": state.mode,

                    "auto_index": state.auto_index,

                    "obstacle": state.obstacle_hit,

                    "last_bt": state.last_bt_cmd,

                    "last": motors.last()

                }

            # sensors snapshot

            data["dht_temp"]  = round(dht.temp_c,1) if dht.temp_c is not None else None

            data["dht_hum"]   = round(dht.hum,1) if dht.hum is not None else None

            data["us_front"]  = round(ultra.distances["front"],1) if ultra.distances["front"] else None

            data["us_left"]   = round(ultra.distances["left"],1)  if ultra.distances["left"]  else None

            data["us_right"]  = round(ultra.distances["right"],1) if ultra.distances["right"] else None

            if max30 and max30.present:

                data["ir"]  = int(max30.ir) if max30.ir else None

                data["red"] = int(max30.red) if max30.red else None

            else:

                data["ir"] = data["red"] = None



            yield f"data: {json.dumps(data)}\n\n"

            time.sleep(1.0)

    return Response(gen(), mimetype="text/event-stream")



# ---------------------- Clean up --------------------------

def _cleanup(*_):

    try:

        motors.stop()

    except: pass

    try:

        ultra.stop()

    except: pass

    try:

        dht.stop()

    except: pass

    GPIO.cleanup()

    os._exit(0)



# ---------------------- Main ------------------------------

if __name__ == "__main__":

    # Threads

    th_ultra = threading.Thread(target=ultrasonic_thread, daemon=True)

    th_ultra.start()



    th_dht = threading.Thread(target=dht_thread, daemon=True)

    th_dht.start()



    th_sens = threading.Thread(target=sensors_thread, daemon=True)

    th_sens.start()



    th_safe = threading.Thread(target=safety_watchdog_thread, daemon=True)

    th_safe.start()



    th_auto = threading.Thread(target=auto_runner_thread, daemon=True)

    th_auto.start()

    
    # API Upload Thread
    if API_UPLOAD_ENABLED:
        th_api = threading.Thread(target=api_upload_thread, daemon=True)
        th_api.start()
        print("✅ API upload thread started")



    # Optional Bluetooth SPP

    if BT_SERIAL_DEV:

        try:

            import serial  # pyserial

            th_bt = threading.Thread(target=bluetooth_reader_thread, daemon=True)

            th_bt.start()

        except Exception:

            pass  # pyserial not installed or no rfcomm; web control still works



    signal.signal(signal.SIGINT, _cleanup)

    signal.signal(signal.SIGTERM, _cleanup)



    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True)

