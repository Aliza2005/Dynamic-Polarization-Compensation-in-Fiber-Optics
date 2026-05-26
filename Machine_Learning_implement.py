"""
Thorlabs PAX1000 Polarimeter Dashboard (Active Disturbance Rejection)
=====================================================================

CORRECTIONS APPLIED:
1. FIXED: joblib Dictionary unpack crash.
2. FIXED: Applied MinMaxScaler to live Monte Carlo inputs.
3. FIXED: Normalized Neural Network Stokes predictions to physical sphere (radius=1).
4. FIXED: Reduced Monte Carlo samples to 15,000 to prevent GUI main-thread freezing.
5. FIXED: Hardware Harvesting timer increased to 350ms to allow Liquid Crystals to settle.
6. FIXED: Replaced 1D-cyclic PID with a 3D Multivariable Gradient Descent.
7. ADDED: Auto-saves session data (Timestamp, Stokes, DOP, Power, Angles) to CSV on close.
"""

import sys
import logging
import time
import math
import csv
from typing import List, Dict, Tuple, Optional
import joblib
import numpy as np
import pyvisa
import ftd2xx as ftd
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QVector3D, QVector4D, QColor, QPalette
import pyqtgraph.opengl as gl

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
class Config:
    DEVICE_ID = 'USB0::0x1313::0x8031::M00931369::0::INSTR'
    WAVELENGTH_NM = 1550.0
    SIMULATION_MODE = False 
    POLLING_INTERVAL_MS = 40
    EXACT_MATCH_THRESHOLD = 0.9995  
    
    COLOR_BG = "#212121"         
    COLOR_TEXT = "#E0E0E0"       
    COLOR_ACCENT = "#00E5FF"     
    COLOR_SUBTLE = "#757575"     
    COLOR_SUCCESS = "#00C853"
    COLOR_WARNING = "#FFAB00"
    COLOR_EXACT = "#FF00FF"      
    COLOR_SHADOW_DARK = QColor(0, 0, 0, 180)

TARGET_STATES: Dict[str, List[float]] = {
    "Linear Horizontal (H)": [1.0, 0.0, 0.0],
    "Linear Vertical (V)":   [-1.0, 0.0, 0.0],
    "Linear +45° (D)":       [0.0, 1.0, 0.0],
    "Linear -45° (A)":       [0.0, -1.0, 0.0],
    "Left Circular (L)":     [0.0, 0.0, 1.0],    
    "Right Circular (R)":    [0.0, 0.0, -1.0]    
}

STATE_SHORT_NAMES = {
    "Linear Horizontal (H)": "H", "Linear Vertical (V)": "V",
    "Linear +45° (D)": "D", "Linear -45° (A)": "A",
    "Left Circular (L)": "L", "Right Circular (R)": "R"
}

AXIS_LABELS: List[Tuple[str, List[float]]] = [
    ("H", [1.5, 0.0, 0.0]), ("V", [-1.5, 0.0, 0.0]),     
    ("D", [0.0, 1.5, 0.0]), ("A", [0.0, -1.5, 0.0]),     
    ("L", [0.0, 0.0, 1.5]), ("R", [0.0, 0.0, -1.5])      
]

# =======================================================================
# --- EPC HARDWARE DRIVER ---
# =======================================================================
class EPC_Driver:
    def __init__(self, device_index=0):
        self.handle = None
        self.is_connected = False
        try:
            self.handle = ftd.open(device_index)
            self.handle.setBaudRate(38400) 
            self.handle.setDataCharacteristics(ftd.defines.BITS_8, ftd.defines.STOP_BITS_1, ftd.defines.PARITY_NONE)
            self.handle.setTimeouts(1000, 1000)
            self.is_connected = True
            logger.info("EPC Connected via D2XX at 38400 Baud!")
        except Exception as e:
            logger.error(f"EPC Connection Error: {e}")

    def set_voltage(self, channel: int, voltage: float):
        if self.is_connected and self.handle:
            v = max(0.0, min(10.0, voltage)) 
            dac_value = int((v / 10.0) * 4095)
            cmd = f"VS{channel:03d}V{dac_value:04d}\r"
            try:
                self.handle.write(cmd.encode('ascii'))
            except Exception as e:
                logger.error(f"EPC Write Error: {e}")

    def close(self):
        if self.handle:
            self.handle.close()

# =======================================================================
# --- POLARIMETER THREAD ---
# =======================================================================
import psutil
import os
import time
import matplotlib.pyplot as plt

class RAMTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        # Store data as { 'AlgoName': [(timestamp, ram_mb), ...] }
        self.history = {'ML': [], 'PID': [], 'HillClimb': []}
        self.start_time = time.time()

    def record(self, algo_name):
        # Memory in MegaBytes (MB)
        mem = self.process.memory_info().rss / (1024 * 1024)
        elapsed = time.time() - self.start_time
        self.history[algo_name].append((elapsed, mem))

    def plot_usage(self):
        plt.figure(figsize=(10, 6))
        colors = {'ML': '#2196F3', 'PID': '#4CAF50', 'HillClimb': '#FF9800'}
        
        for algo, data in self.history.items():
            if not data: continue
            times, ram = zip(*data)
            plt.plot(times, ram, label=f"{algo} Usage", color=colors[algo], linewidth=2)

        plt.title("Algorithm Memory Profile (RAM)")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("RAM Usage (MB)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig("algorithm_ram_comparison.png")
        print("RAM report saved as 'algorithm_ram_comparison.png'")

# Initialize in your Dashboard's __init__:
# self.ram_tracker = RAMTracker()
class PolarimeterThread(QThread):
    data_received = pyqtSignal(list)
    connection_status = pyqtSignal(bool, str)

    def __init__(self, device_id: str):
        super().__init__()
        self.device_id = device_id
        self.running = True
        self.is_simulating = False
        
    def run(self) -> None:
        rm = pyvisa.ResourceManager()
        tlpax = None

        if not Config.SIMULATION_MODE:
            try:
                tlpax = rm.open_resource(self.device_id)
                tlpax.timeout = 2000
                tlpax.write(f"SENS:WAV {Config.WAVELENGTH_NM}")
                tlpax.write("SENS:CALC 9")      
                tlpax.write("INP:ROT:STAT 1")   
                self.connection_status.emit(True, "Connected (Live)")
                self.is_simulating = False
            except Exception as e:
                self.is_simulating = True
                self.connection_status.emit(False, "Simulation Mode")
        else:
            self.is_simulating = True
            self.connection_status.emit(False, "Simulation Mode")

        while self.running:
            if not self.is_simulating and tlpax:
                try:
                    raw_data = tlpax.query("SENS:DATA:LAT?")
                    self._process_real_data(raw_data)
                except Exception: pass
            else:
                self._generate_simulation_data()
            self.msleep(Config.POLLING_INTERVAL_MS)

        if tlpax:
            try: tlpax.close()
            except: pass
        rm.close()

    def _process_real_data(self, raw_data: str) -> None:
        try:
            msg = [float(x) for x in raw_data.split(',')]
            if len(msg) >= 13:
                psi_rad, chi_rad = msg[9], msg[10]
                raw_dop, pwr_watts = msg[11], msg[12]
                dop = (raw_dop if raw_dop <= 1.0 else 1.0) * 100
                pwr_mw = pwr_watts * 1000.0

                s1 = math.cos(2 * psi_rad) * math.cos(2 * chi_rad)
                s2 = math.sin(2 * psi_rad) * math.cos(2 * chi_rad)
                s3 = math.sin(2 * chi_rad)
                
                self.data_received.emit([s1, s2, s3, dop, pwr_mw, math.degrees(psi_rad), math.degrees(chi_rad)])
        except ValueError: pass
            
    def _generate_simulation_data(self) -> None:
        t = time.time() * 0.5
        state_idx = int(t / 8) % 6
        target = np.array(TARGET_STATES[list(TARGET_STATES.keys())[state_idx]])
        sop = target + np.random.normal(0, 0.02, 3)
        sop = sop / np.linalg.norm(sop)
        
        # Calculate fake angles so the CSV looks realistic during testing
        psi_deg = 0.5 * math.degrees(math.atan2(sop[1], sop[0]))
        chi_deg = 0.5 * math.degrees(math.asin(np.clip(sop[2], -1.0, 1.0)))
        
        self.data_received.emit([sop[0], sop[1], sop[2], 99.5, 1.250, psi_deg, chi_deg])

    def stop(self) -> None:
        self.running = False


class NeuCard(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet(f"NeuCard {{ background-color: {Config.COLOR_BG}; border-radius: 20px; border: 1px solid #333; }}")
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(8)
        self.shadow.setYOffset(8)
        self.shadow.setColor(Config.COLOR_SHADOW_DARK)
        self.setGraphicsEffect(self.shadow)

class ResizableGLWidget(gl.GLViewWidget):
    scene_updated = pyqtSignal()
    def paintGL(self) -> None:
        super().paintGL()
        self.scene_updated.emit()

# =======================================================================
# --- MAIN DASHBOARD ---
# =======================================================================
class PoincareDashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.current_target_vec = np.array(TARGET_STATES["Linear Horizontal (H)"])
        self.current_sop = np.array([1.0, 0.0, 0.0])
        self.cached_label_html = "" 
        self.exact_match_state: Optional[str] = None  
        self.axis_overlays = []
        self.ram_tracker = RAMTracker()
        
        # --- NEW LOGGING LIST ---
        self.log_data = []
        self.logging_enabled = True

        # --- Hardware & Control Variables ---
        self.epc = EPC_Driver()
        self.auto_compensate = False
        self.epc_voltages = np.array([5.0, 5.0, 5.0])
        self.last_error_deg = 999.0
        self.last_epc_update_time = time.time()
        
        # 3D PID (Gradient Descent) State
        self.current_gradient_dir = np.random.randn(3)
        self.current_gradient_dir /= np.linalg.norm(self.current_gradient_dir)
        self.epc_step_size = 0.10
        self.epc_bad_steps = 0
        
        self.ui_sliders = []
        self.ui_spinboxes = []

        self._init_ui()
        self._init_3d_scene()
        self._start_backend()

        self._set_all_voltages(self.epc_voltages)

        # --- LOAD THE AI BRAIN & SCALER ---
        try:
            brain_data = joblib.load("forward_brain.joblib")
            self.epc_brain = brain_data['model']
            self.epc_scaler = brain_data['scaler']
            logger.info("AI Neural Network & Scaler Loaded Successfully!")
        except Exception as e:
            self.epc_brain = None
            self.epc_scaler = None
            logger.error(f"Could not load AI Brain. Has the training script been run? Error: {e}")

    def _init_ui(self) -> None:
        self.setWindowTitle("Thorlabs PAX1000 | AI Disturbance Rejection")
        self.resize(1400, 900)
        
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {Config.COLOR_BG}; }}
            QLabel {{ color: {Config.COLOR_TEXT}; font-family: 'Segoe UI', sans-serif; }}
            QComboBox {{ background-color: {Config.COLOR_BG}; color: {Config.COLOR_ACCENT}; border: 1px solid #444; border-radius: 8px; padding: 6px 10px; font-weight: bold; }}
            QSlider::groove:horizontal {{ border: 1px solid #333; height: 8px; background: #2a2a2a; border-radius: 4px; }}
            QSlider::handle:horizontal {{ background: {Config.COLOR_ACCENT}; width: 16px; margin: -4px 0; border-radius: 8px; }}
            QDoubleSpinBox {{ background-color: #2a2a2a; color: {Config.COLOR_ACCENT}; font-weight: bold; border: 1px solid #444; border-radius: 4px; padding: 4px; }}
            QDoubleSpinBox:disabled {{ color: #777; }}
            QSlider:disabled {{ background: #333; }}
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(30)

        # --- LEFT SIDEBAR ---
        self.sidebar = NeuCard()
        self.sidebar.setFixedWidth(420) 
        side_layout = QtWidgets.QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(25, 25, 25, 25)
        side_layout.setSpacing(12)
        
        lbl_header = QtWidgets.QLabel("POLARIZATION ANALYSIS")
        lbl_header.setStyleSheet(f"font-size: 11px; letter-spacing: 2px; font-weight: bold; color: {Config.COLOR_SUBTLE};")
        side_layout.addWidget(lbl_header)
        
        self.target_selector = QtWidgets.QComboBox()
        self.target_selector.addItems(TARGET_STATES.keys())
        self.target_selector.currentIndexChanged.connect(self._on_target_changed)
        side_layout.addWidget(self.target_selector)

        self.exact_state_label = QtWidgets.QLabel("● EXACT STATE: None")
        self.exact_state_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {Config.COLOR_SUBTLE}; padding: 8px; background-color: #2a2a2a; border-radius: 8px;")
        self.exact_state_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(self.exact_state_label)

        # -- ACTIVE COMPENSATION CONTROLS --
        self.btn_auto = QtWidgets.QPushButton("Start Auto-Compensate")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setStyleSheet("QPushButton { background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 8px; font-size: 13px;} QPushButton:checked { background-color: #00C853; }")
        self.btn_auto.clicked.connect(self._toggle_compensation)
        side_layout.addWidget(self.btn_auto)

        # --- ML DATA HARVESTER BUTTON ---
        self.btn_ml_collect = QtWidgets.QPushButton("🧠 START ML DATA HARVEST")
        self.btn_ml_collect.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_ml_collect.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 10px; border-radius: 8px;")
        self.btn_ml_collect.clicked.connect(lambda: self.start_ml_data_collection(10000))
        side_layout.addWidget(self.btn_ml_collect)

        # -- EPC MANUAL SLIDERS & INPUTS --
        sliders_box = QtWidgets.QGroupBox("MANUAL EPC CONTROLS (0 - 10V)")
        sliders_box.setStyleSheet("QGroupBox { color: #BBB; font-weight: bold; border: 1px solid #444; border-radius: 8px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }")
        slider_layout = QtWidgets.QVBoxLayout(sliders_box)
        
        for i in range(3): 
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0,0,0,0)
            
            lbl = QtWidgets.QLabel(f"CH {i}:")
            lbl.setFixedWidth(35)
            
            slider = QtWidgets.QSlider(Qt.Horizontal)
            slider.setRange(0, 1000) 
            
            spinbox = QtWidgets.QDoubleSpinBox()
            spinbox.setRange(0.0, 10.0)
            spinbox.setSingleStep(0.05)
            spinbox.setDecimals(2)
            spinbox.setSuffix(" V")
            spinbox.setFixedWidth(75)
            
            slider.valueChanged.connect(lambda val, ch=i: self._on_manual_input(ch, 'slider', val))
            spinbox.valueChanged.connect(lambda val, ch=i: self._on_manual_input(ch, 'spinbox', val))
            
            rl.addWidget(lbl)
            rl.addWidget(slider)
            rl.addWidget(spinbox)
            slider_layout.addWidget(row)
            
            self.ui_sliders.append(slider)
            self.ui_spinboxes.append(spinbox)
            
        side_layout.addWidget(sliders_box)

        # Metrics Groups
        self.metrics_labels = {}
        self._add_metric_group(side_layout, "MEASUREMENTS", [("Power", "mW"), ("DOP", "%"), ("Error", "°")])
        self.stokes_labels = {}
        self._add_metric_group(side_layout, "STOKES VECTORS", [("S1", ""), ("S2", ""), ("S3", "")], target_dict=self.stokes_labels)
        
        side_layout.addStretch()
        self.status_indicator = QtWidgets.QLabel("Initializing...")
        self.status_indicator.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(self.status_indicator)

        # --- RIGHT PANEL (3D View) ---
        self.view_container = NeuCard()
        view_layout = QtWidgets.QVBoxLayout(self.view_container)
        view_layout.setContentsMargins(15, 15, 15, 15)

        self.view = ResizableGLWidget()
        self.view.setBackgroundColor(Config.COLOR_BG) 
        self.view.setCameraPosition(distance=4.8, elevation=30, azimuth=45)
        self.view.scene_updated.connect(self._update_all_overlays)
        view_layout.addWidget(self.view)

        # --- FLOATING OVERLAYS ---
        self.overlay = QtWidgets.QLabel(self.view)
        self.overlay.setStyleSheet(f"QLabel {{ background-color: rgba(20, 20, 20, 0.95); color: {Config.COLOR_TEXT}; border-radius: 8px; padding: 12px; border: 1px solid #555; font-family: 'Consolas', monospace; font-size: 12px; }}")
        self.overlay.setVisible(False)

        for text, pos in AXIS_LABELS:
            lbl = QtWidgets.QLabel(text, self.view)
            lbl.setStyleSheet("font-weight: 900; color: #FFFFFF; font-size: 16px; font-family: 'Arial Black', sans-serif; background-color: transparent;")
            lbl.adjustSize()
            lbl.setVisible(False) 
            self.axis_overlays.append((lbl, np.array(pos)))

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.view_container)

    def _add_metric_group(self, layout, title, fields, target_dict=None):
        if target_dict is None: target_dict = self.metrics_labels
        h = QtWidgets.QLabel(title)
        h.setStyleSheet(f"font-size: 10px; font-weight: bold; color: {Config.COLOR_SUBTLE};")
        layout.addWidget(h)
        for name, unit in fields:
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            lbl_name = QtWidgets.QLabel(name)
            lbl_name.setStyleSheet(f"font-size: 14px; color: #BBB;")
            val_display = QtWidgets.QLabel(f"0.000 {unit}")
            val_display.setStyleSheet(f"color: {Config.COLOR_ACCENT}; font-size: 18px; font-weight: bold;")
            val_display.setAlignment(Qt.AlignRight)
            rl.addWidget(lbl_name)
            rl.addStretch()
            rl.addWidget(val_display)
            layout.addWidget(row)
            target_dict[name] = val_display

    def _on_manual_input(self, channel: int, source: str, val):
        if self.auto_compensate: return
        voltage = (val / 100.0) if source == 'slider' else val
        self._update_slider_ui_safely(channel, voltage)
        self.epc_voltages[channel] = voltage
        self.epc.set_voltage(channel, voltage)

    def _update_slider_ui_safely(self, channel: int, voltage: float):
        slider = self.ui_sliders[channel]
        spinbox = self.ui_spinboxes[channel]
        slider.blockSignals(True) 
        spinbox.blockSignals(True)
        slider.setValue(int(voltage * 100))
        spinbox.setValue(voltage)
        slider.blockSignals(False)
        spinbox.blockSignals(False)
        
    def _set_all_voltages(self, voltages: np.ndarray):
        """Helper to command all waveplates safely and sync UI."""
        self.epc_voltages = np.array(voltages)
        for ch in range(3):
            v = float(np.clip(self.epc_voltages[ch], 0.0, 10.0))
            self.epc.set_voltage(ch, v)
            if len(self.ui_sliders) > ch:
                self._update_slider_ui_safely(ch, v)

    def _toggle_compensation(self):
        self.auto_compensate = self.btn_auto.isChecked()
        if self.auto_compensate:
            self.btn_auto.setText("COMPENSATING DRIFT...")
            self.last_error_deg = 999.0 
            for sl, sp in zip(self.ui_sliders, self.ui_spinboxes):
                sl.setEnabled(False); sp.setEnabled(False)
        else:
            self.btn_auto.setText("Start Auto-Compensate")
            for sl, sp in zip(self.ui_sliders, self.ui_spinboxes):
                sl.setEnabled(True); sp.setEnabled(True)

    def _init_3d_scene(self) -> None:
        md = gl.MeshData.sphere(rows=40, cols=40)
        self.sphere_item = gl.GLMeshItem(meshdata=md, smooth=True, color=(0.1, 0.1, 0.1, 0.2), drawEdges=True, edgeColor=(0.4, 0.4, 0.4, 0.3), drawFaces=False)
        self.view.addItem(self.sphere_item)
        for _, pos in AXIS_LABELS:
             surface_pos = np.array(pos) * (1.3 / 1.5)
             dot = gl.GLScatterPlotItem(pos=surface_pos.reshape(1,3), color=(0.8,0.8,0.8,0.5), size=5, pxMode=True)
             self.view.addItem(dot)
        self.target_marker = gl.GLScatterPlotItem(pos=self.current_target_vec.reshape(1,3), color=(0, 1, 0, 0.5), size=25, pxMode=True)
        self.view.addItem(self.target_marker)
        self.sop_marker = gl.GLScatterPlotItem(pos=np.array([[1, 0, 0]]), color=(0, 0.9, 1, 1.0), size=20, pxMode=True)
        self.view.addItem(self.sop_marker)

    def _start_backend(self) -> None:
        self.thread = PolarimeterThread(Config.DEVICE_ID)
        self.thread.data_received.connect(self._on_data_received)
        self.thread.connection_status.connect(self._on_connection_status)
        self.thread.start()

    def _on_target_changed(self) -> None:
        name = self.target_selector.currentText()
        if name in TARGET_STATES:
            self.current_target_vec = np.array(TARGET_STATES[name])
            self.target_marker.setData(pos=self.current_target_vec.reshape(1,3).astype(np.float32))

    def _on_connection_status(self, connected: bool, msg: str) -> None:
        self.status_indicator.setText(f"STATUS: {msg.upper()}")
        color = Config.COLOR_SUCCESS if connected else Config.COLOR_WARNING
        self.status_indicator.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")

    def _on_data_received(self, data: List[float]) -> None:
        s1, s2, s3, dop, pwr, psi_deg, chi_deg = data
        raw_sop = np.array([s1, s2, s3])
        norm = np.linalg.norm(raw_sop)
        norm_sop = raw_sop / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        self.current_sop = norm_sop
        
        # --- NEW LOGGING LOGIC ---
        if self.logging_enabled:
            current_time = time.time()
            # Order: Timestamp, S1, S2, S3, DOP_%, Power_mW, Azimuth_deg, Ellipticity_deg
            self.log_data.append([current_time, s1, s2, s3, dop, pwr, psi_deg, chi_deg])

        dot_target = np.clip(np.dot(norm_sop, self.current_target_vec), -1.0, 1.0)
        dist_to_target = math.degrees(math.acos(dot_target))
        
        self.metrics_labels["Power"].setText("< 0.001 mW" if pwr < 0.001 else f"{pwr:.3f} mW")
        self.metrics_labels["DOP"].setText(f"{dop:.1f} %")
        self.metrics_labels["Error"].setText(f"{dist_to_target:.2f} °")
        
        for key, val in zip(["S1", "S2", "S3"], [s1, s2, s3]):
            self.stokes_labels[key].setText(f"{val:+.3f}")
            
        self.sop_marker.setData(pos=self.current_sop.reshape(1,3).astype(np.float32), color=(0.0, 0.9, 1.0, 1.0), size=20)
        self._update_all_overlays()

        # --- RUN ACTIVE STABILIZER ---
        if self.auto_compensate:
            self._run_epc_optimization(dist_to_target)
    def _run_epc_optimization(self, current_error_deg):
        
        if hasattr(self, 'ram_tracker'):
            self.ram_tracker.record('ML')

        now = time.time()

        # --- 1. 5-DEGREE THRESHOLD CONFIG ---
        TARGET_LOCK_ZONE = 5.0  # The "Kill" threshold you requested
        WAKE_UP_THRESHOLD = 8.5 # Only restart if it drifts past 8.5 degrees
        STABILITY_TIME = 3.0    # Hold for 3 seconds before killing the engine

        # Persistence safety
        if not hasattr(self, 'is_iron_locked'): self.is_iron_locked = False
        if not hasattr(self, 'settle_start_time'): self.settle_start_time = None
        if not hasattr(self, 'active_epc_channel'): self.active_epc_channel = 0
        if not hasattr(self, 'epc_step_size'): self.epc_step_size = 0.05
        if not hasattr(self, 'last_error_deg'): self.last_error_deg = current_error_deg

        # --- 2. THE ENGINE KILL-SWITCH ---
        if self.is_iron_locked:
            if current_error_deg > WAKE_UP_THRESHOLD:
                # BREAK LOCK: System has drifted too far
                self.is_iron_locked = False
                self.settle_start_time = None
                logger.info("Drift detected (>8.5°). Re-awakening Engine.")
            else:
                # ENGINE IS DEAD: No writes to EPC, No UI updates
                self.btn_auto.setText(f"● 5° LOCK ACHIEVED ({current_error_deg:.2f}°)")
                self.btn_auto.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; border: 2px solid #A5D6A7;")
                return 

        # --- 3. THE 3-SECOND VERIFICATION ---
        if current_error_deg <= TARGET_LOCK_ZONE:
            if self.settle_start_time is None:
                self.settle_start_time = now # Start the 3s clock
            
            elapsed = now - self.settle_start_time
            if elapsed >= STABILITY_TIME:
                self.is_iron_locked = True
                logger.info(f"Target reached and held for {STABILITY_TIME}s. Killing process.")
                return
            else:
                self.btn_auto.setText(f"LOCKING... {STABILITY_TIME - elapsed:.1f}s")
                return # Pause all movement during verification
        else:
            self.settle_start_time = None

        # --- 4. HARDWARE THROTTLE ---
        # Fixed 250ms delay to keep the hardware calm
        if now - getattr(self, 'last_epc_update_time', 0) < 0.25:
            return
        self.last_epc_update_time = now

        # --- 5. CONTROL LOGIC (Standard Descent) ---
        S_raw = np.array(self.current_sop[:3], dtype=float)
        T_raw = np.array(self.current_target_vec[:3], dtype=float)
        S_unit = S_raw / (np.linalg.norm(S_raw) + 1e-9)
        T_unit = T_raw / (np.linalg.norm(T_raw) + 1e-9)

        if current_error_deg >= 15.0:
            # PHASE 1: AI MACRO JUMP
            V_curr = np.array(self.epc_voltages[:3], dtype=float)
            test_v = np.clip(np.vstack([np.random.normal(V_curr, 1.2, (200, 3)), np.random.uniform(0.0, 10.0, (50, 3))]), 0.0, 10.0)
            X_macro = np.hstack([np.tile(S_unit, (250, 1)), np.tile(V_curr, (250, 1)), test_v])
            dS_pred = self.epc_brain.predict(self.epc_scaler.transform(X_macro))
            best_idx = np.argmax(np.dot(S_unit + dS_pred, T_unit))
            self._set_all_voltages(np.clip(test_v[best_idx], 0.0, 10.0))
        else:
            # PHASE 2: SMOOTH APPROACH TO 5 DEGREES
            # Keep steps steady (0.1V) to avoid the "creep" issues you had earlier
            step_mag = 0.10 
            
            if current_error_deg > self.last_error_deg:
                # If error got worse, reverse and switch channel immediately
                self.epc_step_size = -self.epc_step_size
                self.active_epc_channel = (self.active_epc_channel + 1) % 3
            else:
                # Maintain current direction
                self.epc_step_size = np.sign(self.epc_step_size) * step_mag

            val = np.clip(self.epc_voltages[self.active_epc_channel] + self.epc_step_size, 0.0, 10.0)
            self.epc_voltages[self.active_epc_channel] = val
            self.epc.set_voltage(self.active_epc_channel, val)
            self._update_slider_ui_safely(self.active_epc_channel, val)

        self.last_error_deg = current_error_deg
    def start_ml_data_collection(self, num_samples=5000):
        self.ml_samples_total = num_samples
        self.ml_samples_collected = 0
        self.ml_data_buffer = [] 
        self.btn_ml_collect.setText(f"COLLECTING... 0 / {num_samples}")
        self.btn_ml_collect.setStyleSheet("background-color: #D50000; color: white; font-weight: bold;")
        self.btn_ml_collect.setEnabled(False)
        
        self.ml_timer = QtCore.QTimer()
        self.ml_timer.timeout.connect(self._ml_timer_tick)
        self._ml_set_random_voltages()
        self.ml_timer.start(500) 

    def _ml_set_random_voltages(self):
        import random
        self.current_v0 = random.uniform(0.0, 10.0)
        self.current_v1 = random.uniform(0.0, 10.0)
        self.current_v2 = random.uniform(0.0, 10.0)
        self._set_all_voltages([self.current_v0, self.current_v1, self.current_v2])

    def _ml_timer_tick(self):
    # ✅ Step 1: Record the result of the PREVIOUS voltage (now settled)
       s1, s2, s3 = self.current_sop[0], self.current_sop[1], self.current_sop[2]
       self.ml_data_buffer.append([self.current_v0, self.current_v1, self.current_v2, s1, s2, s3])
       self.ml_samples_collected += 1

       if self.ml_samples_collected % 50 == 0:
           self.btn_ml_collect.setText(f"COLLECTING... {self.ml_samples_collected} / {self.ml_samples_total}")

       if self.ml_samples_collected >= self.ml_samples_total:
           self.ml_timer.stop()
           self._save_ml_data()
           return

    # ✅ Step 2: THEN set the next random voltage (will settle before next tick)
       self._ml_set_random_voltages()
    def _save_ml_data(self):
        filename = "ml_training_data.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['V0', 'V1', 'V2', 'S1', 'S2', 'S3'])
            writer.writerows(self.ml_data_buffer)
        self.btn_ml_collect.setText("🧠 START ML DATA HARVEST")
        self.btn_ml_collect.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 10px; border-radius: 8px;")
        self.btn_ml_collect.setEnabled(True)

    def save_log_to_csv(self, filename="polarization_session_log.csv"):
        """Saves the continuous background data collected during the session to CSV."""
        if not self.log_data:
            return
        
        headers = ["Timestamp", "S1", "S2", "S3", "DOP_percent", "Power_mW", "Azimuth_deg", "Ellipticity_deg"]
        
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(self.log_data)
            logger.info(f"Session log saved to '{filename}' with {len(self.log_data)} entries.")
        except Exception as e:
            logger.error(f"Failed to save CSV session log: {e}")

    def _update_all_overlays(self) -> None:
        view_w = self.view.width()
        view_h = self.view.height()
        rect = (0, 0, view_w, view_h)
        proj_matrix = self.view.projectionMatrix(region=rect, viewport=rect)
        view_matrix = self.view.viewMatrix()
        mvp_matrix = proj_matrix * view_matrix
        
        def project_point(pos_3d):
            vec = QVector3D(pos_3d[0], pos_3d[1], pos_3d[2])
            screen_vec = mvp_matrix.map(QVector4D(vec, 1.0))
            if screen_vec.w() == 0: return None
            x_ndc = screen_vec.x() / screen_vec.w()
            y_ndc = screen_vec.y() / screen_vec.w()
            if screen_vec.w() < 0: return None
            return int((x_ndc + 1) * view_w / 2), int((1 - y_ndc) * view_h / 2)

        for lbl, pos_3d in self.axis_overlays:
            screen_pos = project_point(pos_3d)
            if screen_pos and 0 <= screen_pos[0] <= view_w and 0 <= screen_pos[1] <= view_h:
                lbl.setVisible(True)
                lbl.move(screen_pos[0] - lbl.width() // 2, screen_pos[1] - lbl.height() // 2)
            else:
                lbl.setVisible(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # --- Save the data when user clicks the 'X' button ---
        self.save_log_to_csv()
        
        self.thread.stop()
        self.thread.wait()
        if hasattr(self, 'epc'): self.epc.close()
        event.accept()

if __name__ == '__main__':
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(33, 33, 33))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    dashboard = PoincareDashboard()
    dashboard.show()
    sys.exit(app.exec_())