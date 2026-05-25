"""
Thorlabs PAX1000 Polarimeter Dashboard (Active Disturbance Rejection)
=====================================================================

- Full 10.0V EPC Range (Mapped to 12-bit 0-4095 DAC)
- 3 Waveplate Hardware Integration
- Real-time EPC Voltage Sliders & Manual Number Inputs
- Active Auto-Compensation Loop for Stress/Drift

Author: Coding Partner
"""

import sys
import logging
import time
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import pyvisa
import ftd2xx as ftd
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QVector3D, QVector4D, QColor, QFont, QPalette
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
# --- EPC HARDWARE DRIVER (10V / 12-BIT DAC) ---
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
        s1, s2, s3 = sop
        psi_deg = 0.5 * math.degrees(math.atan2(s2, s1))
        chi_deg = 0.5 * math.degrees(math.asin(s3))
        self.data_received.emit([s1, s2, s3, 99.5, 1.250, psi_deg, chi_deg])

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

class PoincareDashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.current_target_vec = np.array(TARGET_STATES["Linear Horizontal (H)"])
        self.current_sop = np.array([1.0, 0.0, 0.0])
        self.cached_label_html = "" 
        self.exact_match_state: Optional[str] = None  
        self.axis_overlays: List[Tuple[QtWidgets.QLabel, np.ndarray]] = []
        
        self.reference_sop = None
        self.rotation_axis = None
        self.retardance_deg = 0.0
        self.axis_history = []
        self.sop_history = []
        self.log_data = []
        self.logging_enabled = True

        # --- 3-Waveplate EPC Variables ---
        self.epc = EPC_Driver()
        self.auto_compensate = False
        self.epc_voltages = [5.0, 5.0, 5.0] 
        self.active_epc_channel = 0
        self.epc_step_size = 0.20 
        self.last_error_deg = 999.0
        self.last_epc_update_time = time.time()
        self.epc_bad_steps = 0
        
        self.ui_sliders = []
        self.ui_spinboxes = [] # NEW: Precise text input boxes

        self._init_ui()
        self._init_3d_scene()
        self._start_backend()

        # Initialize hardware strictly to 5V upon startup
        for ch in range(3):
            self.epc.set_voltage(ch, self.epc_voltages[ch])
            if hasattr(self, 'ui_sliders') and len(self.ui_sliders) > ch:
                self._update_slider_ui_safely(ch, self.epc_voltages[ch])

    def _init_ui(self) -> None:
        self.setWindowTitle("Thorlabs PAX1000 | Active EPC Stabilizer")
        self.resize(1400, 900)
        
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {Config.COLOR_BG}; }}
            QLabel {{ color: {Config.COLOR_TEXT}; font-family: 'Segoe UI', sans-serif; }}
            QComboBox {{
                background-color: {Config.COLOR_BG}; color: {Config.COLOR_ACCENT};
                border: 1px solid #444; border-radius: 8px; padding: 6px 10px; font-weight: bold;
            }}
            QSlider::groove:horizontal {{ border: 1px solid #333; height: 8px; background: #2a2a2a; border-radius: 4px; }}
            QSlider::handle:horizontal {{ background: {Config.COLOR_ACCENT}; width: 16px; margin: -4px 0; border-radius: 8px; }}
            QDoubleSpinBox {{
                background-color: #2a2a2a; color: {Config.COLOR_ACCENT}; font-weight: bold;
                border: 1px solid #444; border-radius: 4px; padding: 4px;
            }}
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
        self.sidebar.setFixedWidth(420) # Slightly wider for spinboxes
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
            
            # The dragging slider
            slider = QtWidgets.QSlider(Qt.Horizontal)
            slider.setRange(0, 1000) # 0 to 1000 representing 0.00V to 10.00V
            
            # The precise number input box
            spinbox = QtWidgets.QDoubleSpinBox()
            spinbox.setRange(0.0, 10.0)
            spinbox.setSingleStep(0.05)
            spinbox.setDecimals(2)
            spinbox.setSuffix(" V")
            spinbox.setFixedWidth(75)
            
            # Connect both to a unified manual control function
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
        
        # --- Rotation Metrics ---
        self.rot_axis_label = QtWidgets.QLabel("Axis: [0.00, 0.00, 0.00]")
        self.rot_axis_label.setStyleSheet("color: #BBBBBB; font-size: 13px;")
        side_layout.addWidget(self.rot_axis_label)

        self.retardance_label = QtWidgets.QLabel("Retardance: 0.00 °")
        self.retardance_label.setStyleSheet("color: #FFAB00; font-size: 15px; font-weight: bold;")
        side_layout.addWidget(self.retardance_label)

        self.btn_reset_ref = QtWidgets.QPushButton("Zero / Reset 0° Reference")
        self.btn_reset_ref.setStyleSheet("QPushButton { background-color: #4a4a4a; color: white; font-weight: bold; padding: 6px; border-radius: 4px; } QPushButton:hover { background-color: #5a5a5a; }")
        self.btn_reset_ref.clicked.connect(self.reset_reference_state)
        side_layout.addWidget(self.btn_reset_ref)

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
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet("background-color: #444; height: 1px; border: none;")
        layout.addWidget(line)

    def _on_manual_input(self, channel: int, source: str, val):
        """Handles user input from either the visual slider or the number typing box."""
        if self.auto_compensate:
            return # Ignore manual tweaks while the algorithm is running!

        # Normalize the input to an exact Float voltage (0.00 to 10.00)
        voltage = (val / 100.0) if source == 'slider' else val
        
        # Update the UI elements so they stay synced together
        self._update_slider_ui_safely(channel, voltage)
        
        # Send physical command to the EPC hardware!
        self.epc_voltages[channel] = voltage
        self.epc.set_voltage(channel, voltage)

    def _update_slider_ui_safely(self, channel: int, voltage: float):
        """Updates UI without triggering recursive signal loops."""
        slider = self.ui_sliders[channel]
        spinbox = self.ui_spinboxes[channel]
        
        # Block signals so setting values doesn't fire _on_manual_input again
        slider.blockSignals(True) 
        spinbox.blockSignals(True)
        
        slider.setValue(int(voltage * 100))
        spinbox.setValue(voltage)
        
        slider.blockSignals(False)
        spinbox.blockSignals(False)

    def _toggle_compensation(self):
        self.auto_compensate = self.btn_auto.isChecked()
        if self.auto_compensate:
            self.btn_auto.setText("COMPENSATING DRIFT...")
            self.last_error_deg = 999.0 
            # Lock the manual controls visually
            for sl, sp in zip(self.ui_sliders, self.ui_spinboxes):
                sl.setEnabled(False)
                sp.setEnabled(False)
        else:
            self.btn_auto.setText("Start Auto-Compensate")
            # Unlock the manual controls
            for sl, sp in zip(self.ui_sliders, self.ui_spinboxes):
                sl.setEnabled(True)
                sp.setEnabled(True)

    def _init_3d_scene(self) -> None:
        md = gl.MeshData.sphere(rows=40, cols=40)
        self.sphere_item = gl.GLMeshItem(meshdata=md, smooth=True, color=(0.1, 0.1, 0.1, 0.2), drawEdges=True, edgeColor=(0.4, 0.4, 0.4, 0.3), drawFaces=False)
        self.view.addItem(self.sphere_item)
        self.view.addItem(gl.GLLinePlotItem(pos=np.array([[-1.3,0,0],[1.3,0,0]]), color=(1, 0.3, 0.3, 0.8), width=2))
        self.view.addItem(gl.GLLinePlotItem(pos=np.array([[0,-1.3,0],[0,1.3,0]]), color=(0.3, 1, 0.3, 0.8), width=2))
        self.view.addItem(gl.GLLinePlotItem(pos=np.array([[0,0,-1.3],[0,0,1.3]]), color=(0.3, 0.3, 1, 0.8), width=2))
        for _, pos in AXIS_LABELS:
             surface_pos = np.array(pos) * (1.3 / 1.5)
             dot = gl.GLScatterPlotItem(pos=surface_pos.reshape(1,3), color=(0.8,0.8,0.8,0.5), size=5, pxMode=True)
             self.view.addItem(dot)
        self._build_grid()
        self.target_marker = gl.GLScatterPlotItem(pos=self.current_target_vec.reshape(1,3), color=(0, 1, 0, 0.5), size=25, pxMode=True)
        self.view.addItem(self.target_marker)
        self.sop_marker = gl.GLScatterPlotItem(pos=np.array([[1, 0, 0]]), color=(0, 0.9, 1, 1.0), size=20, pxMode=True)
        self.view.addItem(self.sop_marker)

    def _build_grid(self) -> None:
        grid_color = (0.5, 0.5, 0.5, 0.2)
        for lat in [-60, -30, 0, 30, 60]:
            lat_rad = math.radians(lat)
            theta = np.linspace(0, 2*np.pi, 60)
            pts = np.vstack([np.cos(lat_rad)*np.cos(theta), np.cos(lat_rad)*np.sin(theta), np.full_like(theta, np.sin(lat_rad))]).T
            self.view.addItem(gl.GLLinePlotItem(pos=pts, color=grid_color, width=1))
        for lon in [0, 45, 90, 135]:
            lon_rad = math.radians(lon * 2) 
            phi = np.linspace(0, 2*np.pi, 60)
            xc, zc = np.sin(phi), np.cos(phi)
            pts = np.vstack([xc*math.cos(lon_rad), xc*math.sin(lon_rad), zc]).T
            self.view.addItem(gl.GLLinePlotItem(pos=pts, color=grid_color, width=1))

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

    def _find_exact_state(self, sop: np.ndarray) -> Optional[str]:
        norm = np.linalg.norm(sop)
        if norm == 0: return None
        for name, vec in TARGET_STATES.items():
            if np.dot(sop, np.array(vec)) >= Config.EXACT_MATCH_THRESHOLD:
                return name
        return None
    
    def _on_data_received(self, data: List[float]) -> None:
        s1, s2, s3, dop, pwr, psi_deg, chi_deg = data
        raw_sop = np.array([s1, s2, s3])
        norm = np.linalg.norm(raw_sop)
        norm_sop = raw_sop / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        
        exact_state = self._find_exact_state(raw_sop)
        self.exact_match_state = exact_state
        
        if exact_state:
            self.current_sop = np.array(TARGET_STATES[exact_state])
            display_sop = self.current_sop.copy()
            is_exact = True
            short_name = STATE_SHORT_NAMES[exact_state]
            self.exact_state_label.setText(f"● EXACT STATE: {short_name}")
            self.exact_state_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {Config.COLOR_EXACT}; padding: 8px; background-color: #3a1a3a; border: 2px solid {Config.COLOR_EXACT}; border-radius: 8px;")
        else:
            self.current_sop = raw_sop
            display_sop = raw_sop
            is_exact = False
            self.exact_state_label.setText("● EXACT STATE: None")
            self.exact_state_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {Config.COLOR_SUBTLE}; padding: 8px; background-color: #2a2a2a; border-radius: 8px;")
            
        dot_target = np.clip(np.dot(norm_sop, self.current_target_vec), -1.0, 1.0)
        dist_to_target = math.degrees(math.acos(dot_target))
        
        if pwr < 0.001: self.metrics_labels["Power"].setText("< 0.001 mW")
        else: self.metrics_labels["Power"].setText(f"{pwr:.3f} mW")
            
        self.metrics_labels["DOP"].setText(f"{dop:.1f} %")
        self.metrics_labels["Error"].setText(f"{dist_to_target:.2f} °")
        
        if is_exact:
            self.stokes_labels["S1"].setText(f"{s1:+.3f} ✓")
            self.stokes_labels["S2"].setText(f"{s2:+.3f} ✓")
            self.stokes_labels["S3"].setText(f"{s3:+.3f} ✓")
            for lbl in self.stokes_labels.values(): lbl.setStyleSheet(f"color: {Config.COLOR_EXACT}; font-size: 18px; font-weight: bold;")
        else:
            self.stokes_labels["S1"].setText(f"{s1:+.3f}")
            self.stokes_labels["S2"].setText(f"{s2:+.3f}")
            self.stokes_labels["S3"].setText(f"{s3:+.3f}")
            for lbl in self.stokes_labels.values(): lbl.setStyleSheet(f"color: {Config.COLOR_ACCENT}; font-size: 18px; font-weight: bold;")

        best_match_name = "Undefined"
        best_match_val = -1.0
        for name, vec in TARGET_STATES.items():
            similarity = np.dot(norm_sop, np.array(vec))
            if similarity > best_match_val:
                best_match_val = similarity
                best_match_name = name

        display_name = best_match_name.split('(')[0].strip()
        if is_exact:
            state_desc = f"<b style='color:{Config.COLOR_EXACT}'>EXACT: {display_name}</b>"
            marker_color, marker_size = (1.0, 0.0, 1.0, 1.0), 30
        elif best_match_val > 0.95:
            state_desc = f"<b style='color:{Config.COLOR_SUCCESS}'>Match: {display_name}</b>"
            marker_color, marker_size = (0.0, 0.9, 1.0, 1.0), 20
        elif best_match_val > 0.85:
             state_desc = f"<b style='color:{Config.COLOR_ACCENT}'>Near: {display_name}</b>"
             marker_color, marker_size = (0.0, 0.9, 1.0, 1.0), 20
        else:
            state_desc = "<span style='color:#777'>Transitioning...</span>"
            marker_color, marker_size = (0.0, 0.9, 1.0, 1.0), 20

        self.sop_marker.setData(pos=self.current_sop.reshape(1,3).astype(np.float32), color=marker_color, size=marker_size)

        self.cached_label_html = (
            f"<div style='line-height:140%'>"
            f"<span style='color:#757575; font-weight:bold;'>CURRENT POLARIZATION</span><br>"
            f"{state_desc}<br><hr style='border:1px solid #444'>"
            f"<b style='color:#BBB'>S:</b> [{display_sop[0]:.2f}, {display_sop[1]:.2f}, {display_sop[2]:.2f}]<br>"
            f"<b style='color:#BBB'>2&psi;:</b> {2*psi_deg:.1f}&deg; &nbsp; <b style='color:#BBB'>2&chi;:</b> {2*chi_deg:.1f}&deg;"
            f"</div>"
        )

        if getattr(self, 'reference_sop', None) is None: self.reference_sop = norm_sop
        dot_ref = np.clip(np.dot(norm_sop, self.reference_sop), -1.0, 1.0)
        self.distance_from_ref_deg = math.degrees(math.acos(dot_ref))
        
        ref_disp = self.reference_sop
        self.rot_axis_label.setText(f"Ref State: [{ref_disp[0]:.2f}, {ref_disp[1]:.2f}, {ref_disp[2]:.2f}]")
        self.retardance_label.setText(f"Moved From Ref: {self.distance_from_ref_deg:.2f} °")

        if self.logging_enabled:
            self.log_data.append([time.time(), pwr, s1, s2, s3, dop, pwr, psi_deg, chi_deg, self.distance_from_ref_deg, dist_to_target])

        self._update_all_overlays()

        # --- RUN ACTIVE STABILIZER ---
        if self.auto_compensate:
            self._run_epc_optimization(dist_to_target)

    def _run_epc_optimization(self, current_error_deg):
        """Active 3-Waveplate Disturbance Rejection Loop."""
        now = time.time()
        if now - self.last_epc_update_time < 0.2:
            return
            
        self.last_epc_update_time = now

        if current_error_deg < 2.0:
            self.btn_auto.setText("LOCKED ON TARGET")
            self.btn_auto.setStyleSheet("background-color: #00E5FF; color: black; font-weight: bold; padding: 12px; border-radius: 8px;")
            self.epc_step_size = 0.05 
            return
        else:
            self.btn_auto.setText("FIGHTING DRIFT/STRESS...")
            self.btn_auto.setStyleSheet("background-color: #00C853; color: white; font-weight: bold; padding: 12px; border-radius: 8px;")

        base_step = 0.20 if current_error_deg > 20.0 else (0.10 if current_error_deg > 10.0 else 0.05)

        if current_error_deg > self.last_error_deg:
            self.epc_step_size *= -1.0 
            self.epc_bad_steps += 1
            if self.epc_bad_steps >= 2:
                self.active_epc_channel = (self.active_epc_channel + 1) % 3 
                self.epc_bad_steps = 0
                self.epc_step_size = base_step 
        else:
            self.epc_bad_steps = 0
            self.epc_step_size = base_step if self.epc_step_size > 0 else -base_step

        self.epc_voltages[self.active_epc_channel] += self.epc_step_size
        
        if self.epc_voltages[self.active_epc_channel] > 10.0:
            self.epc_voltages[self.active_epc_channel] = 10.0
            self.epc_step_size *= -1.0 
            self.active_epc_channel = (self.active_epc_channel + 1) % 3 
        elif self.epc_voltages[self.active_epc_channel] < 0.0:
            self.epc_voltages[self.active_epc_channel] = 0.0
            self.epc_step_size *= -1.0 
            self.active_epc_channel = (self.active_epc_channel + 1) % 3 

        self.epc.set_voltage(self.active_epc_channel, self.epc_voltages[self.active_epc_channel])
        self._update_slider_ui_safely(self.active_epc_channel, self.epc_voltages[self.active_epc_channel])
        
        self.last_error_deg = current_error_deg

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
            x_pixel = int((x_ndc + 1) * view_w / 2)
            y_pixel = int((1 - y_ndc) * view_h / 2)
            return x_pixel, y_pixel

        if self.cached_label_html:
            self.overlay.setText(self.cached_label_html)
            self.overlay.adjustSize()
            screen_pos = project_point(self.current_sop)
            if screen_pos:
                if not self.overlay.isVisible(): self.overlay.setVisible(True)
                self.overlay.move(screen_pos[0] + 30, screen_pos[1] - 30)
            else:
                self.overlay.setVisible(False)

        for lbl, pos_3d in self.axis_overlays:
            screen_pos = project_point(pos_3d)
            if screen_pos:
                if 0 <= screen_pos[0] <= view_w and 0 <= screen_pos[1] <= view_h:
                    if not lbl.isVisible(): lbl.setVisible(True)
                    lbl.move(screen_pos[0] - lbl.width() // 2, screen_pos[1] - lbl.height() // 2)
                else:
                    lbl.setVisible(False)
            else:
                lbl.setVisible(False)

    def save_csv(self, filename="polarization_log.csv"):
        import csv
        headers = ['Timestamp', 'S0', 'S1', 'S2', 'S3', 'DOP_percent', 'Power_mW', 'Azimuth_deg', 'Ellipticity_deg', 'Dist_from_Ref_deg', 'Dist_to_Target_deg']
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  
            writer.writerows(self.log_data)
        print(f"Data saved to {filename}")
    
    def reset_reference_state(self):
        if not hasattr(self, 'current_sop'): return
        raw_sop = np.array([self.current_sop[0], self.current_sop[1], self.current_sop[2]])
        norm = np.linalg.norm(raw_sop)
        self.reference_sop = raw_sop / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        print(f"Reference state reset to: [{self.reference_sop[0]:.2f}, {self.reference_sop[1]:.2f}, {self.reference_sop[2]:.2f}]")

    def estimate_rotation_axis_pca(self):
         if len(self.sop_history) < 10: return self.rotation_axis
         X = np.array(self.sop_history)
         X_centered = X - np.mean(X, axis=0)
         C = np.cov(X_centered.T)
         eigvals, eigvecs = np.linalg.eigh(C)
         axis = eigvecs[:, np.argmin(eigvals)]
         return axis / np.linalg.norm(axis)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if len(self.log_data) > 0: self.save_csv("log_auto1.csv")
        self.thread.stop()
        self.thread.wait()
        if hasattr(self, 'epc'):
            self.epc.close()
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
    font = QtGui.QFont("Segoe UI", 10)
    app.setFont(font)
    dashboard = PoincareDashboard()
    dashboard.show()
    sys.exit(app.exec_())