import sys
import json
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import threading
from collections import deque

class FourierSynthesizer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.fs = 44100  # Sample rate (Hz)
        self.lock = threading.Lock()
        self.num_sine_oscillators = 10  # Number of sine oscillators
        self.num_cosine_oscillators = 10  # Number of cosine oscillators
        self.num_oscillators = self.num_sine_oscillators + self.num_cosine_oscillators  # Total oscillators
        self.base_freq_values = np.ones(self.num_oscillators) * 100.00  # Base frequencies for oscillators
        self.freq_values = self.base_freq_values.copy()  # Displayed frequencies (base * multiplier)
        self.amp_values = np.zeros(self.num_oscillators)    # Amplitudes for oscillators
        self.phases = np.zeros(self.num_oscillators)        # Phase accumulators for oscillators
        self.dc_value = 0.0               # DC Offset
        self.volume = 1.0                 # Volume (0.0 to 1.0)
        self.freq_multiplier_step = 1      # Frequency Multiplier step (default 1)
        self.freq_multiplier = 1          # Frequency scaling factor (integer)
        self.cumulative_buffer = deque(maxlen=int(2.0 * self.fs))  # Buffer for waveform plotting (2 seconds)
        self.trigger_buffer = deque(maxlen=int(4.0 * self.fs))     # Buffer for trigger detection (4 seconds)
        self.last_trigger_time = -np.inf       # To track holdoff
        self.initUI()
        self.initAudio()
        self.initPlotTimer()              # Initialize the plot update timer

    def initUI(self):
        self.amplitudeControls = []
        self.frequencyControls = []
        self.waveLabels = []

        self.layout = QtWidgets.QGridLayout()

        # Create controls and labels for sine oscillators
        for i in range(self.num_sine_oscillators):
            wave_type = 'Sine'
            wave_num = i + 1
            color = 'green'   # Green color for sine waves

            label = QtWidgets.QLabel(f'{wave_type} Wave {wave_num}')
            label.setStyleSheet(f"color: {color}; font-weight: bold;")  # Set label color and bold font
            self.waveLabels.append(label)

            # Amplitude SpinBox
            amp_spinbox = QtWidgets.QDoubleSpinBox()
            amp_spinbox.setRange(-1.0, 1.0)        # Allow negative and positive amplitudes
            amp_spinbox.setSingleStep(0.0001)      # Fine step size for 4 decimal places
            amp_spinbox.setDecimals(4)             # Four decimal places
            amp_spinbox.setValue(0.0)              # Default amplitude
            amp_spinbox.setToolTip("Adjust the amplitude of the oscillator.")
            self.amplitudeControls.append(amp_spinbox)

            # Frequency SpinBox
            freq_spinbox = QtWidgets.QDoubleSpinBox()
            freq_spinbox.setRange(0.1, self.fs / 2)  # 0.1 Hz to Nyquist frequency (22050 Hz)
            freq_spinbox.setSingleStep(0.01)          # Step size of 0.01 Hz for two decimal places
            freq_spinbox.setDecimals(2)               # Two decimal places
            freq_spinbox.setValue(self.base_freq_values[i] * self.freq_multiplier)  # Default frequency
            freq_spinbox.setToolTip("Set the frequency of the oscillator in Hz.")
            self.frequencyControls.append(freq_spinbox)

            # Add widgets to the grid layout
            row = i
            self.layout.addWidget(label, row, 0)
            self.layout.addWidget(QtWidgets.QLabel('Amplitude'), row, 1)
            self.layout.addWidget(amp_spinbox, row, 2)
            self.layout.addWidget(QtWidgets.QLabel('Frequency (Hz)'), row, 3)
            self.layout.addWidget(freq_spinbox, row, 4)

            # Connect controls to the parameter update function
            amp_spinbox.valueChanged.connect(self.updateParameters)
            freq_spinbox.valueChanged.connect(self.updateOscillatorFrequency)

        # Create controls and labels for cosine oscillators
        for i in range(self.num_cosine_oscillators):
            wave_type = 'Cosine'
            wave_num = i + 1
            color = 'lime'    # Lime color for cosine waves

            label = QtWidgets.QLabel(f'{wave_type} Wave {wave_num}')
            label.setStyleSheet(f"color: {color}; font-weight: bold;")  # Set label color and bold font
            self.waveLabels.append(label)

            # Amplitude SpinBox
            amp_spinbox = QtWidgets.QDoubleSpinBox()
            amp_spinbox.setRange(-1.0, 1.0)        # Allow negative and positive amplitudes
            amp_spinbox.setSingleStep(0.0001)      # Fine step size for 4 decimal places
            amp_spinbox.setDecimals(4)             # Four decimal places
            amp_spinbox.setValue(0.0)              # Default amplitude
            amp_spinbox.setToolTip("Adjust the amplitude of the oscillator.")
            self.amplitudeControls.append(amp_spinbox)

            # Frequency SpinBox
            freq_spinbox = QtWidgets.QDoubleSpinBox()
            freq_spinbox.setRange(0.1, self.fs / 2)  # 0.1 Hz to Nyquist frequency (22050 Hz)
            freq_spinbox.setSingleStep(0.01)          # Step size of 0.01 Hz for two decimal places
            freq_spinbox.setDecimals(2)               # Two decimal places
            freq_spinbox.setValue(self.base_freq_values[self.num_sine_oscillators + i] * self.freq_multiplier)  # Default frequency
            freq_spinbox.setToolTip("Set the frequency of the oscillator in Hz.")
            self.frequencyControls.append(freq_spinbox)

            # Add widgets to the grid layout
            row = self.num_sine_oscillators + i
            self.layout.addWidget(label, row, 0)
            self.layout.addWidget(QtWidgets.QLabel('Amplitude'), row, 1)
            self.layout.addWidget(amp_spinbox, row, 2)
            self.layout.addWidget(QtWidgets.QLabel('Frequency (Hz)'), row, 3)
            self.layout.addWidget(freq_spinbox, row, 4)

            # Connect controls to the parameter update function
            amp_spinbox.valueChanged.connect(self.updateParameters)
            freq_spinbox.valueChanged.connect(self.updateOscillatorFrequency)

        # DC Offset Control
        dc_row = self.num_oscillators
        dc_label = QtWidgets.QLabel('DC Offset')
        dc_label.setStyleSheet("color: cyan; font-weight: bold;")  # Cyan color for DC Offset
        self.layout.addWidget(dc_label, dc_row, 0)

        self.dcSpinBox = QtWidgets.QDoubleSpinBox()
        self.dcSpinBox.setRange(-1.0, 1.0)     # DC offset range
        self.dcSpinBox.setSingleStep(0.0001)   # Step size for 4 decimal places
        self.dcSpinBox.setDecimals(4)          # Four decimal places
        self.dcSpinBox.setValue(0.0)           # Default DC offset
        self.dcSpinBox.setToolTip("Adjust the DC offset of the waveform.")
        self.layout.addWidget(self.dcSpinBox, dc_row, 1)

        self.dcSpinBox.valueChanged.connect(self.updateParameters)

        # Time Scale Control
        time_scale_row = dc_row + 1
        time_scale_label = QtWidgets.QLabel('Time Scale (s)')
        time_scale_label.setStyleSheet("color: yellow; font-weight: bold;")  # Yellow color for Time Scale
        self.layout.addWidget(time_scale_label, time_scale_row, 0)

        self.timeScaleSpinBox = QtWidgets.QDoubleSpinBox()
        self.timeScaleSpinBox.setRange(0.00002, 2.0)    # Increased upper limit to allow longer durations and minimum to ensure at least 1 sample
        self.timeScaleSpinBox.setSingleStep(0.00001)    # Step size of 0.00001 seconds
        self.timeScaleSpinBox.setDecimals(5)            # Five decimal places
        self.timeScaleSpinBox.setValue(0.05)            # Default duration set to 0.05 seconds (5 cycles at 100 Hz)
        self.timeScaleSpinBox.setToolTip("Set the duration of the waveform display in seconds.")
        self.layout.addWidget(self.timeScaleSpinBox, time_scale_row, 1)

        self.timeScaleSpinBox.valueChanged.connect(self.updateParameters)

        # Trigger Controls
        trigger_row = time_scale_row + 1
        trigger_threshold_label = QtWidgets.QLabel('Trigger Threshold')
        trigger_threshold_label.setStyleSheet("color: magenta; font-weight: bold;")  # Magenta color for Trigger Threshold
        self.layout.addWidget(trigger_threshold_label, trigger_row, 0)

        self.triggerThresholdSpinBox = QtWidgets.QDoubleSpinBox()
        self.triggerThresholdSpinBox.setRange(-1.0, 1.0)  # Adjust range as needed
        self.triggerThresholdSpinBox.setSingleStep(0.01)
        self.triggerThresholdSpinBox.setDecimals(2)
        self.triggerThresholdSpinBox.setValue(0.0)       # Default threshold
        self.triggerThresholdSpinBox.setToolTip("Set the amplitude threshold for trigger detection.")
        self.layout.addWidget(self.triggerThresholdSpinBox, trigger_row, 1)

        trigger_edge_label = QtWidgets.QLabel('Trigger Edge')
        trigger_edge_label.setStyleSheet("color: magenta; font-weight: bold;")  # Magenta color for Trigger Edge
        self.layout.addWidget(trigger_edge_label, trigger_row, 2)

        self.triggerEdgeComboBox = QtWidgets.QComboBox()
        self.triggerEdgeComboBox.addItems(['Rising', 'Falling'])
        self.triggerEdgeComboBox.setCurrentText('Rising')  # Default edge
        self.triggerEdgeComboBox.setToolTip("Select the edge type for trigger detection.")
        self.layout.addWidget(self.triggerEdgeComboBox, trigger_row, 3)

        # Connect Trigger controls to updateParameters
        self.triggerThresholdSpinBox.valueChanged.connect(self.updateParameters)
        self.triggerEdgeComboBox.currentIndexChanged.connect(self.updateParameters)

        # Trigger Holdoff Control
        holdoff_row = trigger_row + 1
        holdoff_label = QtWidgets.QLabel('Trigger Holdoff (ms)')
        holdoff_label.setStyleSheet("color: orange; font-weight: bold;")  # Orange color for Holdoff
        self.layout.addWidget(holdoff_label, holdoff_row, 0)

        self.triggerHoldoffSpinBox = QtWidgets.QDoubleSpinBox()
        self.triggerHoldoffSpinBox.setRange(0.0, 1000.0)  # 0 ms to 1000 ms
        self.triggerHoldoffSpinBox.setSingleStep(10.0)     # Step size of 10 ms
        self.triggerHoldoffSpinBox.setDecimals(1)         # One decimal place
        self.triggerHoldoffSpinBox.setValue(100.0)        # Default holdoff of 100 ms
        self.triggerHoldoffSpinBox.setToolTip("Set the holdoff time in milliseconds to prevent rapid triggering.")
        self.layout.addWidget(self.triggerHoldoffSpinBox, holdoff_row, 1)

        # **Volume Control**
        volume_row = holdoff_row + 1
        volume_label = QtWidgets.QLabel('Volume')
        volume_label.setStyleSheet("color: blue; font-weight: bold;")  # Blue color for Volume
        self.layout.addWidget(volume_label, volume_row, 0)

        self.volumeSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.volumeSlider.setRange(0, 100)  # 0% to 100%
        self.volumeSlider.setValue(100)     # Default volume at 100%
        self.volumeSlider.setTickInterval(10)
        self.volumeSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.volumeSlider.setToolTip("Adjust the overall output volume.")
        self.layout.addWidget(self.volumeSlider, volume_row, 1, 1, 3)

        self.volumeLabel = QtWidgets.QLabel('100%')
        self.volumeLabel.setStyleSheet("color: blue; font-weight: bold;")
        self.layout.addWidget(self.volumeLabel, volume_row, 4)

        # Connect Volume Slider to updateVolume
        self.volumeSlider.valueChanged.connect(self.updateVolume)

        # **Frequency Multiplier Control**
        freq_multiplier_row = volume_row + 1
        freq_multiplier_label = QtWidgets.QLabel('Frequency Multiplier')
        freq_multiplier_label.setStyleSheet("color: red; font-weight: bold;")  # Red color for Frequency Multiplier
        self.layout.addWidget(freq_multiplier_label, freq_multiplier_row, 0)

        self.freqMultiplierSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.freqMultiplierSlider.setRange(1, 10)   # Multiplier range from 1 to 10
        self.freqMultiplierSlider.setValue(1)       # Default multiplier is 1
        self.freqMultiplierSlider.setSingleStep(1)   # Step size of 1
        self.freqMultiplierSlider.setTickInterval(1)
        self.freqMultiplierSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.freqMultiplierSlider.setToolTip("Multiply all oscillator frequencies by this integer factor.")
        self.layout.addWidget(self.freqMultiplierSlider, freq_multiplier_row, 1, 1, 2)

        self.freqMultiplierLabel = QtWidgets.QLabel('1')
        self.freqMultiplierLabel.setStyleSheet("color: red; font-weight: bold;")
        self.layout.addWidget(self.freqMultiplierLabel, freq_multiplier_row, 3)

        # Connect Frequency Multiplier Slider to updateParameters and update label
        self.freqMultiplierSlider.valueChanged.connect(self.freqMultiplierChanged)

        # Output Device Selection
        device_combo_row = freq_multiplier_row + 1
        device_label = QtWidgets.QLabel('Output Device')
        device_label.setStyleSheet("color: white; font-weight: bold;")  # White color for Output Device
        self.layout.addWidget(device_label, device_combo_row, 0)

        self.deviceComboBox = QtWidgets.QComboBox()
        self.output_device_indices = []
        devices = self.getOutputDeviceNames()
        for idx, name in devices:
            self.output_device_indices.append(idx)
            self.deviceComboBox.addItem(name)
        if self.output_device_indices:
            self.device = self.output_device_indices[0]
        else:
            self.device = None  # Handle case with no output devices
        self.deviceComboBox.setToolTip("Select the audio output device.")
        self.deviceComboBox.currentIndexChanged.connect(self.changeDevice)

        self.layout.addWidget(self.deviceComboBox, device_combo_row, 1, 1, 4)

        # Save and Load Buttons for All Settings
        save_load_row = device_combo_row + 1
        save_button = QtWidgets.QPushButton("Save All Settings")
        load_button = QtWidgets.QPushButton("Load All Settings")
        save_button.setToolTip("Save all synthesizer settings to a JSON file.")
        load_button.setToolTip("Load synthesizer settings from a JSON file.")
        self.layout.addWidget(save_button, save_load_row, 0)
        self.layout.addWidget(load_button, save_load_row, 1)

        # Connect buttons to their respective functions
        save_button.clicked.connect(self.saveAllSettings)
        load_button.clicked.connect(self.loadAllSettings)

        # Save and Load Buttons for Oscillator Settings
        osc_save_load_row = save_load_row + 1
        save_osc_button = QtWidgets.QPushButton("Save Oscillator Settings")
        load_osc_button = QtWidgets.QPushButton("Load Oscillator Settings")
        save_osc_button.setToolTip("Save only oscillator amplitudes and frequencies to a JSON file.")
        load_osc_button.setToolTip("Load oscillator amplitudes and frequencies from a JSON file.")
        self.layout.addWidget(save_osc_button, osc_save_load_row, 0)
        self.layout.addWidget(load_osc_button, osc_save_load_row, 1)

        # Connect oscillator buttons to their respective functions
        save_osc_button.clicked.connect(self.saveOscillatorSettings)
        load_osc_button.clicked.connect(self.loadOscillatorSettings)

        # Waveform Visualization Plot
        plot_row = osc_save_load_row + 1
        self.plotWidget = pg.PlotWidget(background='k', foreground='g')  # Black background, green foreground
        self.plotWidget.showGrid(x=True, y=True, alpha=0.3)               # Green grid lines with transparency
        self.plotCurve = self.plotWidget.plot([], [], pen=pg.mkPen(color='lime', width=2))  # Bright green waveform
        self.plotWidget.setYRange(-1, 1)
        self.plotWidget.setXRange(0, self.timeScaleSpinBox.value())  # Start at 0
        self.layout.addWidget(self.plotWidget, plot_row, 0, 1, 5)

        # Add Trigger Line Indicator
        self.triggerLine = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('g', width=2))  # Green vertical line
        self.plotWidget.addItem(self.triggerLine)

        self.setLayout(self.layout)
        self.setWindowTitle('Fourier Synthesizer with DC Offset, Scalable Time Axis, Triggering, Volume Control, and Frequency Multiplier')

    def getOutputDeviceNames(self):
        try:
            devices = sd.query_devices()
            output_devices = []
            for idx, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    output_devices.append((idx, device['name']))
            if not output_devices:
                QtWidgets.QMessageBox.warning(self, "No Output Devices", "No audio output devices found.")
            return output_devices
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to query audio devices:\n{e}")
            return []

    def changeDevice(self, index):
        if not self.output_device_indices:
            QtWidgets.QMessageBox.warning(self, "No Output Device", "No output devices are available.")
            return
        # Change the output device
        self.device = self.output_device_indices[index]
        # Restart the audio stream with the new device
        try:
            if hasattr(self, 'stream') and self.stream is not None:
                self.stream.stop()
                self.stream.close()
            self.initAudio()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to change output device:\n{e}")

    def initAudio(self):
        if self.device is None:
            QtWidgets.QMessageBox.warning(self, "No Output Device", "No output device selected.")
            return
        # Initialize audio stream
        with self.lock:
            self.phases = np.zeros(self.num_oscillators)  # All phases start at 0

        try:
            self.stream = sd.OutputStream(device=self.device,
                                          samplerate=self.fs,
                                          channels=1,
                                          callback=self.audioCallback)
            self.stream.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to initialize audio stream:\n{e}")

    def initPlotTimer(self):
        # Timer to update the plot periodically
        self.plotTimer = QtCore.QTimer()
        self.plotTimer.setInterval(50)  # Update every 50 ms
        self.plotTimer.timeout.connect(self.updatePlot)
        self.plotTimer.start()

    def audioCallback(self, outdata, frames, time_info, status):
        if status:
            print(status)
        out = np.zeros(frames)

        # Thread-safe access to amplitude, frequency, DC, and trigger values
        with self.lock:
            amp_values = self.amp_values.copy()
            freq_values = self.freq_values.copy()  # Already multiplied by freq_multiplier
            phases = self.phases.copy()
            dc = self.dc_value
            trigger_threshold = self.triggerThresholdSpinBox.value()
            trigger_edge = self.triggerEdgeComboBox.currentText()
            plot_duration = self.timeScaleSpinBox.value()
            volume = self.volume  # Get the current volume

        for i in range(self.num_oscillators):
            amp = amp_values[i]
            freq = freq_values[i]  # Already scaled
            phase_increment = 2 * np.pi * freq / self.fs
            phase_array = phases[i] + phase_increment * np.arange(frames)

            if i < self.num_sine_oscillators:
                # Sine waves
                out += amp * np.sin(phase_array)
            else:
                # Cosine waves
                out += amp * np.cos(phase_array)

            # Update phase for continuity
            phases[i] += phase_increment * frames
            phases[i] %= 2 * np.pi  # Keep phase within [0, 2π]

        # Add DC offset
        out += dc

        # Apply volume control
        out *= volume

        # Normalize the output to prevent clipping
        max_amp = np.max(np.abs(out))
        total_amp = max_amp + abs(dc)
        if total_amp > 1.0:
            out /= total_amp  # Normalize considering DC offset

        # Write to audio buffer
        outdata[:, 0] = out

        # Update phases in a thread-safe manner and manage buffers
        with self.lock:
            self.phases = phases

            # Accumulate data for plotting
            self.cumulative_buffer.extend(out)
            # No need to check length due to deque's maxlen

            # Accumulate data for trigger detection
            self.trigger_buffer.extend(out)
            # No need to check length due to deque's maxlen

    def detectTrigger(self, buffer, threshold, edge):
        """
        Detect the trigger event in the buffer.

        Parameters:
        - buffer (deque): The audio samples buffer.
        - threshold (float): The amplitude threshold for triggering.
        - edge (str): 'Rising' or 'Falling'.

        Returns:
        - int or None: The index where the trigger occurs, or None if not found.
        """
        buffer_array = np.array(buffer)
        if edge == 'Rising':
            # Look for a transition from below to above the threshold
            crossings = np.where((buffer_array[:-1] < threshold) & (buffer_array[1:] >= threshold))[0]
        else:
            # Look for a transition from above to below the threshold
            crossings = np.where((buffer_array[:-1] > threshold) & (buffer_array[1:] <= threshold))[0]

        if crossings.size > 0:
            # Return the last trigger point found for stability
            return crossings[-1] + 1  # +1 to get the index after crossing
        else:
            return None

    def updatePlot(self):
        with self.lock:
            trigger_buffer = list(self.trigger_buffer)
            duration = self.timeScaleSpinBox.value()
            trigger_threshold = self.triggerThresholdSpinBox.value()
            trigger_edge = self.triggerEdgeComboBox.currentText()
            holdoff_time = self.triggerHoldoffSpinBox.value()

        if len(trigger_buffer) == 0:
            return  # Nothing to plot

        # Detect trigger in the trigger buffer
        trigger_index = self.detectTrigger(trigger_buffer, trigger_threshold, trigger_edge)

        if trigger_index is not None:
            # Get the current time in milliseconds
            current_time = QtCore.QDateTime.currentDateTime().toMSecsSinceEpoch()
            # Check holdoff to prevent multiple rapid triggers
            if (current_time - self.last_trigger_time) >= holdoff_time:
                # Align the plot starting from the trigger point
                aligned_buffer = trigger_buffer[trigger_index:]
                self.last_trigger_time = current_time  # Update last trigger time

                # Position the trigger line at the start
                self.triggerLine.setPos(0)
            else:
                # Ignore triggers within holdoff time
                aligned_buffer = trigger_buffer[-int(duration * self.fs):]
                self.triggerLine.setPos(0)  # Reset trigger line position
        else:
            # If no trigger found, use the most recent samples
            aligned_buffer = trigger_buffer[-int(duration * self.fs):]
            self.triggerLine.setPos(0)  # Reset trigger line position

        # Adjust the buffer to fit the plot duration
        if len(aligned_buffer) > int(duration * self.fs):
            aligned_buffer = aligned_buffer[-int(duration * self.fs):]

        t = np.linspace(0, len(aligned_buffer) / self.fs, len(aligned_buffer), endpoint=False)  # Start at 0
        waveform = np.array(aligned_buffer)

        # Update the plot data
        self.plotCurve.setData(t, waveform)

        # Adjust Y-axis range based on maximum amplitude including DC
        max_amp = np.max(np.abs(waveform))
        if max_amp == 0:
            max_amp = 1  # Avoid division by zero
        self.plotWidget.setYRange(-max_amp, max_amp)
        self.plotWidget.setXRange(0, duration)  # X-axis starts at 0

    def updateParameters(self, value=None):
        """
        Update amplitude, DC, Time Scale, Trigger settings in a thread-safe manner.
        """
        with self.lock:
            for i in range(self.num_oscillators):
                self.amp_values[i] = self.amplitudeControls[i].value()
            self.dc_value = self.dcSpinBox.value()  # Update DC offset
            # Frequency multiplier is handled separately
        # No need to call generateWaveform as plotting is handled by the timer

    def updateOscillatorFrequency(self, value=None):
        """
        Update base frequencies based on user input in frequency spin boxes.
        """
        sender = self.sender()
        if sender in self.frequencyControls:
            index = self.frequencyControls.index(sender)
            scaled_freq = sender.value()
            if self.freq_multiplier != 0:
                base_freq = scaled_freq / self.freq_multiplier
            else:
                base_freq = scaled_freq
            with self.lock:
                self.base_freq_values[index] = base_freq
                self.freq_values[index] = self.base_freq_values[index] * self.freq_multiplier
        # No need to call generateWaveform as plotting is handled by the timer

    def freqMultiplierChanged(self, value):
        """
        Handle changes in the Frequency Multiplier Slider.
        """
        # Update the label to reflect the current multiplier step
        self.freqMultiplierLabel.setText(str(value))
        with self.lock:
            self.freq_multiplier = value
            # Update displayed frequencies based on the new multiplier
            new_freq_values = self.base_freq_values * self.freq_multiplier

        # Block signals to prevent triggering updateOscillatorFrequency
        for spinbox in self.frequencyControls:
            spinbox.blockSignals(True)
            spinbox.setValue(new_freq_values[self.frequencyControls.index(spinbox)])
            spinbox.blockSignals(False)

        # Update internal freq_values
        with self.lock:
            self.freq_values = new_freq_values.copy()

    def updateVolume(self, value):
        """
        Update the volume based on the slider value.
        """
        with self.lock:
            self.volume = value / 100.0  # Convert percentage to 0.0 - 1.0
        self.volumeLabel.setText(f"{value}%")

    def saveAllSettings(self):
        """
        Save all synthesizer settings to a JSON file.
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save All Settings", "", "JSON Files (*.json)", options=options)
        if filename:
            with self.lock:
                settings = {
                    'base_frequencies': self.base_freq_values.tolist(),
                    'amplitudes': self.amp_values.tolist(),
                    'dc_offset': self.dc_value,
                    'time_scale': self.timeScaleSpinBox.value(),
                    'trigger_threshold': self.triggerThresholdSpinBox.value(),
                    'trigger_edge': self.triggerEdgeComboBox.currentText(),
                    'trigger_holdoff': self.triggerHoldoffSpinBox.value(),
                    'volume': self.volume,    # Save as float
                    'frequency_multiplier': self.freq_multiplier  # Save multiplier
                }
            try:
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=4)
                QtWidgets.QMessageBox.information(self, "Success", f"All settings saved to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")

    def loadAllSettings(self):
        """
        Load all synthesizer settings from a JSON file.
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load All Settings", "", "JSON Files (*.json)", options=options)
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                # Validate JSON keys
                required_keys = ['base_frequencies', 'amplitudes', 'dc_offset', 'time_scale', 'trigger_threshold', 'trigger_edge', 'trigger_holdoff', 'volume', 'frequency_multiplier']
                for key in required_keys:
                    if key not in settings:
                        raise ValueError(f"Missing key in JSON: {key}")

                # Validate lengths
                base_frequencies = settings.get('base_frequencies', [100.0]*self.num_oscillators)
                amplitudes = settings.get('amplitudes', [0.0]*self.num_oscillators)
                if len(base_frequencies) != self.num_oscillators or len(amplitudes) != self.num_oscillators:
                    raise ValueError("Base frequencies and amplitude lists must match the number of oscillators.")

                # Extract other settings
                dc_offset = settings.get('dc_offset', 0.0)
                time_scale = settings.get('time_scale', 0.05)
                trigger_threshold = settings.get('trigger_threshold', 0.0)
                trigger_edge = settings.get('trigger_edge', 'Rising')
                trigger_holdoff = settings.get('trigger_holdoff', 100.0)
                volume = settings.get('volume', 1.0)
                freq_multiplier = settings.get('frequency_multiplier', 1)

                # Block signals to prevent updateParameters from being called multiple times
                for spinbox in self.amplitudeControls + self.frequencyControls + [
                    self.dcSpinBox, self.timeScaleSpinBox,
                    self.triggerThresholdSpinBox, self.triggerHoldoffSpinBox,
                    self.freqMultiplierSlider
                ]:
                    spinbox.blockSignals(True)
                self.triggerEdgeComboBox.blockSignals(True)
                self.volumeSlider.blockSignals(True)

                # Apply base frequencies
                with self.lock:
                    self.base_freq_values = np.array(base_frequencies)
                    self.freq_multiplier = freq_multiplier
                    self.freq_values = self.base_freq_values * self.freq_multiplier

                # Update frequency multiplier slider
                self.freqMultiplierSlider.setValue(freq_multiplier)

                # Apply amplitudes
                for i in range(self.num_oscillators):
                    self.amplitudeControls[i].setValue(amplitudes[i])

                # Apply frequencies
                for i in range(self.num_oscillators):
                    self.frequencyControls[i].setValue(self.freq_values[i])

                # Apply other settings
                self.dcSpinBox.setValue(dc_offset)
                self.timeScaleSpinBox.setValue(time_scale)
                self.triggerThresholdSpinBox.setValue(trigger_threshold)
                edge_index = self.triggerEdgeComboBox.findText(trigger_edge)
                if edge_index != -1:
                    self.triggerEdgeComboBox.setCurrentIndex(edge_index)
                self.triggerHoldoffSpinBox.setValue(trigger_holdoff)
                self.volumeSlider.setValue(int(volume * 100))
                self.volumeLabel.setText(f"{int(volume * 100)}%")

                # Unblock signals
                for spinbox in self.amplitudeControls + self.frequencyControls + [
                    self.dcSpinBox, self.timeScaleSpinBox,
                    self.triggerThresholdSpinBox, self.triggerHoldoffSpinBox,
                    self.freqMultiplierSlider
                ]:
                    spinbox.blockSignals(False)
                self.triggerEdgeComboBox.blockSignals(False)
                self.volumeSlider.blockSignals(False)

                # Update internal parameters
                with self.lock:
                    self.amp_values = np.array(amplitudes)
                    # freq_values already updated

                QtWidgets.QMessageBox.information(self, "Success", f"All settings loaded from {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load settings:\n{e}")

    def saveOscillatorSettings(self):
        """
        Save only oscillator settings (base frequencies and amplitudes) to a JSON file.
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Oscillator Settings", "", "JSON Files (*.json)", options=options)
        if filename:
            with self.lock:
                osc_settings = {
                    'base_frequencies': self.base_freq_values.tolist(),
                    'amplitudes': self.amp_values.tolist()
                }
            try:
                with open(filename, 'w') as f:
                    json.dump(osc_settings, f, indent=4)
                QtWidgets.QMessageBox.information(self, "Success", f"Oscillator settings saved to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save oscillator settings:\n{e}")

    def loadOscillatorSettings(self):
        """
        Load only oscillator settings (base frequencies and amplitudes) from a JSON file.
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Oscillator Settings", "", "JSON Files (*.json)", options=options)
        if filename:
            try:
                with open(filename, 'r') as f:
                    osc_settings = json.load(f)
                # Validate JSON keys
                required_keys = ['base_frequencies', 'amplitudes']
                for key in required_keys:
                    if key not in osc_settings:
                        raise ValueError(f"Missing key in JSON: {key}")

                # Validate lengths
                base_frequencies = osc_settings.get('base_frequencies', [100.0]*self.num_oscillators)
                amplitudes = osc_settings.get('amplitudes', [0.0]*self.num_oscillators)
                if len(base_frequencies) != self.num_oscillators or len(amplitudes) != self.num_oscillators:
                    raise ValueError("Base frequencies and amplitude lists must match the number of oscillators.")

                # Extract other settings
                # Frequency multiplier remains unchanged

                # Block signals to prevent updateOscillatorFrequency from being called multiple times
                for spinbox in self.amplitudeControls + self.frequencyControls:
                    spinbox.blockSignals(True)

                # Apply base frequencies
                with self.lock:
                    self.base_freq_values = np.array(base_frequencies)
                    self.freq_values = self.base_freq_values * self.freq_multiplier

                # Update frequency spin boxes
                for i in range(self.num_oscillators):
                    self.frequencyControls[i].setValue(self.freq_values[i])

                # Apply amplitudes
                with self.lock:
                    self.amp_values = np.array(amplitudes)
                for i in range(self.num_oscillators):
                    self.amplitudeControls[i].setValue(amplitudes[i])

                # Unblock signals
                for spinbox in self.amplitudeControls + self.frequencyControls:
                    spinbox.blockSignals(False)

                QtWidgets.QMessageBox.information(self, "Success", f"Oscillator settings loaded from {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load oscillator settings:\n{e}")

    def closeEvent(self, event):
        """
        Handle the window close event to ensure resources are properly released.
        """
        try:
            if hasattr(self, 'stream') and self.stream is not None:
                self.stream.stop()
                self.stream.close()
            if hasattr(self, 'plotTimer') and self.plotTimer.isActive():
                self.plotTimer.stop()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    synth = FourierSynthesizer()
    synth.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
