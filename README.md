Okay, here is a README.md file generated based on the Python script you provided. It includes sections for description, features, requirements, installation, and usage.


# Dash Audio Frequency Visualizer

A real-time audio visualizer built with Python, Dash, Plotly, and Dash Mantine Components. It captures audio from a selected input device and displays the waveform, frequency spectrum, and frequency band energy distribution.

## Features

* **Real-time Audio Capture:** Uses PyAudio to capture audio data from your microphone or other input devices.
* **Input Device Selection:** Allows users to choose from available audio input devices on their system.
* **Multiple Visualizations:**
    * **Waveform:** Displays the audio amplitude over time using a dynamic gradient-filled line chart (`dmc.LineChart`).
    * **Frequency Spectrum:** Shows the magnitude of different frequencies present in the audio using a bar chart (`dcc.Graph` with Plotly) with dynamic coloring.
    * **Frequency Bands Radar:** Visualizes the energy distribution across predefined, logarithmically spaced frequency bands using a radar chart (`dmc.RadarChart`).
* **User Controls:**
    * Start/Stop recording buttons.
    * Adjustable input sensitivity (gain) slider.
    * Adjustable animation speed slider for visualizations.
* **UI Components:** Uses Dash Mantine Components for a clean, modern, dark-themed interface.
* **Debug Information:** Displays helpful debug info like audio queue size, RMS level, and processing status.
* **Signal Level Indicator:** Provides a visual progress bar indicating the current input signal strength (RMS).

## Requirements

### System Dependencies

* **Python:** 3.8+ recommended.
* **PortAudio Library:** `pyaudio` requires the PortAudio library development files to be installed on your system.
    * **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install -y portaudio19-dev`
    * **Fedora/CentOS/RHEL:** `sudo dnf install portaudio-devel` (or `yum`)
    * **macOS (using Homebrew):** `brew install portaudio`
    * **Windows:** Installation can be complex. Consider using pre-compiled wheels or Anaconda.
* **Audio Input Device:** A working microphone or other audio input connected to your system.

### Python Packages

The required Python packages are listed in `requirements.txt`:

```
dash>=2.9.0
dash-mantine-components
plotly>=5.14.0
pyaudio>=0.2.13
numpy>=1.23.0
scipy>=1.10.0
```

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone https://github.com/pip-install-python/audio-frequency-viewer.git
    ```

2.  **Install System Dependency (PortAudio):**
    Run the appropriate command for your operating system (see System Dependencies section above). For Debian/Ubuntu:
    ```bash
    sudo apt-get update && sudo apt-get install -y portaudio19-dev
    ```

3.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Dash application:**
    ```bash
    python app.py
    ```

2.  **Open your web browser** and navigate to the address shown in the terminal (usually `http://127.0.0.1:8888`).

3.  **Select your microphone** or desired audio input device from the "Select Input Device" dropdown menu.

4.  **Adjust Sensitivity:** Use the "Input Sensitivity" slider to increase or decrease the gain applied to the input signal for visualization purposes. Higher values make quieter sounds more visible.

5.  **Adjust Animation Speed:** Use the "Animation Speed" slider to control how quickly the charts update their transitions (lower values are faster).

6.  **Click the "Start Recording" button.**

7.  **Make some noise!** Speak into your microphone or play audio near it. You should see the Waveform, Frequency Spectrum, and Frequency Bands Radar charts update in real-time.

8.  **Click the "Stop Recording" button** when you are finished.

## License

MIT License



