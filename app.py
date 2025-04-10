import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, State, _dash_renderer, ctx
import dash_mantine_components as dmc
import pyaudio
import threading
import time
import queue
from scipy.fft import fft
import colorsys

_dash_renderer._set_react_version("18.2.0")

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
MAX_PLOT_POINTS = 1024  # Number of points to display in waveform

# Initialize PyAudio and data structures
p = pyaudio.PyAudio()
q = queue.Queue()

# Global variables to hold visualization data
audio_data = np.zeros(MAX_PLOT_POINTS)
fft_data = np.zeros(CHUNK // 2)
previous_band_values = {}  # Store previous values for more stable color transitions
debug_info = {"last_processed": 0, "queue_size": 0, "max_amplitude": 0, "update_count": 0, "rms_level": 0}
recording_active = False

# Audio processing lock to prevent race conditions
data_lock = threading.Lock()

# Get list of audio input devices
input_devices = []
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info.get('maxInputChannels') > 0:  # This is an input device
        input_devices.append({
            'index': i,
            'name': device_info.get('name'),
            'channels': device_info.get('maxInputChannels'),
            'default': p.get_default_input_device_info().get('index') == i
        })

# Define frequency bands for radar chart
# Format the band names as requested by the user
freq_bands = [
    "20-35 Hz",
    "35-63 Hz",
    "63-112 Hz",
    "112-200 Hz",
    "200-355 Hz",
    "355-632 Hz",
    "632-1124 Hz",
    "1124-2000 Hz",
    "2000-3556 Hz",
    "3556-6324 Hz",
    "6324-11246 Hz",
    "11246-20000 Hz"
]

# Create logarithmically spaced frequency band edges
freq_min = 20  # Hz, lowest audible frequency
freq_max = 20000  # Hz, highest audible frequency
num_bands = len(freq_bands)
band_edges = np.logspace(np.log10(freq_min), np.log10(freq_max), num_bands + 1)

# Initialize previous band values
for band in freq_bands:
    previous_band_values[band] = 0

# Color palette for different intensity levels (from cool to hot) using explicit HEX values
# Each list represents [fill color, stroke color, opacity] for a given intensity level
color_palette = [
    ["#0d47a1", "#002171", 0.4],  # Deep blue (low frequencies)
    ["#1565c0", "#003c8f", 0.45], # Blue
    ["#1976d2", "#004ba0", 0.5],  # Medium blue
    ["#0288d1", "#005b9f", 0.5],  # Light blue
    ["#0097a7", "#006064", 0.55], # Cyan
    ["#00796b", "#004d40", 0.6],  # Teal
    ["#2e7d32", "#005005", 0.6],  # Green
    ["#558b2f", "#255d00", 0.65], # Light green
    ["#9e9d24", "#6c6f00", 0.7],  # Lime
    ["#f9a825", "#c17900", 0.7],  # Amber
    ["#ef6c00", "#b53d00", 0.75], # Orange
    ["#d84315", "#9f0000", 0.8],  # Deep orange
    ["#c62828", "#8e0000", 0.85], # Red
    ["#ad1457", "#78002e", 0.9],  # Pink
    ["#6a1b9a", "#38006b", 0.9],  # Purple (high frequencies)
]

# Extract just the fill colors for the colorscale
color_scale_values = [c[0] for c in color_palette]


# Function to capture audio
def audio_callback(in_data, frame_count, time_info, status):
    if recording_active:
        q.put(in_data)
    return (in_data, pyaudio.paContinue)


# Function to process audio data
def process_audio():
    global audio_data, fft_data, debug_info, recording_active, previous_band_values

    while recording_active:
        if not q.empty():
            # Get audio data from queue
            data = q.get()
            decoded = np.frombuffer(data, dtype=np.int16)

            # Calculate RMS value (root mean square - a measure of signal strength)
            rms = np.sqrt(np.mean(np.square(decoded.astype(np.float32))))

            # Update debug info
            debug_info["queue_size"] = q.qsize()
            debug_info["last_processed"] = time.time()
            debug_info["max_amplitude"] = np.max(np.abs(decoded))
            debug_info["rms_level"] = rms
            debug_info["update_count"] += 1

            # Normalize for waveform visualization
            normalized = decoded / 32768.0

            # Apply a MUCH higher gain to make small signals more visible
            gain = 20.0  # Increased from 5.0 to 20.0
            normalized = np.clip(normalized * gain, -1.0, 1.0)  # Amplify but clip to valid range

            with data_lock:
                # Update waveform data with rolling window
                audio_data = np.roll(audio_data, -len(normalized))
                audio_data[-len(normalized):] = normalized

                # Compute FFT for frequency visualization
                # Apply window function to reduce spectral leakage
                windowed = normalized * np.hamming(len(normalized))
                fft_result = fft(windowed)

                # Get the magnitude, scale it, and apply some dynamic range compression
                raw_fft = np.abs(fft_result[:CHUNK // 2]) / (CHUNK // 16)  # Increased scaling factor

                # Apply logarithmic scaling to enhance small signals
                log_scaling = 0.1
                fft_data = np.log1p(raw_fft / log_scaling) * log_scaling * 4

        time.sleep(0.01)


# Function to determine color based on value
def get_color_for_value(value, max_value=100):
    """
    Returns a color based on the value relative to max_value
    Higher values get "hotter" colors
    """
    # Normalize value to 0-1 range
    normalized = min(max(value / max_value, 0), 1)

    # Determine color index (0-9)
    index = min(int(normalized * len(color_palette)), len(color_palette) - 1)

    return color_palette[index]


# Function to create a custom colorscale based on intensity
def get_custom_colorscale(intensity, max_intensity=0.5):
    """Create a custom colorscale that shifts based on intensity"""
    # Normalize intensity to 0-1 range with a maximum threshold
    normalized = min(intensity / max_intensity, 1.0)

    # Determine which colors to include
    num_colors = int(normalized * len(color_scale_values)) + 1
    num_colors = max(2, min(num_colors, len(color_scale_values)))

    # Create a colorscale with selected colors
    selected_colors = color_scale_values[:num_colors]

    # Create the colorscale for plotly (pairs of position and color)
    colorscale = []
    for i, color in enumerate(selected_colors):
        position = i / (len(selected_colors) - 1)
        colorscale.append([position, color])

    return colorscale


# Function to convert FFT data to radar chart format with a simpler approach
def fft_to_radar_data(fft_data):
    global previous_band_values

    # Convert frequency bin indices to actual frequencies
    frequencies = np.arange(len(fft_data)) * (RATE / (2 * CHUNK))

    # Initialize data object for the radar chart
    radar_data = []

    # Loop through each frequency band
    for i, band_name in enumerate(freq_bands):
        if i >= len(band_edges) - 1:
            continue

        # Get frequency range for this band
        low = int(band_edges[i])
        high = int(band_edges[i + 1])

        # Find frequencies in this range
        indices = np.where((frequencies >= low) & (frequencies < high))[0]

        # Calculate band value with amplification for better visibility
        if len(indices) > 0:
            band_avg = np.mean(fft_data[indices])
            current_value = min(100, band_avg * 100 * 8)
        else:
            current_value = 0

        # Smooth with previous values
        if band_name in previous_band_values:
            smoothed_value = 0.7 * current_value + 0.3 * previous_band_values[band_name]
        else:
            smoothed_value = current_value

        # Save for next time
        previous_band_values[band_name] = smoothed_value

        # Create an entry in the radar_data for this band
        radar_data.append({
            "band": band_name,
            "value": smoothed_value
        })

    return radar_data


# Function to convert audio data to LineChart format
# Function to convert audio data to LineChart format with enhanced gradient
def audio_to_linechart_data(audio_data, fft_data):
    # Get frequency information across the spectrum
    frequencies = np.arange(len(fft_data)) * (RATE / (2 * CHUNK))

    # Divide the frequency spectrum into bands for richer visualization
    # Use logarithmic bands to better match human hearing
    freq_bands_edges = [20, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000, 20000]
    band_energies = []

    # Calculate energy in each frequency band
    for i in range(len(freq_bands_edges) - 1):
        low = freq_bands_edges[i]
        high = freq_bands_edges[i + 1]
        indices = np.where((frequencies >= low) & (frequencies < high))[0]
        if len(indices) > 0:
            band_energy = np.mean(fft_data[indices])
            # Amplify for better visualization
            band_energies.append(min(1.0, band_energy * 50))
        else:
            band_energies.append(0.0)

    # Find the dominant frequency band
    if len(band_energies) > 0 and max(band_energies) > 0:
        dominant_band = np.argmax(band_energies)
    else:
        dominant_band = 0

    # Format audio data for LineChart
    # We'll use the sample index as the x value and amplitude as the y value
    waveform_data = []

    # Take a subset of points to prevent overcrowding
    step = 8  # Skip every 8 points to reduce data size
    for i in range(0, len(audio_data), step):
        waveform_data.append({
            "time": i,  # X-axis: sample index
            "amplitude": audio_data[i]  # Y-axis: amplitude
        })

    # Create a more colorful gradient with more stops
    # Use all colors from the palette and distribute them based on frequency activity
    gradient_stops = []

    # Add more gradient stops for richer colors (at least 5 stops)
    num_stops = len(color_palette)

    # Create gradient stops with dynamic positioning based on frequency content
    for i in range(num_stops):
        # Adjust offset based on the dominant frequency band
        # This shifts colors toward the part of the spectrum that's most active
        base_offset = (i / (num_stops - 1)) * 100

        # Shift the offsets to emphasize the dominant frequency range
        offset_shift = 0
        if dominant_band > 0:
            # Shift colors based on which part of the spectrum is active
            # Low frequencies shift toward beginning, high frequencies toward end
            normalized_dominant = dominant_band / (len(band_energies) - 1)
            offset_shift = (normalized_dominant - 0.5) * 20  # Shift up to 10% either way

        # Ensure offset stays within 0-100 range
        adjusted_offset = max(0, min(100, base_offset + offset_shift))

        gradient_stops.append({
            "offset": adjusted_offset,
            "color": color_palette[i][0]
        })

    # Make sure we always have a stop at 0 and 100 for complete gradient
    if gradient_stops[0]["offset"] > 0:
        gradient_stops.insert(0, {"offset": 0, "color": gradient_stops[0]["color"]})
    if gradient_stops[-1]["offset"] < 100:
        gradient_stops.append({"offset": 100, "color": gradient_stops[-1]["color"]})

    # Calculate overall intensity for other parameters
    avg_intensity = np.mean(band_energies) * 100

    return waveform_data, gradient_stops, avg_intensity


# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dmc.styles.ALL])

# Define app layout using dash-mantine-components
app.layout = dmc.MantineProvider(
    theme={"colorScheme": "dark"},
    children=[
        dmc.Container(
            fluid=True,
            style={"height": "100vh", "padding": "20px"},
            children=[
                dmc.Title("Audio Visualizer", c="blue", ta='center', order=1, mb=15),

                dmc.Grid(
                    children=[
                        dmc.GridCol(
                            span=12,
                            children=[
                                dmc.Paper(
                                    p="md",
                                    shadow="sm",
                                    withBorder=True,
                                    children=[
                                        # Add device selection dropdown
                                        dmc.Select(
                                            id="device-select",
                                            label="Select Input Device",
                                            placeholder="Choose microphone",
                                            data=[{"value": str(d["index"]),
                                                   "label": f"{d['name']} {'(Default)' if d['default'] else ''}"}
                                                  for d in input_devices],
                                            value=str(
                                                [d["index"] for d in input_devices if d["default"]][0] if [d for d in
                                                                                                           input_devices
                                                                                                           if d[
                                                                                                               "default"]] else "0"),
                                            style={"marginBottom": "15px"}
                                        ),

                                        # Sensitivity slider
                                        dmc.Text("Input Sensitivity", fw="bold", mb=5),
                                        dmc.Slider(
                                            id="sensitivity-slider",
                                            min=1,
                                            max=50,
                                            step=1,
                                            value=20,
                                            marks=[
                                                {"value": 1, "label": "Low"},
                                                {"value": 20, "label": "Medium"},
                                                {"value": 50, "label": "High"}
                                            ],
                                            mb=15
                                        ),

                                        # Animation speed slider
                                        dmc.Text("Animation Speed", fw="bold", mb=5),
                                        dmc.Slider(
                                            id="animation-speed-slider",
                                            min=50,
                                            max=500,
                                            step=50,
                                            value=200,
                                            marks=[
                                                {"value": 50, "label": "Fast"},
                                                {"value": 200, "label": "Medium"},
                                                {"value": 500, "label": "Slow"}
                                            ],
                                            mb=15
                                        ),

                                        dmc.Group(
                                            children=[
                                                dmc.Button(
                                                    "Start Recording",
                                                    id="start-button",
                                                    color="green",
                                                    variant="filled",
                                                    mr=10,
                                                    size="lg"
                                                ),
                                                dmc.Button(
                                                    "Stop Recording",
                                                    id="stop-button",
                                                    color="red",
                                                    variant="filled",
                                                    size="lg"
                                                ),
                                            ]
                                        ),
                                        dmc.Space(h=10),
                                        dmc.Text(
                                            "Click Start Recording to begin visualizing your audio.",
                                            id="status-text",
                                            ta="center",
                                            c="dimmed"
                                        ),
                                        # Debug information display
                                        dmc.Space(h=10),
                                        dmc.Text(
                                            "Debug Info: Not recording",
                                            id="debug-text",
                                            ta="center",
                                            c="yellow",
                                            size="sm"
                                        ),
                                        # Signal level indicator
                                        dmc.Space(h=10),
                                        dmc.Text("Signal Level", ta="center", size="sm", mb=5),
                                        dmc.Progress(
                                            id="signal-level",
                                            value=0,
                                            color="green",
                                            size="lg",
                                            radius="xl"
                                        )
                                    ]
                                )
                            ]
                        ),

                        # Waveform visualization (now using dmc.LineChart)
                        dmc.GridCol(
                            span=6,
                            children=[
                                dmc.Paper(
                                    p="md",
                                    shadow="sm",
                                    withBorder=True,
                                    children=[
                                        dmc.Title("Waveform", order=3, ta="center", mb=10),
                                        html.Div(
                                            id="waveform-container",
                                            style={"height": "250px"}
                                        )
                                    ]
                                )
                            ]
                        ),

                        # Frequency visualization
                        dmc.GridCol(
                            span=6,
                            children=[
                                dmc.Paper(
                                    p="md",
                                    shadow="sm",
                                    withBorder=True,
                                    children=[
                                        dmc.Title("Frequency Spectrum", order=3, ta="center", mb=10),
                                        dcc.Graph(
                                            id="frequency-graph",
                                            style={"height": "250px"},
                                            config={"displayModeBar": False}
                                        )
                                    ]
                                )
                            ]
                        ),

                        # Replace Circular Visualization with RadarChart
                        dmc.GridCol(
                            span=12,
                            children=[
                                dmc.Paper(
                                    p="md",
                                    shadow="sm",
                                    withBorder=True,
                                    children=[
                                        dmc.Title("Frequency Bands Radar", order=3, ta="center", mb=10),
                                        html.Div(
                                            id="radar-chart-container",
                                            style={"height": "350px"}
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),

                # Hidden div for storing the recording state
                html.Div(id="recording-state", style={"display": "none"}, children="0"),

                # Store for current gain value
                dcc.Store(id="current-gain", data=20),

                # Store for animation duration
                dcc.Store(id="animation-duration", data=200),

                # Interval component for updating the graphs - more frequent updates
                dcc.Interval(
                    id="interval-component",
                    interval=50,  # Reduced to 50ms (20 updates per second) for smoother visualization
                    n_intervals=0
                ),

                # Debug interval
                dcc.Interval(
                    id="debug-interval",
                    interval=500,  # Update debug info every 500ms
                    n_intervals=0
                )
            ]
        )
    ]
)

# Stream and processing_thread objects
stream = None
processing_thread = None


# Update the gain value from slider
@callback(
    Output("current-gain", "data"),
    Input("sensitivity-slider", "value")
)
def update_gain(value):
    return value


# Update animation duration from slider
@callback(
    Output("animation-duration", "data"),
    Input("animation-speed-slider", "value")
)
def update_animation_duration(value):
    return value


# Callbacks
@callback(
    [Output("recording-state", "children"),
     Output("status-text", "children"),
     Output("start-button", "disabled"),
     Output("stop-button", "disabled")],
    [Input("start-button", "n_clicks"),
     Input("stop-button", "n_clicks")],
    [State("recording-state", "children"),
     State("device-select", "value")]
)
def toggle_recording(start_clicks, stop_clicks, recording_state, device_index):
    global stream, processing_thread, debug_info, recording_active, previous_band_values

    triggered_id = ctx.triggered_id if not None else "no-id"

    if triggered_id == "start-button" and recording_state == "0":
        # Clear the queue first
        while not q.empty():
            q.get()

        # Reset debug info and previous band values
        debug_info = {"last_processed": time.time(), "queue_size": 0, "max_amplitude": 0, "update_count": 0,
                      "rms_level": 0}
        for band in freq_bands:
            previous_band_values[band] = 0

        # Set recording state to active
        recording_active = True

        try:
            # Start recording with selected input device
            input_device_index = int(device_index) if device_index else None
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback
            )
            stream.start_stream()

            # Start processing thread
            processing_thread = threading.Thread(target=process_audio)
            processing_thread.daemon = True
            processing_thread.start()

            selected_device = next((d for d in input_devices if d["index"] == input_device_index), None)
            device_name = selected_device["name"] if selected_device else "Unknown"

            return "1", f"Recording from {device_name}... Make noise to see the visualizations!", True, False

        except Exception as e:
            recording_active = False
            return "0", f"Error: {str(e)}", False, True

    elif triggered_id == "stop-button" and recording_state == "1":
        # Set recording state to inactive
        recording_active = False

        # Stop recording
        if stream is not None:
            stream.stop_stream()
            stream.close()
            stream = None

        # Clean up processing thread
        if processing_thread is not None and processing_thread.is_alive():
            # Let the thread exit naturally
            # No need to join as it's a daemon thread
            processing_thread = None

        return "0", "Recording stopped.", False, True

    # Initial state or no change
    if recording_state == "0":
        return "0", "Click Start Recording to begin visualizing your audio.", False, True
    else:
        return "1", "Recording audio...", True, False


# Update debug text and signal level
@callback(
    [Output("debug-text", "children"),
     Output("signal-level", "value"),
     Output("signal-level", "color")],
    Input("debug-interval", "n_intervals"),
    State("recording-state", "children")
)
def update_debug(n_intervals, recording_state):
    if recording_state == "1":
        time_since_last = time.time() - debug_info["last_processed"]

        # Calculate signal level as percentage (capped at 100)
        rms_percentage = min(100, int((debug_info["rms_level"] / 10000) * 100))

        # Determine color based on signal level
        if rms_percentage < 10:
            color = "red"  # Very low signal
        elif rms_percentage < 30:
            color = "yellow"  # Low signal
        else:
            color = "green"  # Good signal

        return (
            f"Debug: Queue size: {debug_info['queue_size']}, " +
            f"Last processed: {time_since_last:.2f}s ago, " +
            f"Max amplitude: {debug_info['max_amplitude']}, " +
            f"RMS level: {debug_info['rms_level']:.1f}, " +
            f"Updates: {debug_info['update_count']}",
            rms_percentage,
            color
        )
    else:
        return "Debug Info: Not recording", 0, "gray"


# Update graphs based on audio data
@callback(
    [Output("waveform-container", "children"),
     Output("frequency-graph", "figure"),
     Output("radar-chart-container", "children")],
    [Input("interval-component", "n_intervals"),
     Input("current-gain", "data"),
     Input("animation-duration", "data")],
    State("recording-state", "children")
)
def update_graphs(n_intervals, gain_value, animation_duration, recording_state):
    global audio_data, fft_data

    # Update the gain value for the audio processing
    gain = float(gain_value)

    with data_lock:
        local_audio_data = audio_data.copy()
        local_fft_data = fft_data.copy()

    # Add small random noise to ensure updates even when signal is quiet
    noise_level = 0.0003  # Reduced noise level to make real signals more visible
    local_audio_data = local_audio_data + np.random.normal(0, noise_level, local_audio_data.shape)

    # Create LineChart waveform visualization
    waveform_data, gradient_stops, avg_intensity = audio_to_linechart_data(local_audio_data, local_fft_data)

    # Create LineChart component
    waveform_chart = dmc.LineChart(
        h=250,
        data=waveform_data,
        dataKey="time",
        type="gradient",
        gradientStops=gradient_stops,
        withTooltip=False,
        withDots=False,
        curveType="bump",
        withXAxis=False,
        fillOpacity=0.7,
        series=[
            {"name": "amplitude", "color": gradient_stops[1]["color"]}
        ],
        yAxisProps={"domain": [-1, 1]},
        lineProps={
            "isAnimationActive": True,
            "animationDuration": animation_duration,
            "animationEasing": "ease-in-out",
        },
        strokeWidth=3
    )

    # Create frequency visualization
    frequency_fig = go.Figure()

    # Add a minimal noise floor to make the graph more responsive, but keep it very low
    if recording_state == "1":
        noise_floor = np.random.uniform(0.0001, 0.0005, len(local_fft_data))
        local_fft_data = np.maximum(local_fft_data, noise_floor)

    # Calculate max FFT value for color scaling
    max_fft = np.max(local_fft_data)

    # Normalize FFT data for color mapping (0-1 range)
    normalized_fft = local_fft_data / max(max_fft, 0.05)  # Avoid division by zero/small values

    # Map each FFT value to a color in our palette
    bar_colors = []
    for value in normalized_fft:
        idx = min(int(value * len(color_palette)), len(color_palette) - 1)
        bar_colors.append(color_palette[idx][0])  # Use the fill color

    # Add frequency bars with dynamic colors
    frequency_fig.add_trace(go.Bar(
        x=np.arange(len(local_fft_data)) * (RATE / (2 * CHUNK)),  # Convert to actual frequency in Hz
        y=local_fft_data,
        marker=dict(
            color=bar_colors,  # Use our dynamic colors instead of a colorscale
            line=dict(width=0)  # No border for cleaner look
        )
    ))

    frequency_fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title="Frequency (Hz)",
            showticklabels=True,
            range=[0, 5000],  # Show up to 5kHz for better visibility
            tickfont=dict(color='rgba(255,255,255,0.7)')
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.2)',
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, max(0.05, np.max(local_fft_data) * 1.2)],  # Lower minimum range for better sensitivity
            showticklabels=False
        ),
        showlegend=False
    )

    # Create radar chart data from FFT data
    radar_data = fft_to_radar_data(local_fft_data)

    # Simplified approach: Create a single series with dynamic colors on update
    # Get current color based on overall intensity
    avg_intensity = np.mean([item["value"] for item in radar_data])
    color_idx = min(int(avg_intensity / 100 * len(color_palette)), len(color_palette) - 1)
    current_color = color_palette[color_idx][0]
    current_stroke = color_palette[color_idx][1]

    # Create RadarChart component with simpler configuration
    radar_chart = dmc.RadarChart(
        h=350,
        data=radar_data,
        dataKey="band",
        withPolarGrid=True,
        withPolarAngleAxis=True,
        withPolarRadiusAxis=True,
        radarChartProps={"cx": "50%", "cy": "50%", "outerRadius": "80%"},
        series=[
            {
                "name": "value",
                "fill": True,
                "fillOpacity": 0.6,
                "stroke": current_stroke,
                "strokeWidth": 2,
                "color": current_color
            }
        ],
        radarProps={
            "isAnimationActive": True,
            "animationBegin": 0,
            "animationDuration": animation_duration,
            "animationEasing": "ease-in-out"
        }
    )

    return waveform_chart, frequency_fig, radar_chart


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8888)