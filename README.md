# Real time video effects(Windows GPU Engine)

Real-time video shader engine for Windows. Features multi-threaded camera capture, asyncroness MediaPipe hand landmark tracking, and OpenGL fragment shaders with dynamic coordinate masking.

---

## Key Highlights

- **GPU Shader Pipeline**: Effects execute on the GPU via GLSL fragment shaders.
- **Camera Pipeline**: Uses DirectShow (`cv2.CAP_DSHOW`) or Media Foundation with a single-frame buffer policy (`BUFFERSIZE = 1`) to eliminate driver lag.
- **Direct GPU Texture Upload**: Uploads camera frames directly to GPU memory via `GL_BGR` texture transfers without CPU color conversion.
- **Asyncroness Hand Landmark Detection**: Runs MediaPipe hand tracking on a background thread at 320x240 with adaptive EMA smoothing. The rendering loop runs decoupled at 60-144+ FPS without waiting on inference.
- **In-Shader Coordinate Masking**: The bounding box formed between thumb and index fingers of both hands is evaluated per-fragment in the shader.
- **Hot-Reloadable Shaders**: Save any `.frag` file in the `shaders/` directory to hot-reload immediatly without restarting.

---

## Quick Start

### 1. Requirements & Installation
The app requires Python 3.10+ (tested on Python 3.14):
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Double-click `run.bat` or run:
```bash
python main.py
```

To run with a synthetic mock video pattern (like when no physical webcam is connected):
```bash
python main.py --mock
```

To benchmark uncapped GPU frame rates:
```bash
python main.py --no-vsync
```

---

## Hand Tracking & The Pinch Distortion Box

### Flagship Effect: Hand Pinch Portal (`01_pinch_distortion.frag`)
1. Raise both hands to the camera.
2. Extend your **thumb** and **index finger** on both hands.
3. The application computes the bounding box formed between all 4 finger tips:
   - Hand 1: Thumb tip (Landmark 4) & Index tip (Landmark 8)
   - Hand 2: Thumb tip (Landmark 4) & Index tip (Landmark 8)
4. Visual reticle dynamiclly adapts: connecting the 4 fingertips in a quadrilatiral when tracking two hands, and framing a clean box when tracking a single hand (2 fingers).
5. Everything inside the box undergoes refractive fluid distortion, ripple waves, and chromatic aberration.
6. If only 1 hand is visible than pinching your thumb and index finger creates an interactive pinch portal.
7. Press `M` to switch mask modes:
   - **Mode 0 (Default)**: Inside Hand Box only (clear outside).
   - **Mode 1**: Inverted Hand Box (distorts background, keeps hand interior clear).
   - **Mode 2**: Full Screen (shader effect applies across the entire camera feed).

---

## Interactive Hand Gestures

Control the application handsfree using hand gestures:

| Gesture | Detection Logic | Action |
|---|---|---|
| **Finger Snap** | Thumb & middle finger compression followed by rapid release | **Cycle to Next Shader** (`next_shader()`) + HUD toast |
| **Peace / V-Sign** | Index & middle fingers extended, ring & pinky folded | **Cycle Mask Mode** (Box $\rightarrow$ Inverted $\rightarrow$ Fullscreen) |
| **Thumbs Up** | Fist closed, thumb extended pointing upwards ($-\hat{y}$) | **Increase Shader Intensity** smoothly (rate-limited) |
| **Thumbs Down** | Fist closed, thumb extended pointing downwards ($+\hat{y}$) | **Decrease Shader Intensity** smoothly (rate-limited) |
| **Closed Fist** | All 5 fingers curled into palm with thumb folded | **Toggle Hand Box Reticle** visual (`Key B`) |
| **Open Palm** | All 5 fingers extended spread outward | **Reset Intensity to 1.00** |
| **Rock On / Horns** | Index & pinky extended, middle & ring curled | **Toggle Tracking & Gesture Boost Mode** (`Key V`) |

One-shot gestures use per-hand state isolation, hystreresis debouncing, and cooldown to prevent accidental multi-firing, while continuous gestures (thumbs up/down) adjust intensity smoothly.

> [!TIP]
> **Gesture Isolation & Boost Mode**: Hand tracking uses temporal identity persistence and adaptive micro-jitter filtering. Press `V` or run with `--boost` to enable Boost Mode for higher resolution tracking (480p), snappier smoothing, and accelerated gesture response rates.
> You can toggle all recognition gestures on/off at any time using the **on-screen button** or pressing `G`. When disabled, only the main Hand Box gesture remains active.

---

## Keyboard & Mouse Controls

| Key / Input | Action |
|---|---|
| `1` - `9`, `0` | Direct select shader 1 through 10 |
| `Space` / `Tab` | Cycle to next shader |
| `Shift + Tab` / `Backspace` | Cycle to previous shader |
| `M` | Cycle Mask Mode: Hand Box $\rightarrow$ Inverted $\rightarrow$ Fullscreen |
| `B` | Toggle visual reticle (4-finger connection / 2-finger box) & finger dots |
| `V` | Toggle Tracking & Gesture Boost Mode |
| `G` / Click UI Button | Toggle Extra Gestures (disable all other gestures except the main Hand Box) |
| `X` | Toggle horizontle camera mirroring |
| `Up` / `Down` | Increase / Decrease effect intensity |
| `H` | Toggle on-screen HUD overlay |
| `F` | Toggle Fullscreen / Windowed mode |
| `R` | Force hot-reload all shaders from disk |
| `O` | Open `shaders/` folder in Windows File Explorer |
| `Esc` / `Q` | Exit application |

---

## Extensible Shader Framework: Create Your Own Shaders

The engine scans the `shaders/` directory on startup and automatically compiles any `.frag` file.

### How to Add a New Shader
1. Copy `shaders/template_custom_shader.frag.example` to `shaders/11_my_shader.frag`.
2. Open it in your favorite text editor (Notepad, VS Code, etc.).
3. Edit the fragment code.
4. Save the file. The app detects file modifications and **hot-reloads the shader instantly**!
5. If you introduce a GLSL syntax error, the engine displays the error log on screen and continues running the previous working shader without crashing.

### Available Uniforms

Every shader has access to these uniforms:

```glsl
uniform sampler2D u_camera;       // Webcam video stream texture
uniform vec2      u_resolution;   // Window resolution in pixels (e.g. 1280.0, 720.0)
uniform float     u_time;         // Elapsed time in seconds
uniform vec4      u_hand_box;     // vec4(min_x, min_y, max_x, max_y) in UV coordinates [0.0, 1.0]
uniform float     u_has_box;      // 1.0 if hands detected, 0.0 otherwise
uniform int       u_mask_mode;    // 0: Inside Box, 1: Outside Box, 2: Fullscreen
uniform float     u_intensity;    // User-tunable intensity slider (Up/Down arrow keys)
uniform vec2      u_mouse;        // Mouse cursor normalized coordinate [0.0, 1.0]
uniform vec2      u_thumb1;       // Hand 1 thumb tip UV
uniform vec2      u_index1;       // Hand 1 index tip UV
uniform vec2      u_thumb2;       // Hand 2 thumb tip UV
uniform vec2      u_index2;       // Hand 2 index tip UV
```

### Shared Helper Functions (`#include "common.glsl"`)

- `bool is_inside_box(vec2 uv, vec4 box)`
- `float get_box_mask(vec2 uv, vec4 box, float feather)`: Smooth fethered box mask.
- `float compute_mask(vec2 uv, vec4 box, float has_box, int mode, float feather)`: Evaluates mask weight based on mode and hand detection.
- `mat2 rotate2d(float angle)`
- `float hash21(vec2 p)`: Pseudo-random noise hash.

---

## Built-in Shader Catalog

1. **`01_pinch_distortion.frag`**: Refractive lens distortion, ripples, and chromatic aberration inside the hand box.
2. **`02_cyber_glitch.frag`**: RGB channel splitting, scanlines, and noise.
3. **`03_vortex_whirl.frag`**: Vortex swirl centered between fingers.
4. **`04_thermal_vision.frag`**: Infrared thermal false-color mapping.
5. **`05_neon_edges.frag`**: Sobel edge detection with cycling rainbow aura.
6. **`06_water_ripple.frag`**: Water surface refraction with wave interference.
7. **`07_pixelate_mosaic.frag`**: Retro mosaic pixelation with phosphor sub-pixel pattern.
8. **`08_kaleidoscope.frag`**: Radial kaleidoscopic reflections with rotational symmetry.
9. **`09_ascii_matrix.frag`**: Falling digital code matrix glyphs with luminance character mapping.
10. **`10_magnifying_glass.frag`**: Spherical magnifying lens with chromatic aberration.
