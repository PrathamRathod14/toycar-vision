# Toy Car Global Vision

An OpenCV prototype intended to track one unmodified coloured toy car, map
its image coordinates onto a measured field (approximately 2.5 m x 1.5 m),
estimate motion, and send UTF-8 telemetry over UDP. It accepts an external
webcam, a camera stream, or a recorded video as input.
An older ArUco detector remains available as an optional alternative.

**Status:** The assignment screenshots exposed failures in the original
colour-only detector. The default setup now combines a saved empty-field image
with a calibrated car-colour model and rejects implausible position jumps. The
code and synthetic UDP pipeline run, but tracking accuracy and 60 FPS on the
NTNU camera have not been established. Use raw camera footage to tune and
evaluate it before a live controller demonstration.

The physical setup, measurement, and submission steps are in
[Taiwan team checklist](TAIWAN_TEAM_CHECKLIST.md). The requirement comparison is
in [Taiwan handoff](docs/taiwan_handoff.md).

## What is included

- camera distortion calibration using `cv2.calibrateCamera`;
- manually calibrated image-to-field homography;
- markerless empty-field and colour-based car position and heading detection;
- time-aware filtering for position, velocity, and angular velocity;
- configurable UDP host/port and the assignment's text wire format;
- live overlay, headless mode, and runtime metrics capture;
- ROC and physical mapping-error evaluation utilities;
- unit tests and a report draft that does not invent experimental results.

## Quick start

Use Python 3.10 or newer. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[evaluation,test]"
```

Use a plain floor or a large plain sheet, keep the laptop camera fixed, and use
bright, even lighting. The current colour mode supports one car.

For normal use, one command performs first-time setup and starts the system:

```powershell
python start.py
```

The first run captures varied views of `chessboard.png` shown on a flat phone,
tablet, or second monitor, then an empty field, the car's appearance, and four
measured floor reference points. No printing or chessboard-square measurement
is needed: the lens calibration uses an arbitrary square unit, and the measured
floor points establish millimetres. Later runs reuse calibration and start
immediately. Nothing needs to be attached to the car. Do not move the camera
after setup.

Choose the actual external camera with `python start.py --source 1` (or another
camera index). For a recording, use `--source path/to/video.mp4`. To send packets
to the controller computer, use `--host CONTROLLER_IP --port 5000`.

Start a receiver in one terminal:

```powershell
toycar-receiver --port 5000
```

Start the vision server in another:

```powershell
toycar-vision --config config.yaml --port 5000
```

The port override meets the command-line configurability requirement. Press
`q` in the preview to stop. For a remote controller, set `network.host` to the
controller computer's IP address and allow the selected UDP port through its
firewall.

Example packet:

```text
1000023:"Red Racer",123.000,230.000,90.000,1000.000,-300.000,0.600,237,1024
```

The timestamp is microseconds since server start. Position is millimetres,
heading is degrees in the field coordinate system, linear velocity is mm/s,
and angular velocity is degrees/s. The quoted car name is the wire-format ID
used by the assignment example; the numeric `car_id` stays internal. `u,v` are
image pixels. A configured but
currently invisible car is emitted as `-1000,-1000` with image coordinates
`-1,-1`; set `network.emit_missing: false` to omit missing cars.

## Calibration

### 1. Camera distortion

The first-run wizard collects 12 sharp views of a checkerboard displayed on a
flat screen at different positions and angles, then calls
`cv2.calibrateCamera`. The camera stays fixed while the screen moves. The
`--columns` and `--rows` values are **inner corners**, not squares. Create the
image with:

```powershell
python tools/generate_chessboard.py --output chessboard.png
```

Display the entire image without stretching it; avoid reflections and use a
screen large enough for the camera to resolve its corners. For manual
calibration:

```powershell
python tools/calibrate_camera.py "captures/*.jpg" --columns 9 --rows 6 `
  --square-size 1 --output calibration/camera.yaml
```

The square size can be an arbitrary positive unit because this project retains
only the lens intrinsics and distortion, not the board's metric pose. The
measured floor points define the output's millimetre scale. The default config
already points to `calibration/camera.yaml`. Calibration must use the same
resolution and focus setting as tracking.

### 2. Image-to-field mapping

Measure four floor reference points and keep the camera fixed. If they are
the corners of a true rectangle, generate the homography with:

```powershell
python tools/capture_homography.py --source 0 --width-mm 2500 `
  --height-mm 1500 --camera-calibration calibration/camera.yaml `
  --output calibration/homography.yaml
```

Click top-left, top-right, bottom-right, bottom-left, then press Enter. Supply
the same camera-calibration file used by the server so the clicked and runtime
image coordinates match. For four arbitrary measured positions, set
`field.reference_points_mm` in the config or use `--world-points-mm` with eight
measured coordinates instead of width and height.

### 3. Heading convention

During colour calibration, clicking the car's front teaches the detector the
different colour patterns of its front and rear halves. The reported angle is
`atan2(y, x)` after mapping into field coordinates: 0 degrees is +x and angles
increase toward +y. Verify the direction before collecting evaluation data.

## Performance and data collection

Run without display overhead and log processing time:

```powershell
toycar-vision --config config.yaml --headless --metrics-csv results/runtime.csv
toycar-evaluate runtime --csv results/runtime.csv --output results/runtime.json
```

The console reports average end-to-end capture-loop FPS and last-frame vision
processing time. The target is at most 16.67 ms processing per frame, but actual
capture rate also depends on webcam exposure, driver, USB bandwidth, resolution,
and lighting. Use bright, diffuse light and a short exposure if motion blur
prevents stable colour detection.

The metrics CSV records detector output, not ground truth. For a valid ROC test,
save held-out camera frames containing the car, an empty field, and difficult
background objects. Label `image_path,present,center_u,center_v,radius_px` in a
CSV; positives need a manually marked center and match radius. Do not include
calibration frames. Then run:

```powershell
python tools/collect_frames.py --source 0 --output-dir captures/heldout
python tools/score_labeled_frames.py --config config.yaml `
  --labels captures/heldout/labels.csv --output results/scored_frames.csv
toycar-evaluate detection --csv results/scored_frames.csv `
  --output results/roc_curve.png
```

The scorer disables only the final quality cutoff, so the ROC sweep can test
all quality thresholds. It counts a detection at the wrong image location as a
miss and false match. Include hard negatives with colours similar to the car.

For mapping accuracy, place a stationary car at multiple independently measured
locations and orientations, collect repeated readings with `--metrics-csv`,
and add the independently measured true values to a CSV with:

```text
measured_x_mm,measured_y_mm,measured_theta_deg,true_x_mm,true_y_mm,true_theta_deg
```

Then run:

```powershell
toycar-evaluate mapping --csv mapping_measurements.csv `
  --output results/mapping_error.json
```

It reports mean, median, 95th percentile, and maximum Euclidean position error,
plus circular orientation error. Files in `examples/` only demonstrate the
format and must not be submitted as measured results.

## Tests

```powershell
python -m pytest -q
python -m compileall -q toycar_vision tools
```

## Important practical limits

- A single angled camera cannot observe a car hidden by another object. Multiple
  synchronized cameras would require observation fusion beyond this version.
- A planar homography represents the floor plane. Because the visible car center
  is above the floor, an angled camera introduces parallax that must be measured.
- Colour detection is sensitive to lighting and similarly coloured objects; use
  a plain playing surface and recalibrate after major lighting changes.
- UDP is intentionally low latency and does not guarantee delivery. A controller
  should reject stale timestamps and stop a car when updates time out.
- The preview is useful for setup but headless mode is the honest performance
  measurement.
