# Toy Car Global Vision

An OpenCV prototype intended to track one unmodified coloured toy car, map
its image coordinates onto a measured field (approximately 2.5 m x 1.5 m),
estimates motion, and sends UTF-8 telemetry over UDP. It accepts an external
webcam, a camera stream, or a recorded video as input.
An older ArUco detector remains available as an optional alternative.

**Status:** The assignment screenshots expose failures in the current colour
detector: it can choose background instead of the car, or miss the small car on
tiles. The code and synthetic UDP pipeline run, but tracking accuracy and 60 FPS
on the NTNU camera have not been established. Use raw camera footage to tune and
evaluate the detector before a live controller demonstration.

For the assignment comparison and a checklist for the NTNU camera and car,
see [Taiwan handoff](docs/taiwan_handoff.md).

## What is included

- camera distortion calibration using `cv2.calibrateCamera`;
- manually calibrated image-to-field homography;
- markerless colour-based car position and heading detection;
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
Copy-Item config.color.example.yaml config.yaml
```

Use a plain floor or a large plain sheet, keep the laptop camera fixed, and use
bright, even lighting. The current colour mode supports one car.

For normal use, one command performs first-time setup and starts the system:

```powershell
python start.py
```

The first run guides you through selecting the car and field corners. Later runs
reuse the saved setup and start immediately. Nothing needs to be attached to the
car. Do not move the camera after field setup.

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
and angular velocity is degrees/s. `u,v` are image pixels. A configured but
currently invisible car is emitted as `-1000,-1000` with image coordinates
`-1,-1`; set `network.emit_missing: false` to omit missing cars.

## Calibration

### 1. Camera distortion

Take at least 10-20 sharp chessboard photographs across the entire image at
different angles. The `--columns` and `--rows` values are **inner corners**, not
squares.

```powershell
python tools/calibrate_camera.py "captures/*.jpg" --columns 9 --rows 6 `
  --square-mm 25 --output calibration/camera.yaml
```

Then set `camera.calibration_file: calibration/camera.yaml`. Calibration must be
performed at the same resolution and focus setting used during tracking.

### 2. Image-to-field mapping

Measure the rectangular playing area and keep the camera fixed. Generate the
homography by clicking its four corners:

```powershell
python tools/capture_homography.py --source 0 --width-mm 2500 `
  --height-mm 1500 --camera-calibration calibration/camera.yaml `
  --output calibration/homography.yaml
```

Click top-left, top-right, bottom-right, bottom-left, then press Enter. Supply
the same camera-calibration file used by the server so the clicked and runtime
image coordinates match.

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
background objects. Label them in a CSV with `image_path,present` (`present` is
0 or 1). Do not include frames used to calibrate the colour model. Then run:

```powershell
python tools/score_labeled_frames.py --config config.yaml `
  --labels labels.csv --output results/scored_frames.csv
toycar-evaluate detection --csv results/scored_frames.csv `
  --output results/roc_curve.png
```

The scorer disables only the final quality cutoff, so the ROC sweep can test
all quality thresholds. It keeps the configured colour segmentation settings.
Include hard negatives with colours similar to the car rather than using only
empty frames.

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
