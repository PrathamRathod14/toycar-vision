# Taiwan team: camera setup, test, and evidence checklist

This file is for the students who have the **real NTNU car, camera, and playing
field**. The software is prepared in this repository; its tracking accuracy and
60 FPS rate cannot be certified from the six assignment screenshots. Record
actual results rather than copying example values.

## 1. Prepare and measure the setup

- Mount one webcam firmly at the side angle used for the demonstration. Keep its
  focus, resolution, exposure, and lighting fixed. One laptop and one webcam
  satisfy the platform choice in the assignment.
- Mark at least four visible points on the **floor plane**, preferably spread
  across the full area. Measure their world `(x,y)` positions in **millimetres**
  from one chosen origin. The red points in the assignment images are usable
  only if their real coordinates are measured. The car can move outside their
  quadrilateral, where extrapolation error should also be tested.
- Measure the real usable field dimensions. `2500 × 1500 mm` in the config is an
  approximate assignment value, not a measurement of your room.
- No checkerboard or screen pattern is needed in the default workflow. The
  measured floor points establish millimetres through a planar homography.
  Lens distortion remains uncorrected; include edge positions in the accuracy
  test and report this limitation. The brief also names `calibrateCamera`, so
  ask the instructors whether direct homography alone is acceptable if a
  calibration target will not be used.
- Decide which physical end is the car's front. Note the car name/ID and whether
  a second car will be demonstrated. The default markerless mode supports one
  car; multiple similar cars need a different identification strategy.

## 2. Install and configure

On the laptop connected to the camera:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[evaluation,test]"
```

Edit `config.yaml`: set `camera.source`, the target `width`, `height`, and `fps`,
the car's `car_id` and `name`, and `network.host` to the controller laptop's IP
address. Leave `camera.calibration_file: null`. Keep
`detector.background_file`, `detector.model_file`, and
`field.homography_file` set to their default paths.

If the four floor points are not exact corners of a measured rectangle, add
their independently measured coordinates in click order:

```yaml
field:
  reference_points_mm:
    - [0, 0]          # top-left point in your chosen world coordinates
    - [MEASURED_X, MEASURED_Y]   # top-right
    - [MEASURED_X, MEASURED_Y]   # bottom-right
    - [MEASURED_X, MEASURED_Y]   # bottom-left
```

Replace every `MEASURED_*` value with a number. If you leave this field out,
the setup wizard asks for a rectangle width and height and assigns its corners
to `(0,0)`, `(width,0)`, `(width,height)`, `(0,height)`.

## 3. Run first-time calibration and tracking

Start the UDP receiver on the controller laptop:

```powershell
toycar-receiver --port 5000
```

On the camera laptop, run (replace the capitalized values):

```powershell
python start.py --source CAMERA_INDEX --host CONTROLLER_IP --port 5000
```

On the first run, the program guides you through these steps in order:

1. Remove the car and people from the field. Press **Space** to save an empty
   field image. Keep lighting and camera fixed afterward.
2. Place the car on the field. Press **Space** to freeze, draw a close box around
   it, then click its **front**. The program stores the car appearance model.
   If it warns that front and rear colours are too similar, test heading carefully;
   a visible car tag may be needed to identify direction reliably.
3. Click the four measured floor points **top-left, top-right, bottom-right,
   bottom-left** and press Enter. Confirm that their ordering matches the
   `reference_points_mm` coordinates in your YAML.
4. Watch the overlay while moving the car across the entire field. The arrow
   must point to the real front, and the UDP receiver must show changing pose
   records. Move the car out of view and confirm `-1000.000,-1000.000` is sent.

For later runs, use the same `python start.py --source ... --host ...` command.
If the camera moves, use `--reset-background --reset-car --reset-field`. If
lighting or background changes, use `--reset-background --reset-car`.

If the overlay fails, save raw camera frames and adjust `background_threshold`,
`min_color_strength`, `min_area_fraction`, `min_reference_area_ratio`, and
`max_reference_area_ratio` in `config.yaml`. The defaults are starting values,
not measured optimums. Do not use the assignment's video-player screenshots as
camera calibration or test frames.

## 4. Collect the required evaluation results

**Detection ROC:** Save held-out raw frames with the car at varied locations,
angles, speeds, and lighting, plus empty-field frames and hard negatives.
Calibration images must be excluded. Create `labels.csv` alongside the images:

```powershell
python tools/collect_frames.py --source CAMERA_INDEX `
  --output-dir captures/heldout
```

Press **Space** to save each useful frame and **Q** to finish. Alternatively,
extract every 30th frame from a raw recording with `--source recording.mp4
--every-n 30`. The tool creates a blank `labels.csv`; fill in its labels and
centers by inspecting the saved images. The required columns are:

```csv
image_path,present,center_u,center_v,radius_px
frame_0001.png,1,612,381,20
frame_0002.png,0,,,
```

For positive frames, manually mark the actual car center in image pixels and a
reasonable match radius. Replace the example coordinates with actual labels.
Then run:

```powershell
python tools/score_labeled_frames.py --config config.yaml `
  --labels captures/heldout/labels.csv --output results/scored_frames.csv
toycar-evaluate detection --csv results/scored_frames.csv `
  --output results/roc_curve.png
```

Report sample counts, ROC/AUC, and the selected operating threshold. Inspect
false positives and misses; a visually wrong location must count as a miss.

**Stationary mapping error:** Place the car at several independently measured
positions and orientations, including near each edge and outside the reference
quadrilateral if that is part of the play area. Record several readings after
the filter settles. Run with `--metrics-csv results/runtime.csv`; this includes
reported `x_mm,y_mm,theta_deg`. Pair each reading with measured ground truth in
a CSV containing:

```text
measured_x_mm,measured_y_mm,measured_theta_deg,true_x_mm,true_y_mm,true_theta_deg
```

Calculate the required mean and maximum errors:

```powershell
toycar-evaluate mapping --csv mapping_measurements.csv `
  --output results/mapping_error.json
```

**Speed:** Run a representative moving-car trial at the final camera resolution:

```powershell
python start.py --source CAMERA_INDEX --headless --metrics-csv results/runtime.csv `
  --host CONTROLLER_IP --port 5000
toycar-evaluate runtime --csv results/runtime.csv `
  --output results/runtime.json
```

Report both capture-loop FPS and the median/95th-percentile processing time.
The assignment's target is 60 processed frames/s and 16.67 ms per frame. A
camera setting of 60 FPS alone does not demonstrate this.

## 5. Submit and demonstrate

- Record the live camera overlay and receiving UDP packets while the car moves.
  Show a turn, different parts of the field, and a missing-car event.
- Fill the actual values, group names, difficulties, plots, and conclusions into
  `docs/report.md`, and produce a **2–5 page** final report.
- Include the filled NTNU honesty declaration and submit as the joint group in
  Open Moodle. Prepare to explain the detector, calibration, errors, and timing
  in the oral exam.
- Share the actual calibration files and measurement data with the group. The
  `calibration/`, `captures/`, and `results/` folders are ignored by Git because
  they are specific to your physical setup.
