# Assignment comparison and NTNU setup handoff

Source: [the assignment brief](assignment.md), supplied by the student. Images 1–3
show a bright multicolour car in a cardboard tray; images 4–6 show a darker,
much smaller car on a tiled floor. The tiled example has four red tape points.
Their actual spacing and role are not specified in the assignment, so they
cannot be assumed to be the whole field boundary or car ID tags.

## Requirement check

| Assignment requirement | Current implementation | Physical check still needed |
|---|---|---|
| Oblique camera; lens calibration and image-to-world homography | The default workflow performs direct four-point homography with `cv2.findHomography`. `cv2.calibrateCamera` remains available as an optional tool but is not run without a calibration target. | Measure four floor points and evaluate edge error. Confirm with instructors whether the no-target workflow is acceptable for the brief. |
| Approximately 1.5 m by 2.5 m flat field | Field dimensions are entered in millimetres during corner setup | Measure the actual four reference points. The defaults are examples only. |
| Track and identify a toy car | Default mode is a prototype for **one** unmodified car. It combines an empty-field reference and car colour model; screenshot diagnostics below explain why the original colour-only method was inadequate. | Calibrate and tune on the actual car and raw camera video. Test whether its front and rear are distinguishable. |
| Multiple cars if used | Optional ArUco mode supports distinct tags on cars | Markerless colour mode does not identify multiple similar cars. Agree on car count before a multi-car demonstration. |
| 60 frames/s, 16.67 ms/frame target | Processing time and capture-loop FPS can be measured | No 60 FPS claim until measured on their laptop and webcam. Camera exposure and actual output FPS matter. |
| Position, heading, linear/angular velocity, image centre, relative microsecond timestamp | Implemented in `server.py`, `tracking.py`, and `protocol.py` | Check signs, heading direction, center offset, and units against physical measurements. |
| UTF-8 UDP, default port 5000, port selectable at CLI | Implemented; destination host and port are CLI options | Set the controller laptop IP and confirm receipt through the real network. |
| ROC curve | `tools/score_labeled_frames.py` scores held-out images against manually marked car centers; `toycar-evaluate detection` plots ROC | Collect positive, empty, and hard-negative frames on the actual field. |
| Stationary mapping mean/max position and orientation error | `toycar-evaluate mapping` calculates mean and max (also median and p95) | Record independently measured ground truth at several field positions and headings. |
| 2–5 page report, oral demonstration, NTNU honesty declaration | `docs/report.md` is a draft with unfilled measurement placeholders | Group must add actual data, evidence, names, and the declaration before Moodle submission. |

The assignment allows ArUco/AprilTag markers **outside** the field as an
alternative way to establish the field map. A square marker on the car is not
required by the brief. The current ArUco car detector is a separate optional
mode and would need a readable tag attached to each car.

## What the sample pictures reveal

The two scenes have different car appearance, background, camera geometry, and
apparent car size. In the tiled frames the car is only roughly 20–35 pixels
across. Default colour segmentation and a 9-pixel morphology kernel can erase
such a small target. The red points form a reference quadrilateral, but the car
in image 4 is visibly left of it; clipping detection to that quadrilateral
would miss the car.

A diagnostic trained an automatic foreground mask from the visible car in
image 2 and separately from image 4, then ran the default detector on the
matching scene's screenshots. Before the area-filter fix, it chose a large
background region in both scenes. After the fix, it still chose background in
the cardboard frames and returned no detection in the tiled frames. Those
screenshots include video-player borders and are not raw camera frames, so this
is a failure demonstration, not a fair final accuracy measurement. It motivated
adding an empty-field reference; that new mode still needs physical testing.

## Recommended physical setup

1. Mount the NTNU webcam firmly at the intended side angle. Fix its resolution,
   focus, and exposure as far as the camera allows. Keep at least four measured
   floor reference points visible. The red tape points can serve as manual
   homography references only if the team measures their physical coordinates.
2. Copy `config.color.example.yaml` to a team-specific YAML file. Set the actual
   camera index or stream URL, car name/ID, output host, and UDP port. Keep the
   field size as a measured value; 2500 × 1500 mm is only an approximate brief.
3. Leave `camera.calibration_file: null` for the no-checkerboard workflow.
   Lens distortion will not be corrected, so include image-edge test positions
   in the physical evaluation. The assignment names `calibrateCamera`; obtain
   instructor acceptance or suitable camera calibration data if that step is
   compulsory.
4. Run `python start.py --config team.yaml --source 1 --host CONTROLLER_IP`.
   The wizard captures an empty field, teaches the actual car appearance and
   front, and asks for four floor points. Set `field.reference_points_mm` for
   independently measured, nonrectangular point coordinates. If the camera
   changes, use `--reset-camera` to repeat all calibration steps.
5. Verify that the overlay follows the car across the complete field and its
   arrow points toward the front. Tune the colour threshold, minimum area,
   morphology kernel, and reference-area ratios in the YAML using recorded
   frames before a live run. Run `toycar-receiver --port 5000` on the
   controller laptop and verify UDP packets arrive. If the receiver is on the
   same laptop, use `--host 127.0.0.1`.
6. Run a headless performance trial with
   `python start.py --config team.yaml --headless --metrics-csv results/runtime.csv`.
   Analyze it with `toycar-evaluate runtime --csv results/runtime.csv`.
   The CSV includes mapped pose for detected frames, which helps build the
   stationary mapping dataset.

## Evaluation data to collect together

- **Detection:** save frames from the real camera. Create `labels.csv` with
  `image_path,present,center_u,center_v,radius_px`, where paths are relative to
  the CSV file and positive frames have manually marked car centers. Include many
  car-present and car-absent frames from locations across the field and several
  lighting conditions. Keep calibration frames out. Run the scorer and ROC
  commands in `README.md`.
- **Mapping:** place a stationary car at independently measured `(x,y,theta)`
  positions, including near each corner and the center. Let the filter settle,
  record repeated `x_mm,y_mm,theta_deg` readings, and combine each with its
  `true_x_mm,true_y_mm,true_theta_deg`. Run the mapping command in `README.md`.
- **Speed:** report the measured capture-loop FPS and processing-time percentiles
  at the actual camera resolution. The webcam's 60 FPS setting alone is not
  proof of 60 processed frames/s.
- **Demo:** record the live overlay and the UDP receiver while the car moves.
  Show a missing-car event, a heading turn, and movement at different field
  positions. The assignment calls for convincing oral-exam evidence.

## Known limits that affect the handoff

- The default colour mode supports one car. Distinct car IDs require distinct
  trained appearance models or readable physical tags; that work is outside
  the present markerless implementation.
- The visible car body is above the floor plane. An oblique camera introduces
  parallax, especially toward the far edge. Measure this in the mapping test.
- The actual car, camera feed, calibration images, ground truth, and controller
  endpoint are held by the NTNU team, so no physical accuracy or speed result
  can be asserted from this repository alone.
