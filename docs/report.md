# Global Vision for a Toy Race Car

**Course:** Assignment 1 - Toy Race Car  
**Group members:** _Add names and universities_  
**Date:** _Add submission date_

> Replace every bracketed placeholder with measurements from the real setup.
> Do not submit example values as experimental evidence.

## 1. Approach and design rationale

We implemented a global vision server that tracks one unmodified coloured toy
car on a measured planar field of approximately 2500 mm by 1500 mm. A webcam
observes the field from an oblique angle. The car is detected using a fixed
empty-field reference and an HSV colour model learned interactively from one
camera frame. No marker or physical modification of the car is required.

We selected colour detection because the car in the assignment examples has
distinctive coloured body regions and the assignment explicitly allows this approach. It is
computationally cheaper than a CNN and does not require a large labelled training
dataset. During calibration, the user selects the car and clicks its front. The
system stores a complete colour histogram plus separate front- and rear-half
histograms. At runtime, foreground difference from the empty field proposes
car regions, while colour similarity and size checks reject distractions.
Principal-component analysis gives its long axis, while the two colour
signatures resolve the 180-degree direction ambiguity. The limitation is that
lighting changes and similarly coloured background objects can cause errors.

The implementation separates camera calibration, colour detection, planar
mapping, temporal tracking, UDP serialization, and evaluation. The main loop
records a monotonic timestamp, captures and optionally undistorts a frame,
detects the car, maps its center and heading to field coordinates, filters the
measurements, and sends a UTF-8 UDP datagram.

## 2. Calibration and coordinate mapping

Lens distortion is estimated from several views of a checkerboard displayed on
a flat screen using `cv2.calibrateCamera`. We captured [N] images at the
tracking resolution while moving and tilting the screen across the fixed
camera's image. The board uses one arbitrary unit per square because only lens
intrinsics and distortion are retained; the floor measurements establish the
millimetre scale. The resulting RMS
reprojection error was [VALUE] pixels. Runtime frames are corrected with
precomputed undistortion maps.

Because the playing surface is planar, a 3 by 3 homography maps undistorted image
coordinates `(u,v)` to field coordinates `(x,y)`. With the camera fixed, the user
clicks four independently measured floor reference points. `cv2.findHomography`
then calculates the image-to-world transform. Physical output uses millimetres. Zero
degrees is the positive field x-axis, and angles increase toward positive y.

The visible car body lies above the calibrated floor plane, so an angled camera
can introduce parallax. The mapping-accuracy experiment must measure this
effect rather than assuming the transform is exact.

## 3. Tracking, velocity, and communication

Raw detected centers and headings contain jitter. We apply exponential low-pass
filters whose coefficients depend on elapsed time. Velocity is calculated from
the filtered position and then filtered again. Heading differences are wrapped
to the shortest interval from -180 to 180 degrees, preventing a false angular
velocity spike when orientation crosses 0/360 degrees. A track is reinitialized
after a long observation gap.

Each output record contains a relative timestamp in microseconds, car ID,
position, heading, linear velocity, angular velocity, and image coordinates. It
is sent as one UTF-8 UDP datagram. The destination and port can be configured in
YAML or overridden on the command line. When the car is not visible, its world
position is reported as `-1000,-1000`.

At 60 frames/s the frame budget is 16.67 ms. Colour backprojection is performed
once per frame, camera buffering is minimized, and headless mode removes display
overhead. On [COMPUTER/CPU], at [RESOLUTION], the system processed [N] frames at
[MEAN] FPS. Median, 95th-percentile, and maximum processing times were [VALUES]
ms. These values were measured rather than inferred from the requested camera
frame rate.

## 4. Evaluation

### 4.1 Detection ROC

We recorded a held-out dataset of [N] labelled frames: [P] frames containing the
car and [Q] negative frames. The negatives included an empty field and
[SIMILARLY COLOURED HARD NEGATIVES]. Calibration frames were excluded. The
detector confidence combines colour-backprojection strength, foreground
difference, and contour solidity. A detection must also fall near the manually
marked car center. Sweeping the confidence threshold produced Figure 1.

_Insert `results/roc_curve.png` as Figure 1._

The area under the curve was [AUC]. At the selected threshold [T], true-positive
rate was [TPR] and false-positive rate was [FPR]. We also tested changes in
lighting because colour models are sensitive to exposure and white balance.

### 4.2 Mapping accuracy

We placed the stationary car at [N] independently measured poses across the
field. At every pose, [M] readings were collected after the filter settled.
Ground truth was measured using [METHOD] with estimated uncertainty [VALUE].
Euclidean position error was calculated as

`sqrt((x_measured-x_true)^2 + (y_measured-y_true)^2)`,

and orientation error used the smallest circular angle difference.

| Metric | Mean | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| Position error (mm) | [ ] | [ ] | [ ] | [ ] |
| Orientation error (degrees) | [ ] | [ ] | [ ] | [ ] |

## 5. Problems, limitations, and improvements

The main practical issues were [REPORT ACTUAL ISSUES: lighting, reflections,
motion blur, background colours, camera FPS, or calibration placement]. Bright,
diffuse lighting and a plain floor improved segmentation. A short exposure
reduced motion blur but required more light.

The current colour mode identifies one car. Multiple cars with similar colours
would require separate distinguishable appearances or a learned detector. A
single camera also cannot observe an occluded car. Future work could use two
calibrated cameras, adaptive colour models, or a small CNN trained on images from
the actual field.

In conclusion, the markerless design is intended to track the available
unmodified car and produce mapped pose, motion, and configurable UDP output.
The final submission must include the actual labelled data, plots, timing
measurements, and recorded demonstration from the NTNU setup so the reported
results can be checked.

