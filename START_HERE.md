# Start here

Install the project once, then start the camera setup:

```powershell
python -m pip install -e ".[evaluation,test]"
python start.py
```

The default setup uses no checkerboard. On the first run, the program captures
an empty field image, the car's appearance and front, and four measured floor
reference points. It saves that setup and starts tracking. Keep the camera
fixed afterward.

Later, the same command starts tracking immediately:

```powershell
python start.py
```

If the camera moves, use `--reset-background --reset-car --reset-field`. If
lighting or the background changes, use `--reset-background --reset-car`.

For the NTNU camera and controller, select the camera and UDP destination:

```powershell
python start.py --source 1 --host CONTROLLER_IP --port 5000
```

The visible red points in the tiled assignment images may be used as four
measured floor references; they are not proven to be the full field boundary. See
[the Taiwan handoff](docs/taiwan_handoff.md) before collecting results.
For the exact measurements, commands, and evaluation data to collect, use the
[Taiwan team checklist](TAIWAN_TEAM_CHECKLIST.md).

The no-target default calculates a planar homography but does not correct lens
distortion. The assignment also names `calibrateCamera`; if that method is
required by the examiner, a calibration target or suitable existing camera
calibration data is still needed.

