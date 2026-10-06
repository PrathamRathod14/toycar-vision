# Start here

Install the project once, then start the camera setup:

```powershell
python -m pip install -e ".[evaluation,test]"
python start.py
```

Before the first run, create `chessboard.png` with
`python tools/generate_chessboard.py` and display it on a flat phone, tablet, or
second monitor without stretching the squares. The program captures views of
that screen for lens calibration, then an empty field image, the car's
appearance and front, and four measured floor reference points. No printing or
screen-square measurement is needed. It saves the setup and starts tracking.

Later, the same command starts tracking immediately:

```powershell
python start.py
```

If the camera changes, run `python start.py --reset-camera`. If lighting or the
background changes, use `--reset-background --reset-car`.

For the NTNU camera and controller, select the camera and UDP destination:

```powershell
python start.py --source 1 --host CONTROLLER_IP --port 5000
```

The visible red points in the tiled assignment images may be used as four
measured floor references; they are not proven to be the full field boundary. See
[the Taiwan handoff](docs/taiwan_handoff.md) before collecting results.
For the exact measurements, commands, and evaluation data to collect, use the
[Taiwan team checklist](TAIWAN_TEAM_CHECKLIST.md).

