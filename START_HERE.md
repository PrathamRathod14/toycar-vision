# Start here

Install the project once, then start the camera setup:

```powershell
python -m pip install -e ".[evaluation,test]"
python start.py
```

On the first run, the program will ask you to select the car, click its front,
enter the measured playing-area size, and click the four field corners. It saves
that setup automatically and starts tracking.

Later, the same command starts tracking immediately:

```powershell
python start.py
```

If the camera or lighting changes, run `python start.py --reset-car
--reset-field` to repeat the setup.

For the NTNU camera and controller, select the camera and UDP destination:

```powershell
python start.py --source 1 --host CONTROLLER_IP --port 5000
```

The visible red points in the tiled assignment images may be used as four
measured floor references; they are not proven to be the full field boundary. See
[the Taiwan handoff](docs/taiwan_handoff.md) before collecting results.

