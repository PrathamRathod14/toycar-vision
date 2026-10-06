from toycar_vision.models import Telemetry
from toycar_vision.protocol import serialize


def test_assignment_wire_format_is_utf8_and_newline_terminated():
    record = Telemetry(1000023, 10, "Red Racer", 123.0, 230.0, 90.0,
                       1000.0, -300.0, 0.6, 237, 1024)
    assert serialize(record) == (
        b'1000023:"Red Racer",123.000,230.000,90.000,'
        b'1000.000,-300.000,0.600,237,1024\n'
    )

