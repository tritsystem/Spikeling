#!/usr/bin/env python
"""
wireless_glasses.py — PC-side integration for the fully-wireless glasses
design: a XIAO ESP32S3 Sense (camera, WiFi) + a XIAO ESP32C3 (OLED
display driver, WiFi) as independent wireless peripherals to THIS PC --
not a standalone brain. Claude's reasoning still happens here, on the PC;
the glasses are wireless camera + display peripherals, same as the wired
C922 was, just over WiFi instead of USB.

HONEST STATUS: UNTESTED. No physical ESP32S3/ESP32C3 hardware exists yet
-- this is the correct shape for when it arrives, same discipline as
core/hardware/emg_adapter.py's SerialEMGSource stub. Do not report this
as verified; it hasn't run against real hardware.

Requires the ESP32 side to run standard, well-established firmware:
- XIAO ESP32S3 Sense: Arduino's bundled "CameraWebServer" example sketch
  (ships with the ESP32 Arduino core, needs only WiFi credentials + board
  selection, no custom code) -- its default snapshot endpoint is
  http://<ip>/capture, returning one JPEG per request.
- XIAO ESP32C3: a small custom sketch (not yet written) running a tiny
  HTTP server with a POST /display endpoint that renders received text
  to the SSD1306 via the standard Adafruit_SSD1306 library.

Bluetooth mic+speaker (a real off-the-shelf BT mono earpiece) needs NO
code here at all -- once paired, Windows exposes it as a normal audio
input/output device, usable by the exact same sounddevice-based code
already used throughout this hardware layer (select it by device index,
same as the C922/QuadCast were).

    from wireless_glasses import capture_wireless_frame, send_to_glasses_display
"""
import urllib.request
import urllib.error

DEFAULT_TIMEOUT_S = 5.0


def capture_wireless_frame(esp32_cam_ip: str, out_path: str = "live_frame.jpg",
                            timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    """Fetches one real JPEG snapshot from the XIAO ESP32S3 Sense's
    CameraWebServer /capture endpoint over WiFi -- the wireless
    replacement for capture_frame.py's direct cv2.VideoCapture(0) call.
    Same output contract (a saved JPEG path) so nothing downstream
    (Claude's Read tool, electronics_assistant.py's vault logging) needs
    to change regardless of which camera captured the frame.

    UNTESTED -- no ESP32S3 hardware exists yet to verify this against."""
    url = f"http://{esp32_cam_ip}/capture"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach the wireless camera at {url}: {e}")
    if not data:
        raise RuntimeError(f"Camera at {url} returned an empty response.")
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def send_to_glasses_display(esp32_display_ip: str, text: str,
                             timeout_s: float = DEFAULT_TIMEOUT_S) -> bool:
    """Sends text to the XIAO ESP32C3's /display endpoint to render on
    the OLED -- the wireless replacement for showing guidance/output on
    a monitor. Requires the corresponding ESP32C3 firmware (not yet
    written) to actually render it; this is only the PC-side sender.

    UNTESTED -- no ESP32C3 hardware or matching firmware exists yet."""
    url = f"http://{esp32_display_ip}/display"
    data = text.encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"Content-Type": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 200
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach the wireless display at {url}: {e}")
