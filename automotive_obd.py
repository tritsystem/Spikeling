#!/usr/bin/env python
"""
automotive_obd.py — real-time OBD-II diagnostics: check-engine codes
(DTCs) and live sensor data (RPM, coolant temp, speed, etc.) via a real
ELM327 adapter, using the `obd` (python-OBD) library -- a mature,
well-established package for this, not reinvented. Real API confirmed
against the installed package before writing this (obd.OBD.query/
is_connected/status/port_name/protocol_name; obd.commands.RPM/
COOLANT_TEMP/SPEED/THROTTLE_POS/GET_DTC all exist).

HONEST STATUS: UNTESTED. No physical ELM327 adapter exists yet -- same
discipline as wireless_glasses.py's stub functions. Written in the
correct shape for when hardware arrives; every function raises a real
error on failure rather than returning a fake "connected"/success status.

INDEPENDENT of electronics_assistant.py and the photo-ID/vault-logging
path, per explicit design: zero import dependency either direction, and
nothing here runs at import time (no auto-connect) -- either capability
works completely on its own. A missing/disconnected OBD-II adapter never
affects capture_photo/log_part_note/etc., and vice versa.

    from automotive_obd import connect_obd, read_dtc_codes, read_live_data, disconnect_obd
"""
import obd

_connection = None


def connect_obd(port: str = None, timeout_s: float = 10.0) -> dict:
    """Connects to a real ELM327 adapter. port=None lets python-OBD
    auto-detect (works for USB or Bluetooth-paired-as-COM-port adapters
    on Windows); pass an explicit port (e.g. "COM5") or a WiFi adapter's
    "socket://<ip>:<port>" string otherwise.

    UNTESTED -- no ELM327 hardware exists yet. Raises a real error if no
    adapter is found; never returns a fake "connected" status."""
    global _connection
    _connection = obd.OBD(portstr=port, timeout=timeout_s)
    if not _connection.is_connected():
        status = _connection.status()
        _connection = None
        raise RuntimeError(
            f"Could not connect to a real OBD-II adapter (port={port or 'auto'}). "
            f"Status: {status}")
    return {
        "connected": True,
        "port": _connection.port_name(),
        "protocol": str(_connection.protocol_name()),
    }


def is_obd_connected() -> bool:
    return _connection is not None and _connection.is_connected()


def read_dtc_codes() -> list:
    """Reads real check-engine/diagnostic trouble codes. Requires
    connect_obd() to have succeeded first -- raises if not connected,
    doesn't silently return an empty list that could be mistaken for
    'no codes found'."""
    if not is_obd_connected():
        raise RuntimeError("Not connected to a real OBD-II adapter -- call connect_obd() first.")
    response = _connection.query(obd.commands.GET_DTC)
    if response.is_null():
        return []
    return [{"code": code, "description": desc} for code, desc in response.value]


def read_live_data(pids: list = None) -> dict:
    """Reads real live sensor values. pids: list of python-OBD command
    names (e.g. ["RPM", "COOLANT_TEMP", "SPEED"]); defaults to a small
    common starter set if not given. Requires connect_obd() first. A pid
    the car doesn't support comes back as None -- a real, honest "no
    data", not fabricated."""
    if not is_obd_connected():
        raise RuntimeError("Not connected to a real OBD-II adapter -- call connect_obd() first.")
    pids = pids or ["RPM", "COOLANT_TEMP", "SPEED", "THROTTLE_POS"]
    out = {}
    for pid in pids:
        cmd = getattr(obd.commands, pid, None)
        if cmd is None:
            out[pid] = None
            continue
        response = _connection.query(cmd)
        out[pid] = None if response.is_null() else str(response.value)
    return out


def disconnect_obd() -> dict:
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
    return {"disconnected": True}
