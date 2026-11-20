"""Derive conservative generation defaults from the active ComfyUI device."""

import os
import platform


GIB = 1024**3


def memory(value):
    return value if type(value) is int and value >= 0 else None


def hardware_report(payload, starter=False):
    system = payload.get("system") or {}
    devices = payload.get("devices") or []
    device = devices[0] if devices else {}
    kind = device.get("type", "unknown")
    ram = memory(system.get("ram_total"))
    vram = memory(device.get("vram_total")) if kind not in {"cpu", "mps", "unknown"} else None
    vram_free = memory(device.get("vram_free")) if vram is not None else None

    notes = ["Estimates depend on the checkpoint, other running apps, and optional conditioning."]
    if kind == "cpu":
        width, height, steps, batch_limit = (512, 512, 16, 1) if (ram or 0) >= 12 * GIB else (384, 384, 12, 1)
        label = "CPU mode"
        notes.append("CPU generation can take several minutes or longer. Use batch 1 and start with SD 1.5.")
    else:
        capacity = (ram or 0) // 2 if kind == "mps" else vram or 0
        if capacity >= 12 * GIB:
            width, height, steps, batch_limit = 1024, 768, 28, 2 if capacity >= 16 * GIB else 1
            label = "Accelerated"
        elif capacity >= 7 * GIB:
            width, height, steps, batch_limit = 768, 640, 24, 1
            label = "Balanced"
        else:
            width, height, steps, batch_limit = 512, 512, 18, 1
            label = "Memory-saving"
            notes.append("Start with SD 1.5. Larger checkpoints or ControlNet may exceed available memory.")
        if kind == "mps":
            notes.append("Apple GPU memory is shared with system RAM.")
        if kind == "unknown":
            label = "Conservative"
            notes.append("ComfyUI did not report an active device.")
        if vram is not None and vram_free is not None and vram_free < min(vram // 3, 2 * GIB):
            notes.append("Free GPU memory is low. If generation fails, close other GPU apps or lower the resolution.")
    limits = {"width": width, "height": height, "pixels": width * height, "batch": batch_limit}
    if starter:
        width, height = min(width, 512), min(height, 512)
    ram_free = memory(system.get("ram_free"))
    if ram_free is not None and ram_free < 3 * GIB:
        notes.append("Available system RAM is low. Close other apps before attempting a render.")
    return {
        "label": label,
        "platform": platform.system(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "device": {"name": device.get("name") or "Unavailable", "type": kind,
                   "memory": vram, "free": vram_free},
        "ram": ram,
        "ram_free": ram_free,
        "defaults": {"width": width, "height": height, "steps": steps, "batch": 1},
        "limits": limits,
        "notes": notes,
    }
