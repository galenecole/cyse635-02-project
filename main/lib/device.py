"""Compute device helpers: describe, list, choose, and store the device for a run.

The active device is stored in this module. Call set_compute_device once at the
start of a script. Every other module then calls get_compute_device.
"""
import os
import platform
import subprocess
import sys

import torch

from lib.console import print_heading

# Default device. A script replaces it with set_compute_device().
_active_compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_compute_device(new_compute_device):
    """Store the device that all later computation uses.

    Input:
        new_compute_device (torch.device): Device to use, for example cuda:0 or cpu.
    Output:
        None.
    """
    global _active_compute_device
    _active_compute_device = new_compute_device


def get_compute_device():
    """Return the device that the run uses.

    Input:
        None.
    Output:
        torch.device: The device set with set_compute_device (or the default device).
    """
    return _active_compute_device


def detect_cpu_name():
    """Find the marketing name of the processor, for example "AMD Ryzen 7 5800X".

    The function uses the registry on Windows, sysctl on macOS, and /proc/cpuinfo
    on Linux. If these sources fail, it uses the generic text from platform.processor().

    Input:
        None.
    Output:
        str: Name of the processor. "unknown CPU" if no source gives a name.
    """
    try:
        if sys.platform == "win32":
            import winreg  # Only available on Windows.
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as registry_key:
                processor_name, _ = winreg.QueryValueEx(registry_key, "ProcessorNameString")
            return " ".join(processor_name.split())  # Remove the extra spaces.
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo") as cpu_information_file:
            for line_text in cpu_information_file:
                if line_text.startswith("model name"):
                    return line_text.split(":", 1)[1].strip()
    except (OSError, ImportError, ValueError, subprocess.SubprocessError):
        pass  # Use the fallback below.
    return platform.processor() or "unknown CPU"


def describe_compute_device(device_to_describe=None):
    """Return a short text that names a device that can run the computation.

    Input:
        device_to_describe (torch.device or None): Device to describe.
            None gives the device that this run uses (the active device).
    Output:
        str: "GPU: name (memory in GB)" for a GPU.
            "CPU: name (number of threads)" for the CPU.
    """
    if device_to_describe is None:
        device_to_describe = get_compute_device()
    if device_to_describe.type == "cuda":
        gpu_properties = torch.cuda.get_device_properties(device_to_describe)
        return f"GPU: {gpu_properties.name} ({gpu_properties.total_memory / 1024 ** 3:.1f} GB)"
    return f"CPU: {detect_cpu_name()} ({os.cpu_count()} threads)"


def list_available_compute_devices():
    """List the devices that PyTorch can use on this computer.

    Input:
        None.
    Output:
        list of torch.device: Each GPU first (cuda:0, cuda:1, ...), then the CPU.
            The CPU is always in the list. The first item is the default device.
    """
    available_devices = [torch.device("cuda", gpu_index)
                         for gpu_index in range(torch.cuda.device_count())]
    available_devices.append(torch.device("cpu"))
    return available_devices


def convert_text_to_device(requested_device_text, available_devices):
    """Convert the text of the --device option to one of the available devices.

    Inputs:
        requested_device_text (str): "cpu", "gpu", "cuda", or "cuda:N" (N is a GPU index).
            "gpu" and "cuda" mean the first GPU (cuda:0).
        available_devices (list of torch.device): Devices from list_available_compute_devices.
    Output:
        torch.device or None: The matching device. None if the text is not valid
        or the device is not available on this computer.
    """
    normalized_text = requested_device_text.strip().lower()
    if normalized_text in ("gpu", "cuda"):
        normalized_text = "cuda:0"
    for available_device in available_devices:
        if normalized_text == str(available_device):  # str() gives "cpu" or "cuda:N".
            return available_device
    return None


def select_compute_device(requested_device_text=None):
    """Choose the device for this run. Ask the user if more than one device exists.

    Rules:
        1. If requested_device_text is given (--device option), use it and do not ask.
        2. If only the CPU is available, use it and do not ask.
        3. If the input is not interactive (no console), use the default device
           (the first GPU) and do not ask.
        4. Otherwise, show a numbered list and ask. The Enter key gives the default.

    Input:
        requested_device_text (str or None): Value of the --device option.
    Output:
        torch.device: The device that the run uses.
    Raises:
        SystemExit: If requested_device_text is not valid or not available.
    """
    available_devices = list_available_compute_devices()
    device_labels = [
        describe_compute_device(device) + (f"  [{device}]" if device.type == "cuda" else "")
        for device in available_devices]

    print_heading("Device selection")
    if requested_device_text is not None:
        chosen_device = convert_text_to_device(requested_device_text, available_devices)
        if chosen_device is None:
            valid_choices = ", ".join(str(device) for device in available_devices)
            raise SystemExit(f"The device '{requested_device_text}' is not available. "
                             f"Valid choices on this computer: {valid_choices} (or 'gpu').")
        print(f"  Device set with --device: {describe_compute_device(chosen_device)}")
        return chosen_device
    if len(available_devices) == 1:
        print("  No GPU found. The run uses the CPU.")
        print(f"  {device_labels[0]}")
        return available_devices[0]
    if not sys.stdin.isatty():
        print("  The input is not interactive. The run uses the default device.")
        print(f"  {device_labels[0]}")
        return available_devices[0]

    print("  More than one device is available.\n")
    for option_number, device_label in enumerate(device_labels, start=1):
        default_marker = "  (default)" if option_number == 1 else ""
        print(f"    [{option_number}] {device_label}{default_marker}")
    while True:
        try:
            answer_text = input(f"\n  Select a device (1-{len(available_devices)}, "
                                "Enter for the default): ").strip()
        except EOFError:  # The input closed. Use the default device.
            answer_text = ""
        if answer_text == "":
            return available_devices[0]
        if answer_text.isdigit() and 1 <= int(answer_text) <= len(available_devices):
            return available_devices[int(answer_text) - 1]
        print(f"  '{answer_text}' is not valid. Enter a number from 1 to {len(available_devices)}.")
