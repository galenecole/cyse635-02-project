"""Run management: timestamped run folders, timing history, and incomplete-run reports.

These functions do not depend on any model or data set. Use them in every project.
"""
import json
import time

from lib.console import format_duration, print_heading, print_labeled_value

RUN_DIRECTORY_PREFIX = "run_"  # Each run folder name is this prefix plus a timestamp.
INCOMPLETE_RUN_REPORT_FILE_NAME = "RUN_INCOMPLETE.txt"


def create_new_run_directory(output_root_directory):
    """Create a new timestamped folder for the outputs of this run.

    Input:
        output_root_directory (Path): Folder that holds all run folders.
    Output:
        Path: The new folder, for example outputs/run_20261103_141500.
    """
    timestamp_text = time.strftime("%Y%m%d_%H%M%S")
    new_run_directory = output_root_directory / f"{RUN_DIRECTORY_PREFIX}{timestamp_text}"
    new_run_directory.mkdir(exist_ok=False)
    return new_run_directory


def find_most_recent_completed_run_directory(output_root_directory):
    """Find the newest run folder that contains a results.json file.

    The timestamp in the folder name gives the creation order.
    A folder without results.json is an incomplete run. The function skips it.

    Input:
        output_root_directory (Path): Folder that holds all run folders.
    Output:
        Path or None: The newest completed run folder.
        None if no completed run folder exists.
    """
    completed_run_directories = sorted(
        directory_path for directory_path in output_root_directory.glob(f"{RUN_DIRECTORY_PREFIX}*")
        if directory_path.is_dir() and (directory_path / "results.json").exists())
    return completed_run_directories[-1] if completed_run_directories else None


def load_previous_run_timings(output_root_directory):
    """Read the durations and the device of the last complete run.

    The function reads results.json from the newest completed run folder.
    If no run folder exists, it reads output_root_directory / "results.json"
    (the location that earlier versions of this script used).

    Input:
        output_root_directory (Path): Folder that holds all run folders.
    Output:
        tuple of 3 items:
            previous_total_duration_in_seconds (float or None): Total duration.
                None if no previous run exists.
                Older files without the total duration give the main training time.
            previous_component_durations (dict): Component name -> seconds.
                The dict is empty if the file has no component durations.
            previous_device_description (str): Device of the last run.
                "unknown" if the file does not record the device.
    """
    most_recent_run_directory = find_most_recent_completed_run_directory(output_root_directory)
    if most_recent_run_directory is not None:
        previous_results_path = most_recent_run_directory / "results.json"
    else:
        previous_results_path = output_root_directory / "results.json"  # Older location.
    if not previous_results_path.exists():
        return None, {}, "unknown"
    try:
        previous_results = json.loads(previous_results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None, {}, "unknown"  # The file is damaged. Ignore it.
    previous_total_duration_in_seconds = previous_results.get(
        "total_run_time_s", previous_results.get("train_time_s"))
    return (previous_total_duration_in_seconds,
            previous_results.get("component_durations_s", {}),
            previous_results.get("device", "unknown"))


def record_component_duration(component_durations_in_seconds, component_name,
                              component_start_time_in_seconds):
    """Store the duration of one run component, print it, and start the next one.

    Inputs:
        component_durations_in_seconds (dict): Component name -> seconds.
            The function adds one entry to this dict.
        component_name (str): Name of the component that just ended.
        component_start_time_in_seconds (float): Value of time.time() when
            the component started.
    Output:
        float: The current time. Use it as the start time of the next component.
    """
    current_time_in_seconds = time.time()
    component_durations_in_seconds[component_name] = (
        current_time_in_seconds - component_start_time_in_seconds)
    print(f"  Finished in {format_duration(component_durations_in_seconds[component_name])}.")
    return current_time_in_seconds


def print_duration_summary(component_durations_in_seconds, total_duration_in_seconds,
                           previous_component_durations_in_seconds,
                           previous_total_duration_in_seconds,
                           current_device_description, previous_device_description):
    """Print a table of the duration of each component and the total duration.

    Inputs:
        component_durations_in_seconds (dict): Component name -> seconds, this run.
        total_duration_in_seconds (float): Total duration of this run.
        previous_component_durations_in_seconds (dict): Same format, last run.
            Can be empty.
        previous_total_duration_in_seconds (float or None): Total of the last run.
        current_device_description (str): Device of this run.
        previous_device_description (str): Device of the last run, or "unknown".
    Output:
        None. The function prints to the console.
    """
    name_column_width = max(len(name) for name in component_durations_in_seconds)
    name_column_width = max(name_column_width, len("TOTAL"))
    print_heading("Run time summary (H:MM:SS)")
    print_labeled_value("This run device", current_device_description)
    print_labeled_value("Last run device", previous_device_description)
    if (previous_total_duration_in_seconds is not None
            and previous_device_description != current_device_description):
        print("  Note: the devices are different. Do not compare the times directly.")
    print()
    print(f"  {'Component':<{name_column_width}}  {'This run':>9}  {'Share':>6}  {'Last run':>9}")
    print("  " + "-" * (name_column_width + 30))
    for component_name, component_duration in component_durations_in_seconds.items():
        previous_duration = previous_component_durations_in_seconds.get(component_name)
        previous_duration_text = (
            format_duration(previous_duration) if previous_duration is not None else "-")
        share_of_total = component_duration / total_duration_in_seconds
        print(f"  {component_name:<{name_column_width}}  "
              f"{format_duration(component_duration):>9}  {share_of_total:>6.1%}  "
              f"{previous_duration_text:>9}")
    previous_total_text = (
        format_duration(previous_total_duration_in_seconds)
        if previous_total_duration_in_seconds is not None else "-")
    print("  " + "-" * (name_column_width + 30))
    print(f"  {'TOTAL':<{name_column_width}}  "
          f"{format_duration(total_duration_in_seconds):>9}  {1.0:>6.1%}  "
          f"{previous_total_text:>9}")




def write_incomplete_run_report(run_output_directory, outcome_name,
                                completed_component_durations_in_seconds,
                                elapsed_seconds_before_stop, device_description,
                                error_traceback_text=None):
    """Write a text file in the run folder that explains why the run is incomplete.

    Use this function when a run is cancelled or fails before it writes results.json.

    Inputs:
        run_output_directory (Path): The run folder to write the file in.
        outcome_name (str): "CANCELLED" or "FAILED".
        completed_component_durations_in_seconds (dict): Component name -> seconds.
            Only the components that finished before the stop.
        elapsed_seconds_before_stop (float): Time from the start of the run to the stop.
        device_description (str): Device of this run.
        error_traceback_text (str or None): Full error text. None for a cancellation.
    Output:
        None. The function writes RUN_INCOMPLETE.txt in the run folder.
    """
    report_lines = [
        f"Run status: {outcome_name}",
        f"Time of stop: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Time from start to stop: {format_duration(elapsed_seconds_before_stop)} (H:MM:SS)",
        f"Device: {device_description}",
        "",
        "The run did not write results.json. This folder holds partial results only.",
        "The next run ignores this folder when it looks for the previous run time.",
        "",
        "Components that finished before the stop:",
    ]
    if completed_component_durations_in_seconds:
        for component_name, component_duration in completed_component_durations_in_seconds.items():
            report_lines.append(f"  {component_name}: {format_duration(component_duration)}")
    else:
        report_lines.append("  None")
    report_lines.append("")
    report_lines.append("Files in this folder (other than this report):")
    names_of_other_files = sorted(
        file_path.name for file_path in run_output_directory.iterdir()
        if file_path.name != INCOMPLETE_RUN_REPORT_FILE_NAME)
    report_lines.extend(f"  {file_name}" for file_name in names_of_other_files)
    if not names_of_other_files:
        report_lines.append("  None")
    if error_traceback_text:
        report_lines.extend(["", "Error details:", error_traceback_text])
    (run_output_directory / INCOMPLETE_RUN_REPORT_FILE_NAME).write_text(
        "\n".join(report_lines) + "\n")
