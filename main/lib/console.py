"""Console formatting helpers: durations, headings, labeled values, progress bars.

These functions do not depend on any model or data set. Use them in every project.
"""
import time

def format_duration(duration_in_seconds):
    """Convert a duration in seconds to the text format H:MM:SS.

    Input:
        duration_in_seconds (float): Duration to convert.
    Output:
        str: Duration as hours, minutes, and seconds. Example: "1:05:09".
    """
    total_whole_seconds = int(round(duration_in_seconds))
    hours_part, remainder_seconds = divmod(total_whole_seconds, 3600)
    minutes_part, seconds_part = divmod(remainder_seconds, 60)
    return f"{hours_part}:{minutes_part:02d}:{seconds_part:02d}"


CONSOLE_WIDTH_IN_CHARACTERS = 78  # Width of the rules and the largest width of a progress line.


def print_heading(heading_text):
    """Print a heading between two full-width rules.

    Input:
        heading_text (str): Text of the heading.
    Output:
        None. The function prints to the console.
    """
    rule_line = "=" * CONSOLE_WIDTH_IN_CHARACTERS
    print(f"\n{rule_line}\n {heading_text}\n{rule_line}")


def print_section_header(section_number, total_section_count, section_title):
    """Print a header line that shows the start of a new task of the run.

    The line shows the task number, the task title, and the current clock time.
    Example: "-- [2/6] Training the main model ------------------ 14:05:12"

    Inputs:
        section_number (int): Number of this task.
        total_section_count (int): Total number of tasks.
        section_title (str): Short title of the task.
    Output:
        None. The function prints to the console.
    """
    header_text = f"[{section_number}/{total_section_count}] {section_title}"
    clock_time_text = time.strftime("%H:%M:%S")
    # The fixed text uses 5 characters: "-- " before the title and 2 spaces around the fill.
    fill_character_count = max(
        CONSOLE_WIDTH_IN_CHARACTERS - len(header_text) - len(clock_time_text) - 5, 3)
    print(f"\n-- {header_text} {'-' * fill_character_count} {clock_time_text}")


def print_labeled_value(label_text, value_text, indent_in_characters=2,
                        label_width_in_characters=16):
    """Print one line in the format "label : value" with aligned colons.

    Inputs:
        label_text (str): Name of the value.
        value_text (object): Value to show. The function converts it to text.
        indent_in_characters (int): Number of spaces before the label.
        label_width_in_characters (int): Width of the label column.
    Output:
        None. The function prints to the console.
    """
    print(f"{' ' * indent_in_characters}{label_text:<{label_width_in_characters}}: {value_text}")


def clear_console_line():
    """Remove the text of the current console line and move the cursor to its start.

    Use this function before a message is printed over an open progress bar.

    Input:
        None.
    Output:
        None. The function prints to the console.
    """
    print("\r" + " " * CONSOLE_WIDTH_IN_CHARACTERS + "\r", end="", flush=True)


def print_progress_bar(completed_step_count, total_step_count, start_time_in_seconds,
                       label_text, bar_width_in_characters=20, keep_line_open=False):
    """Print a text progress bar on the current console line.

    The line shows the bar, the step count, the completed percentage, and the
    times as "elapsed<remaining". The remaining time is an estimate from the
    mean time of each completed step. Each call replaces the previous bar.

    Inputs:
        completed_step_count (int): Number of steps that are complete.
        total_step_count (int): Total number of steps in the loop.
        start_time_in_seconds (float): Value of time.time() when the loop started.
        label_text (str): Text to show before the bar.
        bar_width_in_characters (int): Width of the bar in characters.
        keep_line_open (bool): When False, the function starts a new line after
            the last step. When True, the line stays open so the caller can
            replace it with a summary.
    Output:
        None. The function prints to the console.
    """
    completed_fraction = completed_step_count / total_step_count
    filled_character_count = int(bar_width_in_characters * completed_fraction)
    bar_text = "#" * filled_character_count + "." * (bar_width_in_characters - filled_character_count)
    elapsed_seconds = time.time() - start_time_in_seconds
    if completed_step_count > 0:
        # Estimate the remaining time from the mean time of each completed step.
        remaining_time_text = format_duration(
            elapsed_seconds / completed_step_count * (total_step_count - completed_step_count))
    else:
        remaining_time_text = "?:??:??"  # No step is complete. An estimate is not possible.
    line_ends_here = completed_step_count == total_step_count and not keep_line_open
    print(f"\r{label_text} [{bar_text}] {completed_step_count}/{total_step_count} "
          f"{completed_fraction:>4.0%} {format_duration(elapsed_seconds)}<{remaining_time_text}",
          end="\n" if line_ends_here else "", flush=True)
