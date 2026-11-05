"""MNIST CNN classifier (PyTorch) with analysis outputs.

Run:  python main.py
Output folder: ./outputs/run_YYYYMMDD_HHMMSS (one new folder for each run)
Output files: training curves, confusion matrix, filters, feature maps,
misclassified samples, hyperparameter sweep, ablation, results.json,
best_model.pt, ensemble_member_N.pt

The reusable code is in the lib/ package:
    lib/console.py  formatting and progress bars
    lib/device.py   device description and selection
    lib/runs.py     run folders, timing history, incomplete-run reports
    lib/mnist.py    MNIST data, CNN model, training, evaluation, plots
"""
import argparse
import json
import platform
import secrets
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use a file-only backend. No display is necessary.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from lib.console import (clear_console_line, format_duration, print_heading,
                         print_labeled_value, print_progress_bar, print_section_header)
from lib.device import describe_compute_device, select_compute_device, set_compute_device
from lib.mnist import (AUGMENTATION_SETTINGS, MNIST_PIXEL_MEAN,
                       MNIST_PIXEL_STANDARD_DEVIATION, collect_softmax_probabilities,
                       compute_confusion_matrix, compute_per_class_metrics,
                       describe_misclassified_samples, evaluate_model,
                       evaluate_with_test_time_augmentation, find_most_confused_class_pairs,
                       load_mnist_data, plot_confusion_matrix, plot_feature_maps,
                       plot_first_layer_filters, plot_misclassified_images,
                       plot_training_history, summarize_prediction_quality, train_model)
from lib.runs import (INCOMPLETE_RUN_REPORT_FILE_NAME, create_new_run_directory,
                      load_previous_run_timings, print_duration_summary,
                      record_component_duration, write_incomplete_run_report)

#  constants
PROJECT_ROOT_DIRECTORY = Path(__file__).parent
DATASET_DIRECTORY = PROJECT_ROOT_DIRECTORY / "mnist"
OUTPUT_ROOT_DIRECTORY = PROJECT_ROOT_DIRECTORY / "outputs"
OUTPUT_ROOT_DIRECTORY.mkdir(exist_ok=True)

# A new random seed for each run. The operating system gives the random number.
# The range 0 to 2**32 - 1 is the range that numpy accepts. results.json stores the seed.
RANDOM_SEED = secrets.randbelow(2 ** 32)
NUMBER_OF_RUN_SECTIONS = 8  # Number of tasks that run_experiment shows with a section header.

# Training settings of the main model. results.json stores them.
MAIN_TRAINING_SETTINGS = dict(
    maximum_number_of_epochs=30, early_stopping_patience_in_epochs=8,
    optimizer="AdamW", learning_rate_schedule="CosineAnnealingLR (T_max = maximum epochs)",
    gradient_clipping_max_norm=1.0,
    validation_metric_for_early_stopping="validation accuracy (ties: lower validation loss)")
# Settings of the short runs (hyperparameter sweep and ablation study).
SHORT_TRAINING_SETTINGS = dict(
    maximum_number_of_epochs=8, early_stopping_patience_in_epochs=3)
# The ablation repeats every variant with these seed offsets. Averages reduce the noise
# of single short runs, because the differences between variants are small.
ABLATION_SEED_OFFSETS = (0, 1)
# The ensemble has the main model plus this number of extra models. The extra models use
# the same configuration and data split. Only the random seed changes (initialization,
# batch order, and augmentation). The ensemble averages the softmax probabilities.
ENSEMBLE_EXTRA_MODEL_COUNT = 2
ENSEMBLE_SEED_OFFSET = 1000  # Keeps the ensemble seeds apart from the ablation seeds.

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def run_experiment(current_run_output_directory, completed_component_durations_in_seconds,
                   previous_total_duration_in_seconds,
                   previous_component_durations_in_seconds,
                   previous_device_description, current_device_description):
    """Run the full experiment: train, evaluate, plot, sweep, and save the results.

    Inputs:
        current_run_output_directory (Path): Folder for the files of this run.
        completed_component_durations_in_seconds (dict): Empty dict at the start.
            The function adds one entry as each component finishes. The caller
            uses this dict to report the progress if the run stops early.
        previous_total_duration_in_seconds (float or None): Total of the last run.
        previous_component_durations_in_seconds (dict): Component times of the last run.
        previous_device_description (str): Device of the last run, or "unknown".
        current_device_description (str): Device of this run.
    Output:
        None. The function writes images, model weights, and results.json
        to the run folder and prints progress to the console.
    """
    print_heading("MNIST CNN experiment")
    run_start_timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
    print_labeled_value("Started", run_start_timestamp_text)
    print_labeled_value("Device", current_device_description)
    print_labeled_value("PyTorch", torch.__version__)
    print_labeled_value("Random seed", RANDOM_SEED)
    print_labeled_value("Output folder", current_run_output_directory)
    print("\n  Previous run")
    if previous_total_duration_in_seconds is None:
        print("    None found. The duration of this run is unknown.")
    else:
        print_labeled_value("Device", previous_device_description, indent_in_characters=4)
        print_labeled_value("Total time", format_duration(previous_total_duration_in_seconds),
                            indent_in_characters=4)
        if previous_device_description != current_device_description:
            print("    Note: this run uses a different device. Expect a different duration.")
        if previous_component_durations_in_seconds:
            component_name_width = max(len(name) for name in previous_component_durations_in_seconds)
            print("    Time of each component:")
            for component_name, component_duration in previous_component_durations_in_seconds.items():
                print(f"      {component_name:<{component_name_width}}  "
                      f"{format_duration(component_duration):>7}")
        else:
            print("    Note: this older result file has the main training time only.")
    whole_run_start_time_in_seconds = time.time()
    component_durations_in_seconds = completed_component_durations_in_seconds  # Shared with the caller.
    component_start_time_in_seconds = time.time()
    print_section_header(1, NUMBER_OF_RUN_SECTIONS, "Loading and splitting the MNIST data")

    (training_images_tensor, training_labels_tensor,
     test_images_tensor, test_labels_tensor) = load_mnist_data(DATASET_DIRECTORY)
    full_training_dataset = TensorDataset(training_images_tensor, training_labels_tensor)
    # Use 54,000 images for training and 6,000 for validation.
    training_dataset, validation_dataset = random_split(
        full_training_dataset, [54000, 6000],
        generator=torch.Generator().manual_seed(RANDOM_SEED))
    test_data_loader = DataLoader(
        TensorDataset(test_images_tensor, test_labels_tensor), batch_size=1000)
    component_start_time_in_seconds = record_component_duration(
        component_durations_in_seconds, "Data loading and splitting",
        component_start_time_in_seconds)

    # Flattening (no global average pooling) won the first ablation by about 0.5 points.
    # A learning rate of 3e-3 was a middle value between the best sweep values.
    base_model_configuration = dict(
        first_block_channels=32, second_block_channels=64, hidden_units=128,
        convolution_dropout=0.1, fully_connected_dropout=0.4,
        learning_rate=3e-3, weight_decay=1e-4, batch_size=128, use_augmentation=True,
        label_smoothing=0.05, use_residual=True, use_squeeze_excitation=True,
        use_global_average_pooling=False)

    # Train the main model.
    print_section_header(2, NUMBER_OF_RUN_SECTIONS, "Training the main model")
    training_start_time_in_seconds = time.time()
    main_trained_model, main_training_history = train_model(
        base_model_configuration, training_dataset, validation_dataset,
        maximum_number_of_epochs=MAIN_TRAINING_SETTINGS["maximum_number_of_epochs"],
        early_stopping_patience_in_epochs=MAIN_TRAINING_SETTINGS[
            "early_stopping_patience_in_epochs"])
    training_duration_in_seconds = time.time() - training_start_time_in_seconds
    component_start_time_in_seconds = record_component_duration(
        component_durations_in_seconds, "Main model training",
        component_start_time_in_seconds)
    # Ensemble members: the same configuration, trained again with different seeds.
    print_section_header(3, NUMBER_OF_RUN_SECTIONS, "Training the ensemble members")
    ensemble_member_models = [main_trained_model]
    ensemble_member_details = [dict(
        role="main model", seed=RANDOM_SEED,
        epochs_run=len(main_training_history["train_loss"]),
        best_epoch=main_training_history["best_epoch"],
        val_acc=main_training_history["val_acc"][main_training_history["best_epoch"] - 1],
        train_time_s=training_duration_in_seconds)]
    for member_number in range(1, ENSEMBLE_EXTRA_MODEL_COUNT + 1):
        member_seed = RANDOM_SEED + ENSEMBLE_SEED_OFFSET + member_number
        torch.manual_seed(member_seed)
        np.random.seed(member_seed % (2 ** 32))  # numpy accepts seeds below 2**32 only.
        print(f"  Training member {member_number + 1} of {ENSEMBLE_EXTRA_MODEL_COUNT + 1} "
              f"(seed {member_seed})")
        member_start_time_in_seconds = time.time()
        member_model, member_training_history = train_model(
            base_model_configuration, training_dataset, validation_dataset,
            maximum_number_of_epochs=MAIN_TRAINING_SETTINGS["maximum_number_of_epochs"],
            early_stopping_patience_in_epochs=MAIN_TRAINING_SETTINGS[
                "early_stopping_patience_in_epochs"],
            print_progress=False)
        member_duration_in_seconds = time.time() - member_start_time_in_seconds
        torch.save(member_model.state_dict(),
                   current_run_output_directory / f"ensemble_member_{member_number}.pt")
        ensemble_member_models.append(member_model)
        ensemble_member_details.append(dict(
            role=f"extra member {member_number}", seed=member_seed,
            epochs_run=len(member_training_history["train_loss"]),
            best_epoch=member_training_history["best_epoch"],
            val_acc=member_training_history["val_acc"][member_training_history["best_epoch"] - 1],
            train_time_s=member_duration_in_seconds))
        print(f"    {len(member_training_history['train_loss'])} epochs, validation accuracy "
              f"{ensemble_member_details[-1]['val_acc']:.2%}, "
              f"time {format_duration(member_duration_in_seconds)}")
    component_start_time_in_seconds = record_component_duration(
        component_durations_in_seconds,
        f"Ensemble member training ({ENSEMBLE_EXTRA_MODEL_COUNT} models)",
        component_start_time_in_seconds)
    print_section_header(4, NUMBER_OF_RUN_SECTIONS, "Evaluating the main model and saving it")
    plot_training_history(main_training_history, current_run_output_directory / "training_curves.png")
    torch.save(main_trained_model.state_dict(), current_run_output_directory / "best_model.pt")

    # Evaluate the main model on all three data sets.
    _, final_training_accuracy, _, _ = evaluate_model(
        main_trained_model, DataLoader(training_dataset, batch_size=1000))
    _, final_validation_accuracy, _, _ = evaluate_model(
        main_trained_model, DataLoader(validation_dataset, batch_size=1000))
    (final_test_loss, final_test_accuracy,
     test_predicted_labels, test_true_labels) = evaluate_model(main_trained_model, test_data_loader)

    # Class probabilities of the main model and of the ensemble on the test set.
    main_test_probabilities = collect_softmax_probabilities(main_trained_model, test_data_loader)
    member_test_probabilities = [main_test_probabilities] + [
        collect_softmax_probabilities(member_model, test_data_loader)
        for member_model in ensemble_member_models[1:]]
    ensemble_test_probabilities = np.mean(member_test_probabilities, axis=0)
    ensemble_predicted_labels = ensemble_test_probabilities.argmax(axis=1)
    for member_details, probabilities_of_member in zip(
            ensemble_member_details, member_test_probabilities):
        member_details["test_acc"] = float(
            (probabilities_of_member.argmax(axis=1) == test_true_labels).mean())
    main_quality = summarize_prediction_quality(main_test_probabilities, test_true_labels)
    ensemble_quality = summarize_prediction_quality(ensemble_test_probabilities, test_true_labels)
    ensemble_confusion_matrix = compute_confusion_matrix(
        test_true_labels, ensemble_predicted_labels)
    # Misclassification data. Each list starts with the most confident errors.
    main_misclassified_samples = describe_misclassified_samples(
        main_test_probabilities, test_true_labels)
    ensemble_misclassified_samples = describe_misclassified_samples(
        ensemble_test_probabilities, test_true_labels)
    ensemble_error_indices = {sample["test_index"] for sample in ensemble_misclassified_samples}
    for sample in main_misclassified_samples:
        # True when the ensemble also fails on this image. Such images are the hard ones.
        sample["ensemble_also_wrong"] = sample["test_index"] in ensemble_error_indices

    # Test-time augmentation: average the predictions over 5 shifted copies.
    tta_test_accuracy, tta_predicted_labels, _ = evaluate_with_test_time_augmentation(
        main_trained_model, test_data_loader)
    print_labeled_value("Test acc. (TTA)", f"{tta_test_accuracy:.2%}")

    test_confusion_matrix = compute_confusion_matrix(test_true_labels, test_predicted_labels)
    plot_confusion_matrix(test_confusion_matrix, current_run_output_directory / "confusion_matrix.png")
    per_class_metrics = compute_per_class_metrics(test_confusion_matrix)
    total_trainable_parameter_count = sum(
        parameter_tensor.numel() for parameter_tensor in main_trained_model.parameters())
    print_labeled_value("Train accuracy", f"{final_training_accuracy:.2%}")
    print_labeled_value("Validation acc.", f"{final_validation_accuracy:.2%}")
    print_labeled_value("Test accuracy",
                        f"{final_test_accuracy:.2%}  (loss {final_test_loss:.4f})")
    print_labeled_value("Parameters", f"{total_trainable_parameter_count:,}")
    print_labeled_value("Training time", format_duration(training_duration_in_seconds))
    print_labeled_value("Macro F1", f"{main_quality['macro_f1']:.4f}")
    print_labeled_value("Top-2 accuracy", f"{main_quality['top2_accuracy']:.2%}")
    print_labeled_value("Calibration (ECE)", f"{main_quality['expected_calibration_error']:.4f}")
    interval_lower_bound, interval_upper_bound = main_quality["accuracy_95_confidence_interval"]
    print_labeled_value("Test acc. 95% CI",
                        f"{interval_lower_bound:.2%} to {interval_upper_bound:.2%}")
    print_labeled_value(
        "Ensemble test acc.",
        f"{ensemble_quality['accuracy']:.2%}  ({len(ensemble_member_models)} models, "
        f"{ensemble_quality['errors']} errors; main model {main_quality['errors']} errors)")
    component_start_time_in_seconds = record_component_duration(
        component_durations_in_seconds, "Main model evaluation and saving",
        component_start_time_in_seconds)

    print_section_header(5, NUMBER_OF_RUN_SECTIONS, "Creating the analysis plots")
    plot_first_layer_filters(main_trained_model, current_run_output_directory / "filters.png")
    plot_feature_maps(main_trained_model, test_images_tensor[0],
                      current_run_output_directory / "feature_maps.png")
    plot_misclassified_images(test_images_tensor.numpy(), test_true_labels,
                              test_predicted_labels,
                              current_run_output_directory / "misclassified.png")
    component_start_time_in_seconds = record_component_duration(
        component_durations_in_seconds, "Analysis plots",
        component_start_time_in_seconds)

    #  Hyperparameter sweep. Change one setting at a time. Use short training runs.
    candidate_values_for_each_setting = {
        "learning_rate": [1e-4, 1e-3, 3e-3, 1e-2],
        "fully_connected_dropout": [0.0, 0.4, 0.7],
        "batch_size": [32, 128, 512],
        "second_block_channels": [16, 64, 128],
        "use_augmentation": [False, True],
    }
    print_section_header(6, NUMBER_OF_RUN_SECTIONS, "Hyperparameter sweep")
    sweep_results_for_each_setting = {}
    total_sweep_run_count = sum(
        len(list_of_values) for list_of_values in candidate_values_for_each_setting.values())
    completed_sweep_run_count = 0
    sweep_start_time_in_seconds = time.time()
    print(f"  {total_sweep_run_count} short training runs. Each run changes one setting.")
    print_progress_bar(0, total_sweep_run_count, sweep_start_time_in_seconds, "  sweep runs")
    for setting_name, list_of_candidate_values in candidate_values_for_each_setting.items():
        sweep_results_for_each_setting[setting_name] = []
        for candidate_value in list_of_candidate_values:
            # Copy the base configuration. Replace only the setting under test.
            sweep_model_configuration = dict(
                base_model_configuration, **{setting_name: candidate_value})
            sweep_trained_model, sweep_training_history = train_model(
                sweep_model_configuration, training_dataset, validation_dataset,
                **SHORT_TRAINING_SETTINGS, print_progress=False)
            _, sweep_test_accuracy, _, _ = evaluate_model(sweep_trained_model, test_data_loader)
            sweep_results_for_each_setting[setting_name].append(dict(
                value=candidate_value,
                test_acc=sweep_test_accuracy,
                final_train_acc=sweep_training_history["train_acc"][-1],
                best_val_acc=max(sweep_training_history["val_acc"])))
            completed_sweep_run_count += 1
            clear_console_line()  # Remove the open progress bar. Print the result line first.
            print(f"  [{completed_sweep_run_count:>2}/{total_sweep_run_count}] "
                  f"{setting_name:<23} = {str(candidate_value):<7} "
                  f"test accuracy {sweep_test_accuracy:.2%}")
            print_progress_bar(completed_sweep_run_count, total_sweep_run_count,
                               sweep_start_time_in_seconds, "  sweep runs")
    component_start_time_in_seconds = record_component_duration(
        component_durations_in_seconds, f"Hyperparameter sweep ({total_sweep_run_count} runs)",
        component_start_time_in_seconds)

    #  Ablation study. Remove one feature at a time (and all features together)
    #  to measure how much each one contributes. Use short training runs.
    print_section_header(7, NUMBER_OF_RUN_SECTIONS, "Ablation study")
    ablation_variants = {
        "full model": {},
        "no residual connections": dict(use_residual=False),
        "no squeeze-and-excitation": dict(use_squeeze_excitation=False),
        "with global average pooling": dict(use_global_average_pooling=True),
        "no label smoothing": dict(label_smoothing=0.0),
        "no augmentation": dict(use_augmentation=False),
        "plain baseline (all removed)": dict(
            use_residual=False, use_squeeze_excitation=False,
            use_global_average_pooling=False, label_smoothing=0.0, use_augmentation=False),
    }
    ablation_results = {}
    for variant_name, variant_overrides in ablation_variants.items():
        variant_start_time_in_seconds = time.time()
        accuracies_of_variant, validation_accuracies_of_variant = [], []
        for seed_offset in ABLATION_SEED_OFFSETS:
            torch.manual_seed(RANDOM_SEED + seed_offset)  # Different init and batch order.
            np.random.seed((RANDOM_SEED + seed_offset) % (2 ** 32))
            variant_model, variant_history = train_model(
                dict(base_model_configuration, **variant_overrides),
                training_dataset, validation_dataset,
                **SHORT_TRAINING_SETTINGS, print_progress=False)
            _, variant_test_accuracy, _, _ = evaluate_model(variant_model, test_data_loader)
            accuracies_of_variant.append(variant_test_accuracy)
            validation_accuracies_of_variant.append(max(variant_history["val_acc"]))
        ablation_results[variant_name] = dict(
            overrides=variant_overrides,
            seed_offsets=list(ABLATION_SEED_OFFSETS),
            test_acc_per_seed=accuracies_of_variant,
            test_acc=float(np.mean(accuracies_of_variant)),
            test_acc_std=float(np.std(accuracies_of_variant)),
            best_val_acc=float(np.mean(validation_accuracies_of_variant)),
            params=sum(p.numel() for p in variant_model.parameters()),
            train_time_s=time.time() - variant_start_time_in_seconds)
        print(f"  {variant_name:<30} test accuracy {ablation_results[variant_name]['test_acc']:.2%} "
              f"(+/- {ablation_results[variant_name]['test_acc_std']:.2%})  "
              f"params {ablation_results[variant_name]['params']:,}  "
              f"time {format_duration(ablation_results[variant_name]['train_time_s'])}")
    ablation_figure, ablation_axis = plt.subplots(figsize=(8, 3.8))
    ablation_axis.barh(list(ablation_results),
                       [entry["test_acc"] for entry in ablation_results.values()],
                       xerr=[entry["test_acc_std"] for entry in ablation_results.values()])
    ablation_axis.set(xlabel="Mean test accuracy (error bar = standard deviation)",
                      title=f"Ablation study (8-epoch runs, {len(ABLATION_SEED_OFFSETS)} seeds)")
    ablation_axis.set_xlim(min(entry["test_acc"] for entry in ablation_results.values()) - 0.004,
                           1.0)
    ablation_axis.invert_yaxis()
    ablation_axis.grid(alpha=0.3, axis="x")
    ablation_figure.tight_layout()
    ablation_figure.savefig(current_run_output_directory / "ablation.png", dpi=150)
    plt.close(ablation_figure)
    component_start_time_in_seconds = record_component_duration(
        component_durations_in_seconds,
        f"Ablation study ({len(ablation_variants) * len(ABLATION_SEED_OFFSETS)} runs)",
        component_start_time_in_seconds)

    print_section_header(8, NUMBER_OF_RUN_SECTIONS, "Saving the sweep plot and the results")
    sweep_figure, sweep_axes = plt.subplots(
        1, len(candidate_values_for_each_setting), figsize=(16, 3.4))
    for single_axis, (setting_name, list_of_result_dictionaries) in zip(
            sweep_axes, sweep_results_for_each_setting.items()):
        single_axis.plot(
            [str(result_dictionary["value"]) for result_dictionary in list_of_result_dictionaries],
            [result_dictionary["test_acc"] for result_dictionary in list_of_result_dictionaries],
            "o-", label="test")
        single_axis.plot(
            [str(result_dictionary["value"]) for result_dictionary in list_of_result_dictionaries],
            [result_dictionary["best_val_acc"] for result_dictionary in list_of_result_dictionaries],
            "s--", label="validation")  # Validation is the honest basis for choosing a value.
        single_axis.legend(fontsize=7)
        single_axis.set(title=setting_name, xlabel=setting_name, ylabel="Test accuracy")
        single_axis.grid(alpha=0.3)
    sweep_figure.tight_layout()
    sweep_figure.savefig(current_run_output_directory / "hyperparameter_sweep.png", dpi=150)
    plt.close(sweep_figure)
    component_start_time_in_seconds = record_component_duration(
        component_durations_in_seconds, "Sweep plot", component_start_time_in_seconds)

    #  Save all numbers for the report.
    # The keys total_run_time_s and component_durations_s give the next run its expected durations.
    total_run_duration_in_seconds = time.time() - whole_run_start_time_in_seconds
    results_dictionary = dict(
        seed=RANDOM_SEED,
        device=current_device_description,
        started_at=run_start_timestamp_text,
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        environment=dict(
            python=platform.python_version(), platform=platform.platform(),
            torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__,
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()),
        data_split=dict(
            training=len(training_dataset), validation=len(validation_dataset),
            test=len(test_data_loader.dataset),
            pixel_mean=MNIST_PIXEL_MEAN, pixel_standard_deviation=MNIST_PIXEL_STANDARD_DEVIATION),
        training_settings=MAIN_TRAINING_SETTINGS,
        short_run_settings=SHORT_TRAINING_SETTINGS,
        augmentation=AUGMENTATION_SETTINGS,
        test_time_augmentation=dict(shifts=[[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]),
        history=main_training_history,
        best_epoch=main_training_history["best_epoch"],
        stopped_early=len(main_training_history["val_loss"])
        < MAIN_TRAINING_SETTINGS["maximum_number_of_epochs"],
        most_confused_pairs=find_most_confused_class_pairs(test_confusion_matrix),
        ablation_seed_offsets=list(ABLATION_SEED_OFFSETS),
        num_errors_tta=int((tta_predicted_labels != test_true_labels).sum()),
        ablation=ablation_results,
        total_run_time_s=total_run_duration_in_seconds,
        component_durations_s=component_durations_in_seconds,
        config=base_model_configuration, params=total_trainable_parameter_count,
        epochs_run=len(main_training_history["train_loss"]),
        train_time_s=training_duration_in_seconds,
        train_acc=final_training_accuracy, val_acc=final_validation_accuracy,
        test_acc=final_test_accuracy, test_loss=final_test_loss,
        test_acc_tta=tta_test_accuracy,
        main_model_quality=main_quality,
        misclassified_samples=main_misclassified_samples,
        ensemble=dict(
            member_count=len(ensemble_member_models),
            extra_member_count=ENSEMBLE_EXTRA_MODEL_COUNT,
            seed_offset=ENSEMBLE_SEED_OFFSET,
            combination="mean of the softmax probabilities",
            members=ensemble_member_details,
            quality=ensemble_quality,
            per_class=compute_per_class_metrics(ensemble_confusion_matrix),
            confusion=ensemble_confusion_matrix.tolist(),
            most_confused_pairs=find_most_confused_class_pairs(ensemble_confusion_matrix),
            misclassified_samples=ensemble_misclassified_samples),
        per_class=per_class_metrics,
        confusion=test_confusion_matrix.tolist(),
        num_errors=int((test_predicted_labels != test_true_labels).sum()),
        sweep=sweep_results_for_each_setting)
    (current_run_output_directory / "results.json").write_text(
        json.dumps(results_dictionary, indent=2, default=float))
    print(f"  Results saved to: {current_run_output_directory}")
    print_duration_summary(component_durations_in_seconds, total_run_duration_in_seconds,
                           previous_component_durations_in_seconds,
                           previous_total_duration_in_seconds,
                           current_device_description, previous_device_description)


def main():
    """Choose the device, ask how many runs to make, and make the runs one after the other.

    Each run uses the same device and a new random seed.
    Each run creates its own run folder. If the user cancels (Ctrl+C) or an error
    occurs, the run writes RUN_INCOMPLETE.txt in its folder and the remaining
    runs do not start.

    Input:
        None. The optional command-line option --device (cpu, gpu, or cuda:N)
        skips the device question.
    Output:
        None.
    """
    argument_parser = argparse.ArgumentParser(description="Train and analyze the MNIST CNN.")
    argument_parser.add_argument(
        "--device", default=None,
        help="Device to use: cpu, gpu, or cuda:N. Without this option the script asks "
             "when more than one device is available.")
    parsed_arguments = argument_parser.parse_args()
    try:
        set_compute_device(select_compute_device(parsed_arguments.device))
        number_of_runs = ask_number_of_runs()
    except KeyboardInterrupt:
        print("\n  Cancelled before the run started. No run folder was created.")
        return

    for run_number in range(1, number_of_runs + 1):
        if number_of_runs > 1:
            print_heading(f"Run {run_number} of {number_of_runs}")
        execute_one_run()


def ask_number_of_runs():
    """Ask the user how many times to run the whole experiment.

    The Enter key gives the default of 1 run. If the input is not interactive
    (no console), the function returns 1 and does not ask.

    Input:
        None.
    Output:
        int: Number of runs, 1 or more.
    """
    if not sys.stdin.isatty():
        return 1
    while True:
        try:
            answer_text = input("\n  Number of runs (Enter for the default of 1): ").strip()
        except EOFError:  # The input closed. Use the default.
            return 1
        if answer_text == "":
            return 1
        if answer_text.isdigit() and int(answer_text) >= 1:
            return int(answer_text)
        print(f"  '{answer_text}' is not valid. Enter a whole number of 1 or more.")


def execute_one_run():
    """Make one complete run with a new random seed and write a report file if it stops early.

    The function creates the run folder, then calls run_experiment.
    If the user cancels (Ctrl+C) or an error occurs, the function writes
    RUN_INCOMPLETE.txt in the run folder and then raises the same exception again,
    so the console still shows the stop reason.
    A forced stop (closed terminal, killed process, power loss) cannot be caught.
    Such a run folder has neither results.json nor a report file.

    Input:
        None. The function uses the device that set_compute_device stored.
    Output:
        None.
    """
    global RANDOM_SEED  # run_experiment and the data split read this value.
    RANDOM_SEED = secrets.randbelow(2 ** 32)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Read the previous durations before this run creates its own folder.
    (previous_total_duration_in_seconds,
     previous_component_durations_in_seconds,
     previous_device_description) = load_previous_run_timings(OUTPUT_ROOT_DIRECTORY)
    current_device_description = describe_compute_device()
    current_run_output_directory = create_new_run_directory(OUTPUT_ROOT_DIRECTORY)
    completed_component_durations_in_seconds = {}  # The run fills this dict.
    main_start_time_in_seconds = time.time()
    try:
        run_experiment(current_run_output_directory, completed_component_durations_in_seconds,
                       previous_total_duration_in_seconds,
                       previous_component_durations_in_seconds,
                       previous_device_description, current_device_description)
    except KeyboardInterrupt:
        write_incomplete_run_report(
            current_run_output_directory, "CANCELLED",
            completed_component_durations_in_seconds,
            time.time() - main_start_time_in_seconds, current_device_description)
        print(f"\nRun cancelled. Report: "
              f"{current_run_output_directory / INCOMPLETE_RUN_REPORT_FILE_NAME}")
        raise
    except Exception:
        write_incomplete_run_report(
            current_run_output_directory, "FAILED",
            completed_component_durations_in_seconds,
            time.time() - main_start_time_in_seconds, current_device_description,
            error_traceback_text=traceback.format_exc())
        print(f"\nRun failed. Report: "
              f"{current_run_output_directory / INCOMPLETE_RUN_REPORT_FILE_NAME}")
        raise


if __name__ == "__main__":
    main()
