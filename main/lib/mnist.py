"""MNIST data, CNN model, training, evaluation, metrics, and plots.

Use this module in every project that needs the MNIST CNN.
"""
import struct
import time

import matplotlib
matplotlib.use("Agg")  # Use a file-only backend. No display is necessary.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from lib.console import clear_console_line, format_duration, print_progress_bar
from lib.device import get_compute_device

NUMBER_OF_CLASSES = 10
MNIST_PIXEL_MEAN = 0.1307  # Mean of all MNIST training pixels.
MNIST_PIXEL_STANDARD_DEVIATION = 0.3081  # Standard deviation of all MNIST training pixels.


#  data
def read_idx_file(idx_file_path):
    """Read an IDX file and return its content as a numpy array.

    Input:
        idx_file_path (Path): Location of an MNIST IDX file (images or labels).
    Output:
        numpy.ndarray (uint8): Array with the shape that the file header defines.
            Image files give (number_of_images, 28, 28).
            Label files give (number_of_labels,).
    """
    with open(idx_file_path, "rb") as idx_file_handle:
        # The header has 4 bytes: 2 zero bytes, a data type code, and the number of dimensions.
        _, _, _, number_of_dimensions_in_file = struct.unpack(">BBBB", idx_file_handle.read(4))
        # Each dimension size is a 4-byte big-endian integer.
        size_of_each_dimension = struct.unpack(
            ">" + "I" * number_of_dimensions_in_file,
            idx_file_handle.read(4 * number_of_dimensions_in_file))
        raw_data_bytes = idx_file_handle.read()
        return np.frombuffer(raw_data_bytes, dtype=np.uint8).reshape(size_of_each_dimension)


def normalize_images(raw_pixel_images):
    """Scale pixels to [0, 1], standardize them, and add a channel dimension.

    Input:
        raw_pixel_images (numpy.ndarray): Pixel values from 0 to 255.
            Shape: (number_of_images, 28, 28).
    Output:
        torch.Tensor (float32): Standardized images.
            Shape: (number_of_images, 1, 28, 28).
    """
    images_scaled_to_unit_range = torch.tensor(raw_pixel_images, dtype=torch.float32) / 255.0
    images_standardized = (
        (images_scaled_to_unit_range - MNIST_PIXEL_MEAN) / MNIST_PIXEL_STANDARD_DEVIATION)
    return images_standardized.unsqueeze(1)  # Add the channel dimension.


def load_mnist_data(dataset_directory):
    """Load the four MNIST files and return the training and test tensors.

    Input:
        dataset_directory (Path): Folder with the four MNIST IDX files.
    Output:
        tuple of 4 tensors:
            training_images_tensor (float32): Shape (60000, 1, 28, 28).
            training_labels_tensor (int64): Shape (60000,).
            test_images_tensor (float32): Shape (10000, 1, 28, 28).
            test_labels_tensor (int64): Shape (10000,).
    """
    training_images_raw = read_idx_file(dataset_directory / "train-images.idx3-ubyte")
    training_labels_raw = read_idx_file(dataset_directory / "train-labels.idx1-ubyte")
    test_images_raw = read_idx_file(dataset_directory / "t10k-images.idx3-ubyte")
    test_labels_raw = read_idx_file(dataset_directory / "t10k-labels.idx1-ubyte")
    return (
        normalize_images(training_images_raw),
        torch.tensor(training_labels_raw, dtype=torch.long),
        normalize_images(test_images_raw),
        torch.tensor(test_labels_raw, dtype=torch.long),
    )


# Settings of the training augmentation. results.json stores them.
AUGMENTATION_SETTINGS = dict(
    maximum_pixel_shift=2, maximum_rotation_degrees=10.0, scale_range=(0.9, 1.1))


def augment_batch(batch_of_images,
                  maximum_pixel_shift=AUGMENTATION_SETTINGS["maximum_pixel_shift"],
                  maximum_rotation_degrees=AUGMENTATION_SETTINGS["maximum_rotation_degrees"],
                  scale_range=AUGMENTATION_SETTINGS["scale_range"]):
    """Apply a different random shift, rotation, and zoom to each image of a batch.

    All transforms run together on the compute device with one affine grid,
    so the cost is small. Empty space at the border is filled with the
    background value (a black pixel after standardization).

    Inputs:
        batch_of_images (torch.Tensor): Shape (batch_size, 1, 28, 28).
        maximum_pixel_shift (float): Largest shift in pixels for each axis.
        maximum_rotation_degrees (float): Largest rotation in degrees (both directions).
        scale_range (tuple): Smallest and largest zoom factor.
    Output:
        torch.Tensor: Transformed images. The shape is the same as the input.
    """
    number_of_images = batch_of_images.shape[0]
    compute_device = batch_of_images.device
    random_angles = (torch.rand(number_of_images, device=compute_device) * 2 - 1) \
        * maximum_rotation_degrees * np.pi / 180
    random_scales = scale_range[0] + torch.rand(number_of_images, device=compute_device) \
        * (scale_range[1] - scale_range[0])
    # affine_grid uses coordinates from -1 to 1, so one pixel equals 2 / image size.
    random_shifts = (torch.rand(number_of_images, 2, device=compute_device) * 2 - 1) \
        * maximum_pixel_shift * 2 / batch_of_images.shape[-1]
    cosine_values = torch.cos(random_angles) / random_scales
    sine_values = torch.sin(random_angles) / random_scales
    affine_matrices = torch.stack([
        torch.stack([cosine_values, -sine_values, random_shifts[:, 0]], dim=1),
        torch.stack([sine_values, cosine_values, random_shifts[:, 1]], dim=1)], dim=1)
    sampling_grid = functional.affine_grid(
        affine_matrices, list(batch_of_images.shape), align_corners=False)
    background_value = -MNIST_PIXEL_MEAN / MNIST_PIXEL_STANDARD_DEVIATION
    # Subtract the background so the zero padding of grid_sample equals the background.
    return functional.grid_sample(
        batch_of_images - background_value, sampling_grid,
        padding_mode="zeros", align_corners=False) + background_value


#  model
class SqueezeExcitation(nn.Module):
    """Channel attention. It learns one weight from 0 to 1 for each feature map.

    Squeeze: average each feature map to one number.
    Excitation: two small linear layers turn these numbers into channel weights.
    The weights scale the feature maps, so useful channels get more influence.
    """

    def __init__(self, number_of_channels, reduction_factor=8):
        """Create the layers.

        Inputs:
            number_of_channels (int): Channels of the feature maps.
            reduction_factor (int): Size ratio between channels and the bottleneck.
        Output:
            None.
        """
        super().__init__()
        bottleneck_size = max(number_of_channels // reduction_factor, 4)
        self.squeeze_layer = nn.Linear(number_of_channels, bottleneck_size)
        self.excitation_layer = nn.Linear(bottleneck_size, number_of_channels)

    def forward(self, input_feature_maps):
        """Scale each channel by its learned weight.

        Input:
            input_feature_maps (torch.Tensor): Shape (batch, channels, H, W).
        Output:
            torch.Tensor: Same shape as the input.
        """
        channel_summaries = input_feature_maps.mean(dim=(2, 3))  # Shape: (batch, channels).
        channel_weights = torch.sigmoid(self.excitation_layer(
            functional.relu(self.squeeze_layer(channel_summaries))))
        return input_feature_maps * channel_weights[:, :, None, None]


class ConvolutionBlock(nn.Module):
    """Two (convolution, batch normalization) stages with an optional squeeze-and-excitation
    step and an optional residual connection, then ReLU, max pooling, and dropout."""

    def __init__(self, number_of_input_channels, number_of_output_channels,
                 spatial_dropout_probability, use_residual=True, use_squeeze_excitation=True):
        """Create the layers of the block.

        Inputs:
            number_of_input_channels (int): Channels that enter the block.
            number_of_output_channels (int): Channels that each convolution produces.
            spatial_dropout_probability (float): Chance to drop a complete feature map.
            use_residual (bool): Add a 1x1 projected copy of the input to the output
                of the second convolution. This helps the gradient flow.
            use_squeeze_excitation (bool): Apply channel attention after the second convolution.
        Output:
            None.
        """
        super().__init__()
        self.use_residual = use_residual
        self.use_squeeze_excitation = use_squeeze_excitation
        if use_residual:
            # A 1x1 convolution matches the channel count of the skip path.
            self.skip_convolution_layer = nn.Conv2d(
                number_of_input_channels, number_of_output_channels, 1, bias=False)
            self.skip_batch_normalization_layer = nn.BatchNorm2d(number_of_output_channels)
        if use_squeeze_excitation:
            self.squeeze_excitation_layer = SqueezeExcitation(number_of_output_channels)
        self.first_convolution_layer = nn.Conv2d(
            number_of_input_channels, number_of_output_channels, 3, padding=1, bias=False)
        self.first_batch_normalization_layer = nn.BatchNorm2d(number_of_output_channels)
        self.second_convolution_layer = nn.Conv2d(
            number_of_output_channels, number_of_output_channels, 3, padding=1, bias=False)
        self.second_batch_normalization_layer = nn.BatchNorm2d(number_of_output_channels)
        # Dropout2d removes complete feature maps, not single pixels.
        self.spatial_dropout_layer = nn.Dropout2d(spatial_dropout_probability)

    def forward_stages(self, input_feature_maps):
        """Run the block and return the intermediate activations (before dropout).

        Input:
            input_feature_maps (torch.Tensor): Shape (batch, input_channels, H, W).
        Output:
            tuple of 3 tensors: first stage activations (H, W), second stage
            activations (H, W), and pooled activations (H / 2, W / 2).
        """
        first_stage_activations = functional.relu(self.first_batch_normalization_layer(
            self.first_convolution_layer(input_feature_maps)))
        second_stage_pre_activations = self.second_batch_normalization_layer(
            self.second_convolution_layer(first_stage_activations))
        if self.use_squeeze_excitation:
            second_stage_pre_activations = self.squeeze_excitation_layer(
                second_stage_pre_activations)
        if self.use_residual:
            second_stage_pre_activations = second_stage_pre_activations + (
                self.skip_batch_normalization_layer(self.skip_convolution_layer(input_feature_maps)))
        second_stage_activations = functional.relu(second_stage_pre_activations)
        pooled_activations = functional.max_pool2d(second_stage_activations, 2)  # Halve height and width.
        return first_stage_activations, second_stage_activations, pooled_activations

    def forward(self, input_feature_maps):
        """Run the block on a batch of feature maps.

        Input:
            input_feature_maps (torch.Tensor): Shape (batch, input_channels, H, W).
        Output:
            torch.Tensor: Shape (batch, output_channels, H / 2, W / 2).
        """
        return self.spatial_dropout_layer(self.forward_stages(input_feature_maps)[2])


class MnistConvolutionalNetwork(nn.Module):
    """Two residual squeeze-and-excitation convolution blocks, global average pooling,
    and two fully connected layers. Each extra feature can be switched off for ablations."""

    def __init__(self, first_block_channels=32, second_block_channels=64,
                 hidden_units=128, convolution_dropout=0.1, fully_connected_dropout=0.4,
                 use_residual=True, use_squeeze_excitation=True,
                 use_global_average_pooling=True):
        """Create the layers of the network.

        Inputs:
            first_block_channels (int): Channels in the first convolution block.
            second_block_channels (int): Channels in the second convolution block.
            hidden_units (int): Neurons in the hidden fully connected layer.
            convolution_dropout (float): Feature map drop chance in the convolution blocks.
            fully_connected_dropout (float): Neuron drop chance after the hidden layer.
            use_residual (bool): Use residual connections in the convolution blocks.
            use_squeeze_excitation (bool): Use channel attention in the convolution blocks.
            use_global_average_pooling (bool): Average each final feature map to one number
                (fewer parameters). When False, flatten all 7x7 positions instead.
        Output:
            None.
        """
        super().__init__()
        self.use_global_average_pooling = use_global_average_pooling
        self.first_convolution_block = ConvolutionBlock(
            1, first_block_channels, convolution_dropout,
            use_residual, use_squeeze_excitation)  # Image size: 28 -> 14.
        self.second_convolution_block = ConvolutionBlock(
            first_block_channels, second_block_channels, convolution_dropout,
            use_residual, use_squeeze_excitation)  # Image size: 14 -> 7.
        fully_connected_input_size = (
            second_block_channels if use_global_average_pooling
            else second_block_channels * 7 * 7)
        self.hidden_fully_connected_layer = nn.Linear(fully_connected_input_size, hidden_units)
        self.hidden_dropout_layer = nn.Dropout(fully_connected_dropout)
        self.output_fully_connected_layer = nn.Linear(hidden_units, NUMBER_OF_CLASSES)

    def forward(self, batch_of_images):
        """Run the network on a batch of images.

        Input:
            batch_of_images (torch.Tensor): Shape (batch_size, 1, 28, 28).
        Output:
            torch.Tensor: Class scores (logits). Shape (batch_size, 10).
        """
        final_feature_maps = self.second_convolution_block(
            self.first_convolution_block(batch_of_images))
        if self.use_global_average_pooling:
            fully_connected_input = final_feature_maps.mean(dim=(2, 3))  # (batch, channels)
        else:
            fully_connected_input = final_feature_maps.flatten(1)
        hidden_layer_activations = functional.relu(
            self.hidden_fully_connected_layer(fully_connected_input))
        return self.output_fully_connected_layer(
            self.hidden_dropout_layer(hidden_layer_activations))


#  train / evaluate
@torch.no_grad()
def evaluate_model(model_to_evaluate, data_loader_to_evaluate):
    """Measure the loss and the accuracy of a model on a data set.

    Inputs:
        model_to_evaluate (nn.Module): The network to test.
        data_loader_to_evaluate (DataLoader): Batches of (images, labels).
    Output:
        tuple of 4 items:
            mean_loss_per_sample (float): Mean cross-entropy loss.
            accuracy_fraction (float): Fraction of correct predictions, 0 to 1.
            predicted_labels_array (numpy.ndarray): Predicted class for each sample.
            true_labels_array (numpy.ndarray): True class for each sample.
    """
    model_to_evaluate.eval()  # Disable dropout. Use the batch normalization running statistics.
    total_summed_loss = 0.0
    total_correct_prediction_count = 0
    total_sample_count = 0
    list_of_predicted_label_batches = []
    list_of_true_label_batches = []
    for batch_of_images, batch_of_true_labels in data_loader_to_evaluate:
        batch_of_images = batch_of_images.to(get_compute_device())
        batch_of_true_labels = batch_of_true_labels.to(get_compute_device())
        batch_of_logits = model_to_evaluate(batch_of_images)
        # Use the sum here. The final division by the sample count gives the exact mean.
        total_summed_loss += functional.cross_entropy(
            batch_of_logits, batch_of_true_labels, reduction="sum").item()
        batch_of_predicted_labels = batch_of_logits.argmax(1)
        total_correct_prediction_count += (
            batch_of_predicted_labels == batch_of_true_labels).sum().item()
        total_sample_count += len(batch_of_true_labels)
        list_of_predicted_label_batches.append(batch_of_predicted_labels.cpu())
        list_of_true_label_batches.append(batch_of_true_labels.cpu())
    return (total_summed_loss / total_sample_count,
            total_correct_prediction_count / total_sample_count,
            torch.cat(list_of_predicted_label_batches).numpy(),
            torch.cat(list_of_true_label_batches).numpy())


@torch.no_grad()
def evaluate_with_test_time_augmentation(model_to_evaluate, data_loader_to_evaluate,
                                         pixel_shifts=((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))):
    """Evaluate a model with test-time augmentation (TTA).

    Each batch is shifted several times. The softmax probabilities of all
    shifted copies are averaged before the final prediction.

    Inputs:
        model_to_evaluate (nn.Module): The network to test.
        data_loader_to_evaluate (DataLoader): Batches of (images, labels).
        pixel_shifts (tuple): (vertical, horizontal) shifts in pixels.
    Output:
        tuple of 3 items:
            accuracy_fraction (float): Fraction of correct predictions, 0 to 1.
            predicted_labels_array (numpy.ndarray): Predicted class for each sample.
            true_labels_array (numpy.ndarray): True class for each sample.
    """
    model_to_evaluate.eval()
    predicted_batches = []
    true_batches = []
    for batch_of_images, batch_of_true_labels in data_loader_to_evaluate:
        batch_of_images = batch_of_images.to(get_compute_device())
        summed_probabilities = 0
        for vertical_shift, horizontal_shift in pixel_shifts:
            shifted_images = torch.roll(
                batch_of_images, shifts=(vertical_shift, horizontal_shift), dims=(2, 3))
            summed_probabilities = summed_probabilities + functional.softmax(
                model_to_evaluate(shifted_images), dim=1)
        predicted_batches.append(summed_probabilities.argmax(1).cpu())
        true_batches.append(batch_of_true_labels)
    predicted_labels_array = torch.cat(predicted_batches).numpy()
    true_labels_array = torch.cat(true_batches).numpy()
    return (float((predicted_labels_array == true_labels_array).mean()),
            predicted_labels_array, true_labels_array)


def train_model(model_configuration, training_dataset, validation_dataset,
                maximum_number_of_epochs=30, early_stopping_patience_in_epochs=5,
                print_progress=True):
    """Train a network and return the best model and the training history.

    Optimizer: AdamW with weight decay.
    Learning rate schedule: cosine annealing.
    Early stopping: stop when the validation accuracy (ties broken by the lower
    validation loss) does not improve for `early_stopping_patience_in_epochs` epochs.
    Then restore the best weights. Accuracy is used because the validation loss is noisy
    with augmentation and label smoothing, and it stopped training too early before.

    Inputs:
        model_configuration (dict): Hyperparameters. Required keys:
            first_block_channels, second_block_channels, hidden_units,
            convolution_dropout, fully_connected_dropout, learning_rate,
            weight_decay, batch_size. Optional key: use_augmentation.
        training_dataset (Dataset): Training samples (image, label).
        validation_dataset (Dataset): Validation samples (image, label).
        maximum_number_of_epochs (int): Upper limit of training epochs.
        early_stopping_patience_in_epochs (int): Epochs to wait for an improvement.
        print_progress (bool): Print one line for each epoch when True.
    Output:
        tuple of 2 items:
            trained_model (nn.Module): Network with the best validation weights.
            training_history (dict): Lists of train_loss, train_acc, val_loss, val_acc,
                and the integer best_epoch (1-based epoch of the restored weights).
    """
    trained_model = MnistConvolutionalNetwork(
        model_configuration["first_block_channels"],
        model_configuration["second_block_channels"],
        model_configuration["hidden_units"],
        model_configuration["convolution_dropout"],
        model_configuration["fully_connected_dropout"],
        model_configuration.get("use_residual", True),
        model_configuration.get("use_squeeze_excitation", True),
        model_configuration.get("use_global_average_pooling", True),
    ).to(get_compute_device())
    adamw_optimizer = torch.optim.AdamW(
        trained_model.parameters(),
        lr=model_configuration["learning_rate"],
        weight_decay=model_configuration["weight_decay"])
    cosine_learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        adamw_optimizer, T_max=maximum_number_of_epochs)
    training_data_loader = DataLoader(
        training_dataset, batch_size=model_configuration["batch_size"], shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1000)

    training_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_validation_score_so_far = (-1.0, -float("inf"))  # (accuracy, -loss)
    best_epoch_number = 0
    best_model_weights_so_far = None
    number_of_epochs_without_improvement = 0

    if print_progress:
        print(f"  {'Epoch':<7}| {'Train loss':>10} | {'Train acc':>9} | "
              f"{'Val loss':>8} | {'Val acc':>7} | {'Time':>7}")
        print("  " + "-" * 62)
    for current_epoch_number in range(1, maximum_number_of_epochs + 1):
        trained_model.train()  # Enable dropout and batch normalization updates.
        epoch_summed_training_loss = 0.0
        epoch_correct_training_predictions = 0
        epoch_training_sample_count = 0
        epoch_start_time_in_seconds = time.time()
        total_batches_in_epoch = len(training_data_loader)
        for batch_index, (batch_of_images, batch_of_true_labels) in enumerate(training_data_loader):
            batch_of_images = batch_of_images.to(get_compute_device())
            batch_of_true_labels = batch_of_true_labels.to(get_compute_device())
            if model_configuration.get("use_augmentation", True):
                batch_of_images = augment_batch(batch_of_images)
            adamw_optimizer.zero_grad()  # Remove the gradients of the previous batch.
            batch_of_logits = trained_model(batch_of_images)
            # Label smoothing softens the targets. It reduces overconfidence.
            mean_batch_loss = functional.cross_entropy(
                batch_of_logits, batch_of_true_labels,
                label_smoothing=model_configuration.get("label_smoothing", 0.0))
            mean_batch_loss.backward()  # Compute the gradients.
            # Clip the gradient norm. This keeps the updates stable at high learning rates.
            torch.nn.utils.clip_grad_norm_(trained_model.parameters(), max_norm=1.0)
            adamw_optimizer.step()  # Update the weights.
            epoch_summed_training_loss += mean_batch_loss.item() * len(batch_of_true_labels)
            epoch_correct_training_predictions += (
                batch_of_logits.argmax(1) == batch_of_true_labels).sum().item()
            epoch_training_sample_count += len(batch_of_true_labels)
            if print_progress:
                print_progress_bar(
                    batch_index + 1, total_batches_in_epoch, epoch_start_time_in_seconds,
                    f"  epoch {current_epoch_number:02d}/{maximum_number_of_epochs:02d}",
                    keep_line_open=True)
        cosine_learning_rate_scheduler.step()

        epoch_validation_loss, epoch_validation_accuracy, _, _ = evaluate_model(
            trained_model, validation_data_loader)
        epoch_mean_training_loss = epoch_summed_training_loss / epoch_training_sample_count
        epoch_training_accuracy = epoch_correct_training_predictions / epoch_training_sample_count
        for history_key, history_value in zip(
                training_history,
                (epoch_mean_training_loss, epoch_training_accuracy,
                 epoch_validation_loss, epoch_validation_accuracy)):
            training_history[history_key].append(history_value)

        if print_progress:
            epoch_label_text = f"{current_epoch_number:02d}/{maximum_number_of_epochs:02d}"
            epoch_duration_text = format_duration(time.time() - epoch_start_time_in_seconds)
            clear_console_line()  # Replace the progress bar with the summary row.
            print(f"  {epoch_label_text:<7}| {epoch_mean_training_loss:>10.4f} | "
                  f"{epoch_training_accuracy:>9.2%} | {epoch_validation_loss:>8.4f} | "
                  f"{epoch_validation_accuracy:>7.2%} | {epoch_duration_text:>7}")

        if (epoch_validation_accuracy, -epoch_validation_loss) > best_validation_score_so_far:
            best_validation_score_so_far = (epoch_validation_accuracy, -epoch_validation_loss)
            best_epoch_number = current_epoch_number
            number_of_epochs_without_improvement = 0
            # Copy the weights. A reference changes when training continues.
            best_model_weights_so_far = {
                parameter_name: parameter_tensor.clone()
                for parameter_name, parameter_tensor in trained_model.state_dict().items()}
        else:
            number_of_epochs_without_improvement += 1
            if number_of_epochs_without_improvement >= early_stopping_patience_in_epochs:
                if print_progress:
                    print(f"  Early stopping: no validation improvement for "
                          f"{early_stopping_patience_in_epochs} epochs. Best weights restored.")
                break

    trained_model.load_state_dict(best_model_weights_so_far)
    training_history["best_epoch"] = best_epoch_number
    return trained_model, training_history


#  plots
def plot_training_history(training_history, output_image_path):
    """Plot the loss and the accuracy for each epoch and save the figure.

    Inputs:
        training_history (dict): Lists of train_loss, train_acc, val_loss, val_acc.
        output_image_path (Path): Location of the PNG file to write.
    Output:
        None. The function writes one image file.
    """
    history_figure, history_axes = plt.subplots(1, 2, figsize=(10, 4))
    epoch_numbers = range(1, len(training_history["train_loss"]) + 1)
    history_axes[0].plot(epoch_numbers, training_history["train_loss"], label="train")
    history_axes[0].plot(epoch_numbers, training_history["val_loss"], label="validation")
    history_axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    history_axes[1].plot(epoch_numbers, training_history["train_acc"], label="train")
    history_axes[1].plot(epoch_numbers, training_history["val_acc"], label="validation")
    history_axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for single_axis in history_axes:
        single_axis.legend()
        single_axis.grid(alpha=0.3)
    history_figure.tight_layout()
    history_figure.savefig(output_image_path, dpi=150)
    plt.close(history_figure)


def compute_confusion_matrix(true_labels_array, predicted_labels_array):
    """Count the predictions for each pair of true class and predicted class.

    Inputs:
        true_labels_array (numpy.ndarray): True class for each sample.
        predicted_labels_array (numpy.ndarray): Predicted class for each sample.
    Output:
        numpy.ndarray (int): Matrix of shape (10, 10).
            Rows are true classes. Columns are predicted classes.
    """
    confusion_count_matrix = np.zeros((NUMBER_OF_CLASSES, NUMBER_OF_CLASSES), dtype=int)
    # add.at counts repeated index pairs correctly.
    np.add.at(confusion_count_matrix, (true_labels_array, predicted_labels_array), 1)
    return confusion_count_matrix


def plot_confusion_matrix(confusion_count_matrix, output_image_path):
    """Draw the confusion matrix with the count in each cell and save the figure.

    Inputs:
        confusion_count_matrix (numpy.ndarray): Matrix of shape (10, 10).
        output_image_path (Path): Location of the PNG file to write.
    Output:
        None. The function writes one image file.
    """
    matrix_figure, matrix_axis = plt.subplots(figsize=(6, 5))
    matrix_image_handle = matrix_axis.imshow(confusion_count_matrix, cmap="Blues")
    matrix_axis.set(xlabel="Predicted", ylabel="True", title="Confusion matrix (test set)",
                    xticks=range(NUMBER_OF_CLASSES), yticks=range(NUMBER_OF_CLASSES))
    for true_class_index in range(NUMBER_OF_CLASSES):
        for predicted_class_index in range(NUMBER_OF_CLASSES):
            cell_count = confusion_count_matrix[true_class_index, predicted_class_index]
            # Use white text on dark cells so the number stays readable.
            cell_text_color = "white" if cell_count > confusion_count_matrix.max() / 2 else "black"
            matrix_axis.text(predicted_class_index, true_class_index, cell_count,
                             ha="center", va="center", fontsize=7, color=cell_text_color)
    matrix_figure.colorbar(matrix_image_handle)
    matrix_figure.tight_layout()
    matrix_figure.savefig(output_image_path, dpi=150)
    plt.close(matrix_figure)


def plot_first_layer_filters(trained_model, output_image_path):
    """Draw the 3x3 filters of the first convolution layer and save the figure.

    Inputs:
        trained_model (MnistConvolutionalNetwork): Network with trained weights.
        output_image_path (Path): Location of the PNG file to write.
    Output:
        None. The function writes one image file.
    """
    first_layer_filter_weights = (
        trained_model.first_convolution_block.first_convolution_layer
        .weight.detach().cpu().numpy())  # Shape: (channels, 1, 3, 3).
    filters_figure, filters_axes = plt.subplots(4, 8, figsize=(8, 4.4))
    for single_axis, single_filter_weights in zip(filters_axes.flat, first_layer_filter_weights):
        single_axis.imshow(single_filter_weights[0], cmap="gray")
        single_axis.axis("off")
    filters_figure.suptitle("First-layer 3x3 convolution filters")
    filters_figure.tight_layout()
    filters_figure.savefig(output_image_path, dpi=150)
    plt.close(filters_figure)


@torch.no_grad()
def plot_feature_maps(trained_model, single_input_image, output_image_path):
    """Draw the first 8 channels of three convolution layers for one image.

    Inputs:
        trained_model (MnistConvolutionalNetwork): Network with trained weights.
        single_input_image (torch.Tensor): One image. Shape (1, 28, 28).
        output_image_path (Path): Location of the PNG file to write.
    Output:
        None. The function writes one image file.
    """
    trained_model.eval()
    batch_with_one_image = single_input_image.unsqueeze(0).to(get_compute_device())
    first_block = trained_model.first_convolution_block
    second_block = trained_model.second_convolution_block
    # forward_stages returns the intermediate activations of a block.
    (first_convolution_feature_maps, second_convolution_feature_maps,
     pooled_feature_maps) = first_block.forward_stages(batch_with_one_image)
    third_convolution_feature_maps = second_block.forward_stages(pooled_feature_maps)[0]

    figure_rows = [("Input", None),
                   ("Conv 1_1", first_convolution_feature_maps),
                   ("Conv 1_2 (+SE, +skip)", second_convolution_feature_maps),
                   ("Conv 2_1", third_convolution_feature_maps)]
    maps_figure, maps_axes = plt.subplots(4, 8, figsize=(10, 5.6))
    for row_index, (row_title, row_feature_maps) in enumerate(figure_rows):
        for channel_index in range(8):
            single_axis = maps_axes[row_index, channel_index]
            single_axis.axis("off")
            if row_feature_maps is None:
                if channel_index == 0:  # Show the input image one time only.
                    single_axis.imshow(batch_with_one_image[0, 0].cpu(), cmap="gray")
            else:
                single_axis.imshow(row_feature_maps[0, channel_index].cpu(), cmap="viridis")
            if channel_index == 0:
                single_axis.set_title(row_title, fontsize=8, loc="left")
    maps_figure.suptitle("Feature maps for one test digit (first 8 channels per layer)")
    maps_figure.tight_layout()
    maps_figure.savefig(output_image_path, dpi=150)
    plt.close(maps_figure)


def plot_misclassified_images(test_images_array, true_labels_array, predicted_labels_array,
                              output_image_path, maximum_number_of_images=16):
    """Draw the first misclassified test images with true and predicted labels.

    Inputs:
        test_images_array (numpy.ndarray): All test images. Shape (N, 1, 28, 28).
        true_labels_array (numpy.ndarray): True class for each image.
        predicted_labels_array (numpy.ndarray): Predicted class for each image.
        output_image_path (Path): Location of the PNG file to write.
        maximum_number_of_images (int): Largest number of errors to draw.
    Output:
        None. The function writes one image file.
    """
    indices_of_misclassified_images = np.where(
        true_labels_array != predicted_labels_array)[0][:maximum_number_of_images]
    errors_figure, errors_axes = plt.subplots(2, 8, figsize=(10, 3))
    for single_axis in errors_axes.flat:
        single_axis.axis("off")
    for single_axis, misclassified_image_index in zip(errors_axes.flat,
                                                      indices_of_misclassified_images):
        single_axis.imshow(test_images_array[misclassified_image_index, 0], cmap="gray")
        single_axis.set_title(
            f"T:{true_labels_array[misclassified_image_index]} "
            f"P:{predicted_labels_array[misclassified_image_index]}", fontsize=8)
    errors_figure.suptitle("Misclassified test digits (T = true, P = predicted)")
    errors_figure.tight_layout()
    errors_figure.savefig(output_image_path, dpi=150)
    plt.close(errors_figure)


#  extra metrics, misclassification data, and ensembles
@torch.no_grad()
def collect_softmax_probabilities(model_to_evaluate, data_loader_to_evaluate):
    """Return the class probabilities of a model for every sample of a data set.

    Inputs:
        model_to_evaluate (nn.Module): The network to test.
        data_loader_to_evaluate (DataLoader): Batches of (images, labels).
            The loader must not shuffle, so row i belongs to sample i.
    Output:
        numpy.ndarray (float64): Shape (number_of_samples, 10). Each row sums to 1.
    """
    model_to_evaluate.eval()  # Disable dropout. Use the batch normalization running statistics.
    list_of_probability_batches = []
    for batch_of_images, _ in data_loader_to_evaluate:
        batch_of_logits = model_to_evaluate(batch_of_images.to(get_compute_device()))
        list_of_probability_batches.append(functional.softmax(batch_of_logits, dim=1).cpu())
    return torch.cat(list_of_probability_batches).double().numpy()


def compute_expected_calibration_error(probabilities, true_labels, number_of_bins=15):
    """Measure how well the confidence of a model matches its accuracy.

    The predictions are sorted into bins by confidence (the highest probability).
    In each bin the function compares the accuracy with the mean confidence.
    The result is the average gap, weighted by the number of samples in each bin.
    A perfectly calibrated model gives 0.

    Inputs:
        probabilities (numpy.ndarray): Shape (number_of_samples, 10).
        true_labels (numpy.ndarray): True class for each sample.
        number_of_bins (int): Number of equal-width confidence bins.
    Output:
        float: Expected calibration error, 0 to 1.
    """
    confidence_of_each_prediction = probabilities.max(axis=1)
    prediction_is_correct = probabilities.argmax(axis=1) == true_labels
    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    expected_calibration_error = 0.0
    for lower_edge, upper_edge in zip(bin_edges[:-1], bin_edges[1:]):
        sample_is_in_bin = (confidence_of_each_prediction > lower_edge) & (
            confidence_of_each_prediction <= upper_edge)
        if sample_is_in_bin.any():
            bin_accuracy = prediction_is_correct[sample_is_in_bin].mean()
            bin_confidence = confidence_of_each_prediction[sample_is_in_bin].mean()
            expected_calibration_error += sample_is_in_bin.mean() * abs(
                bin_accuracy - bin_confidence)
    return float(expected_calibration_error)


def compute_wilson_confidence_interval(success_count, total_count, z_score=1.96):
    """Compute the Wilson score interval of a proportion (95% for z = 1.96).

    This interval behaves better than the normal approximation when the
    proportion is close to 1, as it is for the MNIST accuracy.

    Inputs:
        success_count (int): Number of correct predictions.
        total_count (int): Number of predictions.
        z_score (float): Normal quantile of the confidence level.
    Output:
        tuple of 2 floats: Lower and upper bound of the proportion.
    """
    observed_proportion = success_count / total_count
    denominator = 1 + z_score ** 2 / total_count
    interval_center = (observed_proportion + z_score ** 2 / (2 * total_count)) / denominator
    interval_half_width = z_score * np.sqrt(
        observed_proportion * (1 - observed_proportion) / total_count
        + z_score ** 2 / (4 * total_count ** 2)) / denominator
    return float(interval_center - interval_half_width), float(interval_center + interval_half_width)


def summarize_prediction_quality(probabilities, true_labels):
    """Compute the extra metrics for a set of predictions (one model or an ensemble).

    Inputs:
        probabilities (numpy.ndarray): Class probabilities. Shape (number_of_samples, 10).
        true_labels (numpy.ndarray): True class for each sample.
    Output:
        dict with the keys accuracy, errors, accuracy_95_confidence_interval (Wilson),
        top2_accuracy, macro_precision, macro_recall, macro_f1 (the unweighted mean
        over the classes), log_loss, expected_calibration_error, and mean_confidence.
    """
    predicted_labels = probabilities.argmax(axis=1)
    sample_count = len(true_labels)
    correct_prediction_count = int((predicted_labels == true_labels).sum())
    metrics_for_each_class = compute_per_class_metrics(
        compute_confusion_matrix(true_labels, predicted_labels))
    probability_of_true_class = probabilities[np.arange(sample_count), true_labels]
    two_most_likely_classes = np.argsort(probabilities, axis=1)[:, -2:]
    true_class_is_in_top_two = (two_most_likely_classes == true_labels[:, None]).any(axis=1)
    interval_lower_bound, interval_upper_bound = compute_wilson_confidence_interval(
        correct_prediction_count, sample_count)
    return dict(
        accuracy=correct_prediction_count / sample_count,
        errors=sample_count - correct_prediction_count,
        accuracy_95_confidence_interval=[interval_lower_bound, interval_upper_bound],
        top2_accuracy=float(true_class_is_in_top_two.mean()),
        macro_precision=float(np.mean([m["precision"] for m in metrics_for_each_class.values()])),
        macro_recall=float(np.mean([m["recall"] for m in metrics_for_each_class.values()])),
        macro_f1=float(np.mean([m["f1"] for m in metrics_for_each_class.values()])),
        # The clip prevents log(0) when a model gives the true class a probability of 0.
        log_loss=float(-np.log(np.clip(probability_of_true_class, 1e-12, 1.0)).mean()),
        expected_calibration_error=compute_expected_calibration_error(probabilities, true_labels),
        mean_confidence=float(probabilities.max(axis=1).mean()))


def describe_misclassified_samples(probabilities, true_labels):
    """List every misclassified sample with the information needed for error analysis.

    Inputs:
        probabilities (numpy.ndarray): Class probabilities. Shape (number_of_samples, 10).
        true_labels (numpy.ndarray): True class for each sample.
    Output:
        list of dict: One entry for each error, the most confident errors first.
            Keys: test_index (position in the test set), true_label, predicted_label,
            confidence (probability of the predicted class), and
            true_class_probability (probability that the model gave the true class).
    """
    predicted_labels = probabilities.argmax(axis=1)
    list_of_samples = []
    for test_index in np.where(predicted_labels != true_labels)[0]:
        list_of_samples.append(dict(
            test_index=int(test_index),
            true_label=int(true_labels[test_index]),
            predicted_label=int(predicted_labels[test_index]),
            confidence=float(probabilities[test_index, predicted_labels[test_index]]),
            true_class_probability=float(probabilities[test_index, true_labels[test_index]])))
    list_of_samples.sort(key=lambda sample: sample["confidence"], reverse=True)
    return list_of_samples


#  main
def find_most_confused_class_pairs(confusion_count_matrix, number_of_pairs=5):
    """List the (true, predicted) class pairs with the most errors.

    Input:
        confusion_count_matrix (numpy.ndarray): Matrix of shape (10, 10).
        number_of_pairs (int): How many pairs to return.
    Output:
        list of dict: Keys true_class, predicted_class, count. Largest count first.
    """
    error_counts = confusion_count_matrix.copy()
    np.fill_diagonal(error_counts, 0)  # Ignore the correct predictions.
    flat_indices_by_count = np.argsort(error_counts, axis=None)[::-1][:number_of_pairs]
    return [dict(true_class=int(index // NUMBER_OF_CLASSES),
                 predicted_class=int(index % NUMBER_OF_CLASSES),
                 count=int(error_counts.flat[index])) for index in flat_indices_by_count]


def compute_per_class_metrics(confusion_count_matrix):
    """Compute precision, recall, F1 score, and support for each class.

    Input:
        confusion_count_matrix (numpy.ndarray): Matrix of shape (10, 10).
            Rows are true classes. Columns are predicted classes.
    Output:
        dict: Class index (as text) -> dict with the keys
            precision, recall, f1, and support.
    """
    metrics_for_each_class = {}
    for class_index in range(NUMBER_OF_CLASSES):
        true_positive_count = confusion_count_matrix[class_index, class_index]
        # The maximum with 1 prevents a division by zero.
        total_predicted_as_this_class = max(confusion_count_matrix[:, class_index].sum(), 1)
        total_actually_this_class = max(confusion_count_matrix[class_index].sum(), 1)
        class_precision = true_positive_count / total_predicted_as_this_class
        class_recall = true_positive_count / total_actually_this_class
        class_f1_score = (2 * class_precision * class_recall
                          / max(class_precision + class_recall, 1e-12))
        metrics_for_each_class[str(class_index)] = dict(
            precision=class_precision, recall=class_recall, f1=class_f1_score,
            support=int(confusion_count_matrix[class_index].sum()))
    return metrics_for_each_class
