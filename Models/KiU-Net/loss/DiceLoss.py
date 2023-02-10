import numpy as np
import torch

def dice_loss(y_true, y_pred):
    # if type(y_true).__module__ != np.__name__:
    #     y_true = y_true.cpu().detach().numpy()
    #     y_pred = torch.round(y_pred) # Just only for binary segmentation with 1 output channel
    #     y_pred = y_pred.cpu().detach().numpy()
    if type(y_true).__module__ != np.__name__:
      y_true = y_true.cpu().detach().numpy()
      y_pred = torch.argmax(y_pred, dim=1) # Cross-entropy loss (>= 2 channels)
      y_pred = y_pred.cpu().detach().numpy()

    # Find the unique labels in each matrix
    labels = [0,1,2]
    dice_loss = {}
    
    # Iterate over the labels
    for label in labels:

        # Find the indices of the label in each matrix
        y_true_indices = y_true == label
        y_pred_indices = y_pred == label

        # If at least one element is True
        if y_true_indices.any():

            # Calculate the number of true positives for the label
            true_positives = np.sum(np.logical_and(y_true_indices, y_pred_indices))

            # Update the numerator and denominator
            numerator = (2 * true_positives) 
            denominator = (np.sum(y_true_indices) + np.sum(y_pred_indices)) 

            # Calculate the Dice loss
            dice_loss[str(label)] = round(1 - numerator/denominator,3)

        elif not y_pred_indices.any() and not y_true_indices.any(): # All labels in prediction and ground truth are false
            dice_loss[str(label)] = "No label gt and pred"

        elif not y_pred_indices.any():  # All labels in prediction are false
            dice_loss[str(label)] = "No label pred"

        elif not y_true_indices.any():  # All labels in ground truth are false
            dice_loss[str(label)] = "No label gt"

    return dice_loss
