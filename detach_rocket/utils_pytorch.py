"""
Utility functions for Pytorch Detach-ROCKET model.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import numpy as np

# Define device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=20, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: 
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def compute_accuracy(model, train_data, test_data, batch_size=256):
    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=False)

    # Get train and test labels
    y_train = train_data.tensors[1].detach().numpy()
    y_test = test_data.tensors[1].detach().numpy()

    # Initialize lists for train and test predictions
    train_predictions = []
    test_predictions = []

    # Compute train predictions
    model.eval()
    # If it is a binary classification problem, convert probabilities to class labels
    if len(np.unique(y_train)) == 2:
        for data, target in train_loader:
            output = model(data.float())
            output = torch.sigmoid(output)
            output = output.detach().numpy()
            # Convert probabilities to class labels
            output = np.where(output > 0.5, 1, 0)
            train_predictions.extend(output)
    # If it is a multiclass classification problem, select class with maximum probability
    else:
        for data, target in train_loader:
            output = model(data.float())
            output = torch.sigmoid(output)
            output = output.detach().numpy()
            # Select class with maximum probability
            output = np.argmax(output, axis=1)
            # Convert probabilities to class labels
            train_predictions.extend(output)

    # Compute test predictions 
    model.eval()
    # If it is a binary classification problem, convert probabilities to class labels
    if len(np.unique(y_train)) == 2:
        for data, target in test_loader:
            output = model(data.float())
            output = torch.sigmoid(output)
            output = output.detach().numpy()
            # Convert probabilities to class labels
            output = np.where(output > 0.5, 1, 0)
            test_predictions.extend(output)
    # If it is a multiclass classification problem, select class with maximum probability
    else:
        for data, target in test_loader:
            output = model(data.float())
            output = torch.sigmoid(output)
            output = output.detach().numpy()
            # Select class with maximum probability
            output = np.argmax(output, axis=1)
            # Convert probabilities to class labels
            test_predictions.extend(output)

    # Compute accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return train_accuracy, test_accuracy


def get_feature_importance(model, multilabel_type="max"):
    """
    Compute feature importance of a trained model.

    Parameters:
    - model: Trained model.
    - multilabel_type: Type of feature ranking in case of multilabel classification ("max" by default).

    Returns:
    - feature_importance: Feature importance vector.
    """
    
    # Get last layer weights
    last_layer_weights = model.head[3].weight.detach().numpy()

    # Check if it is a binary or multiclass classification problem looking at number of dimensions of last layer weights
    if len(last_layer_weights.shape) > 1:
        if multilabel_type == "norm":
            feature_importance   = np.linalg.norm(last_layer_weights,axis=0,ord=2)
        elif multilabel_type == "max":
            feature_importance  = np.linalg.norm(last_layer_weights,axis=0,ord=np.inf)
        elif multilabel_type == "avg":
            feature_importance= np.linalg.norm(last_layer_weights,axis=0,ord=1)
        else:
            raise ValueError('Invalid multilabel_type argument. Choose from: "norm", "max", or "avg".')
    else:
        feature_importance = np.abs(last_layer_weights)
    return feature_importance


# Define train function for Pytorch model with epochs and early stopping
def train_model(model, train_data, test_data, n_epochs = 100, batch_size=256, patience=20, L2_regularization_weight = 1e-3, verbose=True):
    """
    Train a Pytorch model with epochs and early stopping.

    Parameters:
    - model: Pytorch model.
    - train_data: Training data. It should be a Pytorch Dataset.
    - test_data: Test data. It should be a Pytorch Dataset.
    - n_epochs: Number of epochs.
    - batch_size: Batch size.
    - patience: Patience for early stopping.
    - L2_regularization_weight: L2 regularization weight.

    Returns:
    - train_losses: List with train losses.
    - test_losses: List with test losses.
    """

    # Define early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Compute model output dimension
    c_out = model.head[3].weight.shape[0]

    # Move model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device)

    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=False)

    # Define loss function and optimizer (for multi-class classification)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=L2_regularization_weight)

    # Initialize lists for train and test losses
    train_losses = []
    test_losses = []

    # Defined number of total epochs trained
    total_epochs = n_epochs

    # Train model
    for epoch in range(n_epochs):
        # Initialize variables to monitor training and test loss
        train_loss = 0.0
        test_loss = 0.0
        # Train model
        model.train()
        for data, target in train_loader:
            # Move data and target to device
            data, target = data.to(device), target.to(device)
            # Clear the gradients of all optimized variables
            optimizer.zero_grad()
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())
            # Calculate the batch loss
            target = target.long()
            # If not multi-class classification, target needs to be reshaped
            if c_out == 1:
                target = target.unsqueeze(1)
            #print('output: ',output.shape)
            #print('target: ',target.shape)
            loss = criterion(output, target)
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Perform a single optimization step (parameter update)
            optimizer.step()
            # Update training loss
            train_loss += loss.item()*data.size(0)
        # Validate model
        model.eval()
        for data, target in test_loader:
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())
            # Calculate the batch loss
            target = target.long()
            if c_out == 1:
                target = target.unsqueeze(1)
            loss = criterion(output, target)
            # Update average test loss 
            test_loss += loss.item()*data.size(0)

        # Calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        test_loss = test_loss/len(test_loader.dataset)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Print training and test results
        if verbose:
            print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {test_loss:.6f}')

        # Early stop
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            # Print early stopping at iteration number
            print("Early stopping at iteration number: ", epoch+1)
            # Total number of epochs trained
            total_epochs = epoch+1
            break
        
    return train_losses, test_losses, total_epochs


def prune_model_SFD(model, drop_percentage=0.05, multilabel_type="max", verbose=True):
    """
    Prune a trained model using the SFD method.

    Parameters:
    - model: Trained model.
    - drop_percentage: Percentage of features to drop.
    - multilabel_type: Type of feature ranking in case of multilabel classification ("max" by default).
    - verbose: If True, print information about the pruning process.

    Returns:
    - model: Pruned model.
    - feature_importance: Feature importance vector.
    """

    # Compute feature importance
    feature_importance = get_feature_importance(model, multilabel_type=multilabel_type)

    # Compute number of features
    n_features = len(feature_importance)

    # Compute number of non-zero features
    n_nonzero_features = np.count_nonzero(feature_importance)

    # Compute number of zero features
    n_zero_features = n_features - n_nonzero_features

    # Compute number of features to drop
    n_features_to_drop = int(np.floor(drop_percentage*n_nonzero_features)) + n_zero_features

    # Create mask of features to drop
    feature_mask = np.ones(len(feature_importance))
    feature_mask[np.argsort(feature_importance)[:n_features_to_drop]] = 0

    if verbose:
        # Print number of features to drop
        print(f"Number of features pruned: {n_features_to_drop}")
        # Print number of features to keep
        print(f"Number of features kept: {int(np.sum(feature_mask))}")

    ## UPDATE MASK LAYER -----------------------------

    # Load feature mask as parameters of mask layer
    model.head[2].weight.data = torch.nn.Parameter(torch.from_numpy(feature_mask).float())

    ## UPDATE FEATURES WEIGHTS -----------------------------
    # (send to zero weights of dropped features)

    # Create mask of features to drop
    weight_feature_mask = np.zeros(len(feature_importance), dtype=bool)
    weight_feature_mask[np.argsort(feature_importance)[:n_features_to_drop]] = True

    # Compute model output dimension
    c_out = model.head[3].weight.shape[0]

    # Create a weights mask - True if we will drop the weight (shape: (c_out,n_features))
    weight_matrix_mask = np.zeros((c_out,len(feature_importance)), dtype=bool)
    for i in range(c_out):
        weight_matrix_mask[i,:] = weight_feature_mask

    # Create weight matrix sending to zero weights of dropped features
    weights_matrix = np.ones((c_out,len(feature_importance)))
    weights_matrix[weight_matrix_mask] = 0
    weights_matrix = np.multiply(model.head[3].weight.detach().numpy(),weights_matrix)

    if verbose:
        # Print amount of zeros in weight matrix
        print(f"Percentage of zeros in weight matrix: {np.sum(weights_matrix == 0)/(c_out*n_features)} %")

    # Convert weights_matrix to torch torch.nn.parameter 
    weights_matrix = torch.from_numpy(weights_matrix)
    weights_matrix = torch.nn.Parameter(weights_matrix.float())

    # Load weights matrix as parameters of last layer
    model.head[3].weight = weights_matrix

    return feature_importance
