from keras import callbacks, Model
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import os
import numpy as np
from config import Config
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns

config = Config.getInstance()

__save_path = config["model_save_path"]

save_callback = callbacks.ModelCheckpoint(
    filepath=__save_path,
    save_weights_only=True,
    verbose=1
)

# Add learning rate scheduler that decreases LR slightly each epoch
lr_scheduler = callbacks.LearningRateScheduler(
    lambda epoch: config['learning_rate'] * (0.93 ** epoch),
    verbose=1
)

# Add early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    monitor='val_AUCROC',  # Monitor validation AUCROC instead of loss
    patience=5,  # Reduced from 8 for earlier stopping
    restore_best_weights=True,
    verbose=1,
    mode='max'  # Changed to max since we want higher AUCROC
)

def load_weights(model):
    if not os.path.exists(__save_path):
        print(f"No weights found at {__save_path}")
        return
    model.load_weights(__save_path)

def train(model : Model, dataset : dataset.Dataset, epochs : int):
    print("Generating training data...")
    train_ds = dataset.GenerateTrainSet()
    print("Generating testing data...")
    test_ds = dataset.GenerateTestSet()
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        # Use the actual batch size from the data
        epochs=epochs,
        steps_per_epoch=100,  # Increased from 60 for better learning
        validation_data=test_ds,
        validation_steps=50,  # Increased proportionally
        callbacks=[save_callback, lr_scheduler, early_stopping]
    )
    print("Recording history...")
    
    # Save training history to file
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'training_history.csv', index=False)
    print(f"History saved to training_history.csv")
    
    # Add debugging to see what metrics are available
    print("Available metrics in history:", list(history.history.keys()))


    plt.figure("Triplet loss vs Epoch")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Triplet Loss subplot
    if ('triplet_loss' in history.history):
        axes[0].plot(history.history['triplet_loss'], label='training loss')
    if ('val_triplet_loss' in history.history):
        axes[0].plot(history.history['val_triplet_loss'], label='validation loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Triplet Loss')
    axes[0].set_yscale('log')
    min_loss = 1e+10
    if ('triplet_loss' in history.history):
        min_loss = min(min_loss, np.min(history.history['triplet_loss']))
    if ('val_triplet_loss' in history.history):
        min_loss = min(min_loss, np.min(history.history['val_triplet_loss']))
    max_loss = -1e+10
    if ('triplet_loss' in history.history):
        max_loss = max(max_loss, np.max(history.history['triplet_loss']))
    if ('val_triplet_loss' in history.history):
        max_loss = max(max_loss, np.max(history.history['val_triplet_loss']))
    axes[0].set_ylim([min_loss, max_loss])
    axes[0].legend(loc='lower right')
    axes[0].set_title('Triplet Loss')
    
    # AUCROC subplot
    if ('AUCROC' in history.history):
        axes[1].plot(history.history['AUCROC'], label='training AUCROC')
    if ('val_AUCROC' in history.history):
        axes[1].plot(history.history['val_AUCROC'], label='validation AUCROC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUCROC')
    axes[1].legend(loc='lower right')
    axes[1].set_title('AUCROC')
    
    # Accuracy subplot
    if ('accuracy' in history.history):
        axes[2].plot(history.history['accuracy'], label='training accuracy')
    if ('val_accuracy' in history.history):
        axes[2].plot(history.history['val_accuracy'], label='validation accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend(loc='lower right')
    axes[2].set_title('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'training_plots.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("Training complete.")

def validate(model: Model, data : dataset.Dataset):
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    test_ds = data.GenerateTestSet()

    # Collect predictions and labels from the validation set
    predictions = []
    labels = []
    
    for batch_idx, (inputs, targets) in enumerate(test_ds.take(50)):  # Take 50 batches for validation
        # inputs is ((anchor, label, positive, negative), label)
        batch_predictions = model.predict_on_batch(inputs)
        batch_labels = targets.numpy()
        
        predictions.extend(batch_predictions.flatten())
        labels.extend(batch_labels.flatten())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Convert predictions to binary (0 or 1)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Create confusion matrix
    cm = confusion_matrix(labels, binary_predictions)
    class_labels = ['benign', 'malignant']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create ROC curve
    fpr, tpr, _ = roc_curve(labels, predictions)
    auc_score = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("plots/roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation complete. AUC: {auc_score:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {np.sum(binary_predictions == labels) / len(labels):.3f}")
   