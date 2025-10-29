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
        steps_per_epoch=30,
        validation_data=test_ds,
        validation_steps=15,
        callbacks=[save_callback]
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
    test_ds = data.GenerateTestSet()

    # Get model predictions for the validation set
    outputs = model.predict(test_ds)
    
    
    labels = test_ds.unbatch().map(lambda x, y: x).map(lambda a,y,p,n: y)
    
    cm = confusion_matrix(labels, outputs)
    class_labels = ['benign', 'malignant']
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(labels, outputs)
    
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("plots/roc.png")
    plt.close()
   