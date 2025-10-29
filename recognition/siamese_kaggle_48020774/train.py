from keras import callbacks, Model
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import os
import numpy as np
from config import Config

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
    test_ds = dataset.GenerateTestSet()
    print("Generating testing data...")
    train_ds = dataset.GenerateTrainSet()
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        # Use the actual batch size from the data
        epochs=epochs,
        steps_per_epoch=2,
        validation_data=test_ds,
        validation_steps=10,
        callbacks=[save_callback]
    )
    print("Recording history...")
    
    # Add debugging to see what metrics are available
    print("Available metrics in history:", list(history.history.keys()))


    plt.figure("Triplet loss vs Epoch")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Triplet Loss subplot
    axes[0].plot(history.history['triplet_loss'], label='training loss')
    axes[0].plot(history.history['val_triplet_loss'], label='validation loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Triplet Loss')
    axes[0].set_yscale('log')
    min_loss = np.min([tf.reduce_min(history.history['triplet_loss']), tf.reduce_min(history.history['val_triplet_loss'])])
    max_loss = np.max([tf.reduce_max(history.history['triplet_loss']), tf.reduce_max(history.history['val_triplet_loss'])])
    axes[0].set_ylim([min_loss, max_loss])
    axes[0].legend(loc='lower right')
    axes[0].set_title('Triplet Loss')
    
    # AUCROC subplot
    axes[1].plot(history.history['AUCROC'], label='training AUCROC')
    axes[1].plot(history.history['val_AUCROC'], label='validation AUCROC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUCROC')
    axes[1].legend(loc='lower right')
    axes[1].set_title('AUCROC')
    
    # Accuracy subplot
    axes[2].plot(history.history['accuracy'], label='training accuracy')
    axes[2].plot(history.history['val_accuracy'], label='validation accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend(loc='lower right')
    axes[2].set_title('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'training_plots.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("Training complete.")

def validate(model, data : dataset.Dataset):
    test_ds = data.GenerateTestSet()

    # Get model predictions for the validation set
    outputs = model.predict(test_ds)
   