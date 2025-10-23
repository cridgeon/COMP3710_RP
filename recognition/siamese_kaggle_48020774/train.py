from keras import callbacks, Model
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import dataset
import os
import numpy as np

config = json.load(open('recognition/siamese_kaggle_48020774/utility/config.json'))

save_path = config["model_save_path"]
epochs = config["epochs"]
batch_size = config["batch_size"]

save_callback = callbacks.ModelCheckpoint(
    filepath=save_path,
    save_weights_only=True,
    verbose=1
)

def load_weights(model):
    if not os.path.exists(save_path):
        print(f"No weights found at {save_path}")
        return
    model.load_weights(save_path)

def train(model, dataset, rounds = 10, setSize=100):


    for i in range(rounds):
        print(f"--- Training round {i+1}/{rounds} ---")
        print("Generating training data...")
        X_train, Y_train, P_train, N_train = dataset.GenerateTrainSet(setSize)
        print("Generating testing data...")
        X_test, Y_test, P_test, N_test = dataset.GenerateTestSet(int(setSize / 6))
        print("Starting training...")
        history = model.fit(
            {
                "input_anchor": X_train,
                "input_label": Y_train,
                "input_positive": P_train,
                "input_negative": N_train
            },
            epochs=epochs,
            batch_size=batch_size,
            validation_data={
                "input_anchor": X_test,
                "input_label": Y_test,
                "input_positive": P_test,
                "input_negative": N_test
            },
            callbacks=[save_callback]
        )
        print("Recording history...")


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
        plt.savefig(f'training_plots_round_{i+1}.png', dpi=300, bbox_inches='tight')
        # plt.show()

def validate(model, validate_data):
    X_val, Y_val, P_val, N_val = validate_data

    # Get model predictions for the validation set
    outputs = model.predict({
        "input_anchor": X_val,
        "input_label": Y_val,
        "input_positive": P_val,
        "input_negative": N_val
    })

    print("Raw outputs:", outputs.shape)

    outputs = tf.reshape(outputs, [-1])
    total = outputs.shape[0]
    correct = 0
    for i in range(total):
        if outputs[i] == (Y_val[i] > 0):
            correct += 1
    # outputs_int = tf.cast(outputs, tf.int32)
    # Y_val_int = tf.cast(Y_val, tf.int32)
    # correct = tf.add(outputs_int, Y_val_int)
    # correct -= 1
    # correct = tf.reduce_sum(tf.abs(correct)).numpy()

    # Compare each output with its corresponding label (assuming binary classification)
    # correct = tf.reduce_sum(tf.cast(tf.equal(Y_val_int, outputs_int), tf.int32)).numpy()

    print(f"Validation Accuracy: {correct}/{total} = {correct/total:.2%}")
    
    # For demonstration, use the first 20 samples
    num_samples = min(25, len(X_val))
    labels = Y_val[:num_samples]
    outputs = outputs[:num_samples]

    # correct = tf.reduce_sum(tf.equal(labels, outputs))[0]

    dataset.plot_outputs(X_val[:num_samples], labels, outputs)
   