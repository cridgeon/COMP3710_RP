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

def train(model, dataset, rounds = 10):

    histories = []

    for i in range(rounds):
        print(f"--- Training round {i+1}/{rounds} ---")
        print("Generating training data...")
        X_train, Y_train, P_train, N_train = dataset.GenerateTrainSet(4000)
        print("Generating testing data...")
        X_test, Y_test, P_test, N_test = dataset.GenerateTestSet(500)
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
        if i == 0:
            continue
        print("Recording history...")
        histories.append(history)

    total_history_dict = {}
    for hist in histories:
        for key in hist.history.keys():
            total_history_dict[key] = total_history_dict.get(key, []) + hist.history[key]

    plt.figure("Triplet loss vs Epoch")
    plt.plot(total_history_dict['triplet_loss'], label='training loss')
    plt.plot(total_history_dict['val_triplet_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.yscale('log')
    min = np.min([tf.reduce_min(total_history_dict['triplet_loss']), tf.reduce_min(total_history_dict['val_triplet_loss'])])
    max = np.max([tf.reduce_max(total_history_dict['triplet_loss']), tf.reduce_max(total_history_dict['val_triplet_loss'])])
    plt.ylim([min, max])
    plt.legend(loc='lower right')
    plt.show()

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
        if outputs[i]:
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
   