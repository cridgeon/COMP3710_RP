from keras import callbacks
import matplotlib.pyplot as plt
import json
import tensorflow as tf

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
    model.load_weights(save_path)

def train(model, train_data, test_data):
    X_train, Y_train, P_train, N_train = train_data
    X_test, Y_test, P_test, N_test = test_data

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

    plt.figure("Triplet loss vx  Epoch")
    plt.plot(history.history['triplet_loss'], label='triplet loss')
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.yscale('log')
    min = tf.reduce_min(history.history['triplet_loss'])
    max = tf.reduce_max(history.history['triplet_loss'])
    plt.ylim([min, max])
    plt.legend(loc='lower right')
    plt.show()
