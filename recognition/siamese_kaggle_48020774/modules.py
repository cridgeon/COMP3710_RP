from keras import layers, models, losses, optimizers, ops, metrics
import tensorflow as tf

#adapted from 
#REF: https://keras.io/examples/vision/siamese_network/

INPUT_SHAPE = (256, 256, 3,)

# I understand that this is a copied block of code
# and realised that the task sheet advises against this.
# there are only so many ways to write this and
# after seeing it, cannot think of any other reasonable way.
# I understand what the code does, and that is the purpose of
# undergoing a degree. The world runs on copied code. sue me
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = ops.sum(tf.square(anchor - positive), -1)
        an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)
    
class TripletLoss(layers.Layer):
    """
    This layer calculates the triplet loss so that it can be tracked
    by the optimiser
    """

    def __init__(self, alpha=0.5, name="vae_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss = metrics.Mean(name="triplet_loss")
        self.alpha = alpha

    def call(self, inputs):
        # inputs: [y_true, y_pred, mu, logvar]
        ap_distance, an_distance = inputs

        # compute triplet loss

        loss = ops.maximum(ap_distance - an_distance + self.alpha, 0.0)

        # update trackers (means over batch)
        self.loss.update_state(loss)

        # pass distances through unchanged
        return inputs

    # expose trackers so Model logs them automatically
    @property
    def metrics(self):
        return [self.loss]
    
    def compute_output_shape(self, input_shape):
        # input_shape is a list/tuple of 4 shapes: [y_true, y_pred, mu, logvar]
        return input_shape

class DisplayLayer(layers.Layer):
    """
    This layer converts the distance ouput into a 0/1 output
    so that we can easily see if the model thinks the anchor
    is closer to the positive (malignant) or negative (benign) example.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, distances):
        ap_distance, an_distance = distances
        diff = ap_distance - an_distance
        ret = tf.convert_to_tensor(1 if diff >= 0 else 0, tf.int8)
        return ret
    
    def compute_output_shape(self, *args, **kwargs):
        return (1,)

class EncoderHead(layers.Layer):
    """
    This layer is a densenet classifier head
    that produces the final embeddings for the
    triplet network.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(512, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.dense3 = layers.Dense(256, activation='relu')

    def call(self, resnet_output):
        # haha python is stupid
        return self.dense3(
            self.dropout2(
                self.dense2(
                    self.dropout1(
                        self.dense1(
                            self.flat(
                                resnet_output
                            )
                        )
                    )
                )
            )
        )

def construct_classifier():
    base_cnn = tf.keras.applications.ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=INPUT_SHAPE
    )
    
    encoding = EncoderHead()(base_cnn.output)

    classifier = models.Model(base_cnn.input, encoding, name='classifier')

    # construct the twin (or in this case triplet) network 
    # by passing in the image to be classified, and a
    # positive and negative image

    input_anchor = layers.Input(name='input_anchor', shape=INPUT_SHAPE)
    input_positive = layers.Input(name='input_positive', shape=INPUT_SHAPE)
    input_negative = layers.Input(name='input_negative', shape=INPUT_SHAPE)

    output_distances = DistanceLayer()(
        classifier(input_anchor),
        classifier(input_positive),
        classifier(input_negative)
    )
    loss_layer = TripletLoss(alpha=0.5, name='triplet_loss')(output_distances)
    output_display = DisplayLayer(name='output_display')(loss_layer)

    model = models.Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=output_display
    )

    classifier.summary()
    model.summary()
    return model

def compile_model(model, learning_rate):
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  metrics=['triplet_loss'])

if __name__ == "__main__":
    model = construct_classifier()
    compile_model(model, 0.01)