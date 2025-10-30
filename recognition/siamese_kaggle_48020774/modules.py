from keras import layers, models, losses, optimizers, ops, metrics, Model, applications
import tensorflow as tf
import dataset

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
        # Use euclidean distance instead of squared distance for better stability
        ap_distance = tf.norm(anchor - positive, axis=1, keepdims=True)
        an_distance = tf.norm(anchor - negative, axis=1, keepdims=True)

        return (ap_distance, an_distance)
    
    def compute_output_shape(self, input_shape):
        # input_shape is a list/tuple of 4 shapes: [anchor, label, positive, negative]
        return (None,), (None,)
    
class TripletLoss(layers.Layer):
    """
    This layer calculates the triplet loss so that it can be tracked
    by the optimiser
    """

    def __init__(self, alpha=1.0, name="triplet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.triplet_loss_tracker = metrics.Mean(name="triplet_loss")
        self.alpha = alpha

    def call(self, distances, training=None):
        ap_distance, an_distance = distances
        # compute triplet loss

        loss = ops.mean(ops.maximum(ap_distance - an_distance + self.alpha, 0.0))

        # update trackers (means over batch)
        self.triplet_loss_tracker.update_state(loss)
        self.add_loss(loss)
        # self.add_metric(self.triplet_loss_tracker.result(), name="triplet_loss", aggregation="mean")

        # pass distances through unchanged
        return distances

    def reset_states(self):
        self.triplet_loss_tracker.reset_states()

    # expose trackers so Model logs them automatically
    @property
    def metrics(self):
        return [self.triplet_loss_tracker]
    
    def compute_output_shape(self, input_shape):
        return input_shape

class DisplayLayer(layers.Layer):
    """
    This layer converts the distance ouput into a 0/1 output
    so that we can easily see if the model thinks the anchor
    is closer to the positive (malignant) or negative (benign) example.
    """

    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.accuracy_tracker = metrics.BinaryAccuracy(name="accuracy")
        self.AUCROC_tracker = metrics.AUC(name="AUCROC")
        self.cross_entropy = losses.BinaryCrossentropy()
        self.cross_entropy_tracker = metrics.Mean(name="cross_entropy")
        self.threshold = threshold

    def call(self, distances, labels, training=None):
        ap_distance, an_distance = distances
        total_diff = tf.add(ap_distance, an_distance)
        positive_probability = tf.divide(an_distance, total_diff)
        classifications = tf.greater(positive_probability, tf.ones_like(positive_probability, dtype=tf.float32) * self.threshold)
        
        classifications = tf.logical_not(classifications)
        labels_bool = tf.cast(labels, tf.bool)
        
        # logical XOR
        classifications = tf.logical_and(tf.logical_or(classifications, labels_bool), tf.logical_not(tf.logical_and(classifications, labels_bool)))
        
        self.accuracy_tracker.update_state(labels_bool, classifications)
        self.AUCROC_tracker.update_state(labels_bool, classifications)
        
        ce = self.cross_entropy(labels, positive_probability)
        self.add_loss(ce)
        self.cross_entropy_tracker.update_state(ce)

        
        return classifications
    
    def reset_states(self):
        """Reset the metrics state between training and validation phases"""
        self.accuracy_tracker.reset_states()
        self.AUCROC_tracker.reset_states()
    
    # expose trackers so Model logs them automatically
    @property
    def metrics(self):
        return [self.accuracy_tracker, self.AUCROC_tracker]
    
    def compute_output_shape(self, *args, **kwargs):
        return (None,)

class EncoderHead(layers.Layer):
    """
    This layer is a densenet classifier head
    that produces the final embeddings for the
    triplet network.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation='relu')
        self.dropout1 = layers.Dropout(0.3)  # Increased from 0.5
        self.dense2 = layers.Dense(512, activation='relu')  # Reduced from 512
        self.dropout2 = layers.Dropout(0.3)  # Increased from 0.5
        self.dense3 = layers.Dense(256, activation='relu')  # Reduced from 256
        self.dropout3 = layers.Dropout(0.3)  # Increased from 0.5

    def call(self, resnet_output):
        # haha python is stupid
        return self.dropout3(
            self.dense3(
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
        )


def construct_classifier():
    base_cnn = applications.ResNet50(
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
    input_label = layers.Input(name='input_label', shape=(1,))
    input_positive = layers.Input(name='input_positive', shape=INPUT_SHAPE)
    input_negative = layers.Input(name='input_negative', shape=INPUT_SHAPE)

    output_distances = DistanceLayer()(
        classifier(dataset.TrainTestPreprocessor()(input_anchor)),
        classifier(dataset.TrainTestPreprocessor()(input_positive)),
        classifier(dataset.TrainTestPreprocessor()(input_negative))
    )
    loss_layer = TripletLoss(alpha=0.2, name='triplet_loss')(output_distances)
    output_display = DisplayLayer(name='output_display')(loss_layer, input_label)
    model = models.Model(
        inputs=[input_anchor, input_label, input_positive, input_negative],
        outputs=output_display
    )
    
    # classifier.summary()
    model.summary()
    return model

def compile_model(model : Model, learning_rate):
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=1e-4  # Add L2 regularization
        ),
    )

if __name__ == "__main__":
    model = construct_classifier()
    compile_model(model, 0.01)