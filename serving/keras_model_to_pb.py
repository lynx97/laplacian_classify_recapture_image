import keras as K
from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

# Function to export Keras model to Protocol Buffer format
# Inputs:
#       path_to_h5: Path to Keras h5 model
#       export_path: Path to store Protocol Buffer model
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.backend.round(K.backend.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.backend.round(K.backend.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.backend.sum(y_pos * y_pred_pos)
    tn = K.backend.sum(y_neg * y_pred_neg)

    fp = K.backend.sum(y_neg * y_pred_pos)
    fn = K.backend.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.backend.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.backend.epsilon())
def export_h5_to_pb(path_to_h5, export_path):

    # Set the learning phase to Test since the model is already trained.
    K.backend.set_learning_phase(0)

    # Load the Keras model
    custom_objects={'matthews_correlation':matthews_correlation}
    keras_model = load_model(path_to_h5, custom_objects = custom_objects)

    # Build the Protocol Buffer SavedModel at 'export_path'
    builder = saved_model_builder.SavedModelBuilder(export_path)

    # Create prediction signature to be used by TensorFlow Serving Predict API
    signature = predict_signature_def(inputs={"images": keras_model.input},
                                      outputs={"scores": keras_model.output})

    with K.backend.get_session() as sess:
        # Save the meta graph and the variables
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                         signature_def_map={"predict": signature})

    builder.save()
keras_model = "/home/liem/hai/recapture_classification/trained_model/v3_train_id/checkpoint/weights-improvement-141-0.95.hdf5"
export_model = "/home/liem/hai/recapture_classification/serving/export"
export_h5_to_pb(keras_model, export_model)