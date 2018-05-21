import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # DONE : Implemented function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Loading the pretrained vgg model
    tf.saved_model.loader.load(sess,vgg_tag,vgg_path);

    graph = tf.get_default_graph();

    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name);
    keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name);
    vgg_layer_3_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name);
    vgg_layer_4_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name);
    vgg_layer_7_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name);
    
    return input_tensor,keep_prob_tensor,vgg_layer3_out_tensor_name,vgg_layer4_out_tensor_name,vgg_layer7_out_tensor_name;

tests.test_load_vgg(load_vgg, tf);


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # stoping the gradient to back propogate from layer7
    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out);
    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out);
    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out);

    dec_layer_7_1x1 = tf.layers.conv2d(inputs = vgg_layer7_out, filters=num_classes, kernel_size=1,
                  strides=(1,1), padding='SAME',name='dec_layer_7_1x1',
                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),activation=tf.nn.relu);

    dec_layer7_1x1_2x = tf.layers.conv2d_transpose(dec_layer_7_1x1, filters=num_classes, kernel_size = 3,
                                                          strides=(2, 2), padding='SAME', name='dec_layer_7_1x1_2x',
                                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                          activation=tf.nn.relu);

    dec_layer4_1x1_out = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                          name="dec_layer4_1x1_out", activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    dec_layer_4_7_combined = tf.add(dec_layer7_1x1_2x, dec_layer4_1x1_out, name="dec_layer_4_7_combined");

    dec_layer4_7_upsampled = tf.layers.conv2d_transpose(dec_layer_4_7_combined, filters=num_classes, kernel_size=3,
                                                       strides=(2, 2), name="dec_layer4_7_upsampled", padding='SAME',
                                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                       activation=tf.nn.relu);

    dec_layer3_1x1_out = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                      name="dec_layer3_1x1_out", kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          activation=tf.nn.relu);

    output_layer_3_4_7_combined = tf.add(dec_layer3_1x1_out, dec_layer4_7_upsampled);

    dec_final_layer_upsampled_8x = tf.layers.conv2d_transpose(output_layer_3_4_7_combined, filters=num_classes, kernel_size=16,
                                                              strides=(8, 8), name="dec_final_layer_upsampled_8x",
                                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                              padding='SAME');

    return dec_final_layer_upsampled_8x;

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    return None, None, None
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
