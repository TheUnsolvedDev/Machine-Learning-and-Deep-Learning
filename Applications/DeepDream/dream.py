import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

sky_image = tf.keras.utils.get_file("sky.jpg", "https://i.imgur.com/aGBdQyK.jpg")

layer_settings = {
    "mixed4": 1.0,
    "mixed5": 1.5,
    "mixed6": 2.0,
    "mixed7": 2.5,
}

step = 0.01 
num_octave = 3  
octave_scale = 1.4  
iterations = 20 
max_loss = 15.0

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.0
    x += 0.5
    x *= 255.0
    x = np.clip(x, 0, 255).astype("uint8")
    return x

def compute_loss(input_image):
    features = feature_extractor(input_image)
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
        loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return loss

@tf.function
def gradient_ascent_step(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    img += learning_rate * grads
    return loss, img


def gradient_ascent_loop(img, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss, img = gradient_ascent_step(img, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print("... Loss value at step %d: %.2f" % (i, loss))
    return img

if __name__ == "__main__":
    sky = cv2.imread(sky_image,1)
    show_image = cv2.cvtColor(sky,cv2.COLOR_BGR2RGB)
    plt.imshow(show_image)
    plt.show()
    
    model = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False)
    outputs_dict = dict(
        [
            (layer.name, layer.output)
            for layer in [model.get_layer(name) for name in layer_settings.keys()]
        ]
    )

    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=outputs_dict)
    original_img = preprocess_image(sky_image)
    original_shape = original_img.shape[1:3]

    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

    img = tf.identity(original_img)  # Make a copy
    for i, shape in enumerate(successive_shapes):
        print("Processing octave %d with shape %s" % (i, shape))
        img = tf.image.resize(img, shape)
        img = gradient_ascent_loop(
            img, iterations=iterations, learning_rate=step, max_loss=max_loss
        )
        upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
        same_size_original = tf.image.resize(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img

        img += lost_detail
        shrunk_original_img = tf.image.resize(original_img, shape)

    plt.imshow(img.numpy()[0])
    plt.show()