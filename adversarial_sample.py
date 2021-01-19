import glob
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 20, 10

model = tf.keras.applications.MobileNetV2(include_top=True, weights="imagenet")
model.trainable = False

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0][1:]

def load_image(path):
    raw = tf.io.read_file(path)
    image = tf.image.decode_image(raw)
    image = preprocess(image)
    probs = model.predict(image)
    label_idx = np.argmax(probs)
    label = tf.one_hot(label_idx, probs.shape[-1])
    label = tf.reshape(label, (1, probs.shape[-1]))
    return image, label

loss_fn = tf.keras.losses.CategoricalCrossentropy()
def compute_perturbations(image, label, signed=False):
    with tf.GradientTape() as tape:
        tape.watch(image)
        y_pred = model(image)
        loss = loss_fn(label, y_pred)

    grads = tape.gradient(loss, image)
    if signed:
        grads = tf.sign(grads)
    else:
        grads = grads / (tf.norm(grads) + 1e-12)
    return grads[0]

def visualize(image, label):
    plt.subplot(2, 3, 1)
    perturbations = compute_perturbations(image, label, signed=True)
    plt.imshow(perturbations*0.5+0.5)  # [-1, 1] to [0,1]
    plt.title("perturbation")
    plt.axis("off")

    epsilons = [0, 0.01, 0.1, 0.15, 0.2]
    for i, eps in enumerate(epsilons, start=2):
        adv_image = image + eps * perturbations
        adv_image = tf.clip_by_value(adv_image, -1, 1)
        plt.subplot(2, 3, i)
        label, confidence = get_imagenet_label(model.predict(adv_image))
        desc = "eps = {:0.3f}".format(eps) if eps else "origin image"
        title = "{} \n {} : {:.2f}%".format(desc, label, confidence*100)
        plt.imshow(adv_image[0]*0.5+0.5)
        plt.title(title)
        plt.axis("off")

    plt.show()

if __name__ == "__main__":    
    files = glob.glob("images/**/*.jpg")
    for path in files:
        image, label = load_image(path)
        visualize(image, label)
