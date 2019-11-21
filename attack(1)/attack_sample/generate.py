from fgsmOnNet import *


def generate(images, shape):
    if np.max(images) > 1:
        images = images/255
    generate_images = make_fgsm(sess, sess_model, images, eps=0.02, epochs=12)
    return generate_images
