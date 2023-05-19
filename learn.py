
def svg_to_tf():
    import tensorflow as tf
    import numpy as np
    train_ds = tf.keras.utils.image_dataset_from_directory("app/pngArchives", class_names=[str(x) for x in range(1, 5+1)], color_mode="rgb",
                                                                     image_size=(32, 32), interpolation='gaussian')
    train_ds = train_ds.map(lambda x, y: (tf.cast(tf.math.round(x), tf.uint8), tf.cast(y, tf.uint8))) #tf.transpose(x, (0, 3, 1, 2))
    #return train_ds, valid_ds
    train_ds = train_ds.unbatch()
    return (tf.squeeze(tf.convert_to_tensor(np.asarray(list(train_ds.map(lambda x, y: x))))).numpy(),
             tf.squeeze(tf.convert_to_tensor(np.asarray(list(train_ds.map(lambda x, y: y))))).numpy())

"""
def render_image_and_patches(image, patches):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.3), dpi=100.0)
    fig.suptitle("Image broken into patches (‘tokens’), using the code block above.")
    from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
    #fig.imshow(tf.cast(image[0], tf.uint8))
    imagebox = OffsetImage(tf.cast(image[0], tf.uint8), zoom = 8.0)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    renderer = canvas.get_renderer()
    axsize = ax[0].get_tightbbox(renderer=renderer)
    ab = AnnotationBbox(imagebox, (0.529, 0.53), frameon = False)
    ax[0].add_artist(ab)
    ax[0].set_xticks([i/6 for i in range(7)], [str(x) for x in range(0, 35, 5)])
    ax[0].set_yticks([i/6 for i in range(7)], [str(x) for x in range(0, 35, 5)])
    ax[1].set_xlabel(class_types[np.argmax(train_iter_7label)], fontsize=13)
    #n = int(np.sqrt(patches.shape[1]))
    #plt.figure(figsize=(6, 6))
    ax[1].set_title("Image Patches", size=13)
    for i, patch in enumerate(patches[0]):
        #ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        imagebox = OffsetImage(patch_img.numpy().astype("uint8"), zoom = 6.0)
        ab = AnnotationBbox(imagebox, (i % 8 / 8, 0.05 + (7 - i // 8) / 8), frameon = False)
        ax[1].add_artist(ab)
        #ax.imshow(patch_img.numpy().astype("uint8"))
        ax[1].axis('off')
    fig.savefig("test.svg", format="svg")
    fig.savefig("test.png", format="png")
"""