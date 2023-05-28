#https://towardsdatascience.com/understand-and-implement-vision-transformer-with-tensorflow-2-0-f5435769093

import tensorflow as tf
import numpy as np
#from tensorflow.keras.datasets import cifar10
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
import learn
x_train, y_train = learn.svg_to_tf()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20,
                                                    stratify=y_train, random_state=int(np.random.random()*(1<<32-1)), shuffle = True)

#print(y_train)
train_im, valid_im, train_lab, valid_lab = train_test_split(x_train, y_train, test_size=0.20,
                                                            stratify=y_train, random_state=int(np.random.random()*(1<<32-1)), shuffle = True)
train_lab = tf.keras.utils.to_categorical(train_lab, num_classes=5, dtype='uint8')
valid_lab = tf.keras.utils.to_categorical(valid_lab, num_classes=5, dtype='uint8')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5, dtype='uint8')
training_data = tf.data.Dataset.from_tensor_slices((train_im, train_lab))
validation_data = tf.data.Dataset.from_tensor_slices((valid_im, valid_lab))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
autotune = tf.data.AUTOTUNE
#train_data_batches = training_data.shuffle(buffer_size=40000).batch(128).prefetch(buffer_size=autotune)
#valid_data_batches = validation_data.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=autotune)
#test_data_batches = test_data.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=autotune)

from tensorflow.keras import layers
#### load data and process
autotune = tf.data.AUTOTUNE 

##### generate patches 
class generate_patch(layers.Layer):
  def __init__(self, patch_size):
    super(generate_patch, self).__init__()
    self.patch_size = patch_size
    
  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(images=images, 
                                       sizes=[1, self.patch_size, self.patch_size, 1], 
                                       strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims]) #here shape is (batch_size, num_patches, patch_h*patch_w*c) 
    return patches

#############
# visualize
#############
from itertools import islice, count

train_iter_7im, train_iter_7label = next(islice(training_data, 7, None)) # access the 7th element from the iterator

patch_size = 4

train_iter_7im = tf.expand_dims(train_iter_7im, 0)
train_iter_7label = train_iter_7label.numpy() 

generate_patch_layer = generate_patch(patch_size=patch_size)
patches = generate_patch_layer(train_iter_7im)

print ('patch per image and patches shape: ', patches.shape[1], '\n', patches.shape)

#class_types = {0: "airplane", 1: "automobile", 2: "bird",
#    3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
#    8: "ship", 9: "truck"}
class_types = {1: 'Unreadable', 2: 'Low Readability', 3: 'Average Readability', 4: 'Good Readability', 5: 'Excellent Readability'}

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
    ax[1].set_xlabel(class_types[np.argmax(train_iter_7label)+1], fontsize=13)
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

render_image_and_patches(train_iter_7im, patches)

'''
This part takes images as inputs,
Conv layer filter matches query dim of multi-head attention layer 
Add embeddings by randomly initializing the weights
'''

def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
  patches = layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
  row_axis, col_axis = (1, 2) # channels last images
  seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
  x = tf.reshape(patches, [-1, seq_len, hidden_size])
  return x


### Positonal Encoding Layer
class PatchEncode_Embed(layers.Layer):
  #‘’’
  #2 steps happen here
  #1. flatten the patches
  #2. Map to dim D; patch embeddings
  #‘’’
  def __init__(self, num_patches, projection_dim):
    super(PatchEncode_Embed, self).__init__()
    self.num_patches = num_patches
    self.projection = layers.Dense(units=projection_dim)
    self.position_embedding = layers.Embedding(
    input_dim=num_patches, output_dim=projection_dim)
  def call(self, patch):
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    encoded = self.projection(patch) +               self.position_embedding(positions)
    return encoded
  
hyperparameters = {"stddev": 0.02, "transformer_layers": 6}

### Positonal Encoding Layer

class AddPositionEmbs(layers.Layer):
  """inputs are image patches 
  Custom layer to add positional embeddings to the inputs."""

  def __init__(self, posemb_init=None, **kwargs):
    super().__init__(**kwargs)
    self.posemb_init = posemb_init
    #posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

    return inputs + pos_embedding

pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=hyperparameters["stddev"])) 

'''
part of ViT Implementation
this block implements the Transformer Encoder Block
Contains 3 parts--
1. LayerNorm 2. Multi-Layer Perceptron 3. Multi-Head Attention
For repeating the Transformer Encoder Block we use Encoder_f function. 
'''

def mlp_block_f(mlp_dim, inputs):
  x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
  x = layers.Dropout(rate=0.1)(x) # dropout rate is from original paper,
  x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x) # check GELU paper
  x = layers.Dropout(rate=0.1)(x)
  return x

def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
  x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
  x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x) 
  # self attention multi-head, dropout_rate is from original implementation
  x = layers.Add()([x, inputs]) # 1st residual part 
  
  y = layers.LayerNormalization(dtype=x.dtype)(x)
  y = mlp_block_f(mlp_dim, y)
  y_1 = layers.Add()([y, x]) #2nd residual part 
  return y_1

def Encoder_f(num_layers, mlp_dim, num_heads, inputs):
  x = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input')(inputs)
  x = layers.Dropout(rate=0.2)(x)
  for _ in range(num_layers):
    x = Encoder1Dblock_f(num_heads, mlp_dim, x)

  encoded = layers.LayerNormalization(name='encoder_norm')(x)
  return encoded

### augment train but not test 

rescale_layer = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255)])

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2), 
  layers.experimental.preprocessing.RandomZoom(height_factor=(0.2, 0.3), width_factor=(0.2, 0.3)),
  layers.experimental.preprocessing.RandomTranslation(0.3, 0.3, fill_mode='reflect', interpolation='bilinear',)
])


train_ds = (training_data.shuffle(40000).batch(128).map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=autotune).prefetch(autotune))
valid_ds = validation_data.shuffle(10000).batch(32).prefetch(autotune)

'''
Building blocks of ViT
Check other gists or the complete notebook
[]
Patches (generate_patch_conv_orgPaper_f) + embeddings (within Encoder_f)
Transformer Encoder Block (Encoder_f)
Final Classification 
'''

######################################
# hyperparameter section 
###################################### 
transformer_layers = 6
patch_size = 4
hidden_size = 64
num_heads = 4
mlp_dim = 128 #multilayer perceptron dimension

######################################

rescale_layer = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255)])


def build_ViT():
  inputs = layers.Input(shape=train_im.shape[1:])
  # rescaling (normalizing pixel val between 0 and 1)
  rescale = rescale_layer(inputs)
  # generate patches with conv layer
  patches = generate_patch_conv_orgPaper_f(patch_size, hidden_size, rescale)

  ######################################
  # ready for the transformer blocks
  ######################################
  encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, patches)  

  #####################################
  #  final part (mlp to classification)
  #####################################
  #encoder_out_rank = int(tf.experimental.numpy.ndim(encoder_out))
  im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)
  # similar to the GAP, this is from original Google GitHub

  logits = layers.Dense(units=len(class_types), name='head', kernel_initializer=tf.keras.initializers.zeros)(im_representation)
  # !!! important !!! activation is linear 
  # class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer',
  #                 'dog', 'frog', 'horse', 'ship', 'truck'] # from cifar-10 website

  final_model = tf.keras.Model(inputs = inputs, outputs = logits)
  return final_model



ViT_model = build_ViT()
ViT_model.summary()

tf.keras.utils.plot_model(ViT_model, rankdir='TB')

### model 

#ViT_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3), 
#                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
#                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5 acc')]) 


ViT_model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3), 
                   loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1), 
                   metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5 acc'), 
                            tf.keras.metrics.Precision(name='pre'), 
                            tf.keras.metrics.Recall(name='rec')])


#tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")],) 
# from logits = True, because Dense layer has linear activation


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=5, min_lr=1e-5, verbose=1)

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=10, verbose=0, 
                                      mode="auto", baseline=None,restore_best_weights=False)


ViT_Train = ViT_model.fit(train_ds, 
                        epochs = 100, 
                        validation_data=valid_ds, callbacks=[reduce_lr])


### Plot train and validation curves
loss = ViT_Train.history['loss']
v_loss = ViT_Train.history['val_loss']

acc = ViT_Train.history['accuracy'] 
v_acc = ViT_Train.history['val_accuracy']

top5_acc = ViT_Train.history['top5 acc']
val_top5_acc = ViT_Train.history['val_top5 acc']
epochs = range(len(loss))

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.yscale('log')
plt.plot(epochs, loss, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Loss')
plt.plot(epochs, v_loss, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Loss')
# plt.ylim(0.3, 100)
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(1, 3, 2)
plt.plot(epochs, acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Acc')
plt.plot(epochs, v_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Acc') 
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(1, 3, 3)
plt.plot(epochs, top5_acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Top 5 Acc')
plt.plot(epochs, val_top5_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Top5 Acc') 
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Top5 Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('train_acc.png', dpi=250)
plt.show()
     

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def conf_matrix(predictions):
    import numpy as np
    ''' Plots conf. matrix and classification report '''
    cm=confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    print("Classification Report:\n")
    cr=classification_report(np.argmax(y_test, axis=1),
                                np.argmax(predictions, axis=1), 
                                target_names=[class_types[i] for i in sorted(class_types)])
    print(cr)
    plt.figure(figsize=(12,12))
    sns_hmp = sns.heatmap(cm, annot=True, xticklabels = [class_types[i] for i in sorted(class_types)], 
                yticklabels = [class_types[i] for i in class_types], fmt="d")
    fig = sns_hmp.get_figure()
    fig.savefig('heatmap.png', dpi=250)

pred_class_resnet50 = ViT_model.predict(x_test)
print(pred_class_resnet50)
conf_matrix(pred_class_resnet50)