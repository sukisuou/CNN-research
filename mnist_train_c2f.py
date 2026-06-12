# ================================== Experimental Log ==================================
# 1.0 : Basic CNN + ANN                             - 0.9877, 0.0390     --> [acc, loss]
# 1.1 : added bottlenecking                         - 0.9890, 0.0384
# 1.2 : C2f style with bottleneck                   - 0.9883, 0.0339

# 2.0 : full CNN model                              - 0.9838, 0.0522
# 2.1 : half channels                               - 0.9803, 0.0647
# 2.2 : learnt CNN classifier head with conv layer
# 2.3 : introduces strides                          - 0.9737, 0.0878, no early_stopping
# 2.4 : too aggressive, use maxpool (ref #2.0)      - 0.9885, 0.0351

# 3.0 : taking inspiration from YOLO's C2f block
# 3.1 : introduces splitting (32, 32) -> (32, 16)   - 0.9842, 0.0495, no early_stopping
# 3.2 : no dimension reduction (1x1 projection)     - 0.9861, 0.0454, no early_stopping
# deduction : optimization diff increases as model becomes more sophisticated

# 4.0 : increases epoch (50 -> 100) for no-early-stop cases
# 4.1 : 1x1 project rerun                           - 0.9876, 0.0418 [67]
# ======================================================================================

# import libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Rescaling, Concatenate, Dense, Dropout, LeakyReLU, Conv2D, MaxPooling2D, GlobalAveragePooling2D
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# import dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# add a channel dimension for CNN (60000, 28, 28, 1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# all 10 unique labels
label_classes = np.unique(y_train)

# print function
def print_image(image, label):
    print(f'Example number {label}')
    for layer in image:
        for pixel in layer:
            if pixel[0] > 255/2.0:
                print('#', end = ' ')
            else:
                print('.', end = ' ')
        print()
    print()

# get example images
examples = {}
for image, label in zip(x_train, y_train):
    if label not in examples:
        examples[label] = image
    if len(examples) == 10:
        break

for label, image in sorted(examples.items()):
    print_image(image, label)

# modelling (using functional API)
def create_model():
    inputs = Input(shape = x_train.shape[1:])
    x = Rescaling(1/255.0)(inputs)

    # first conv layer
    x = Conv2D(32, (3, 3), name = 'first_layer')(x)
    x = LeakyReLU(alpha = 0.01)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # second conv layer
    x = Conv2D(64, (3, 3), name = 'second_layer')(x)
    x = LeakyReLU(alpha = 0.01, name = 'og_leaky')(x)

    # split branch (32 - 32) using wrapper
    branch_keep, branch_bottleneck = tf.keras.layers.Lambda(
        lambda t: tf.split(t, num_or_size_splits = 2, axis = -1),
        name = 'split_branch'
    )(x)

    # bottleneck block - keep 32, only transform, not reduce dim
    branch_bottleneck = Conv2D(32, (1, 1), name = 'bottleneck')(branch_bottleneck)
    branch_bottleneck = LeakyReLU(alpha = 0.01, name = 'bn_tensor')(branch_bottleneck)

    # merge paths
    x = Concatenate()([branch_keep, branch_bottleneck])
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # classifier head
    x = Conv2D(len(label_classes), (1, 1), name = 'class_head')(x)
    x = GlobalAveragePooling2D()(x)

    # output layer
    outputs = tf.keras.layers.Activation('softmax')(x)
    
    # build the model from linked input and output nodes
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(
        optimizer = 'Adam', 
        loss = 'sparse_categorical_crossentropy', 
        metrics = ['accuracy']
    )

    return model

# early stopped to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_accuracy',
    mode = 'max',
    patience = 10,
    restore_best_weights = True
)

# create model
model = create_model()
model.summary()

# train model 
print('Training model...')
history = model.fit(
    x_train, y_train,
    epochs = 100,
    batch_size = 32,
    verbose = True,
    validation_data = (x_test, y_test),
    callbacks = [early_stopping]
)
print('Training done!\n')

# validate with predictions
y_prob = model.predict(x_test, verbose = 0)
y_pred = np.argmax(y_prob, axis = 1)
y_true = y_test

# evaluate model's metrics
history_dict = list(history.history.items())
with open('training_log_c2f.txt', 'w') as file:     # save in a file
    file.write(f"Stopped at epoch: {len(history.history['loss'])}\n")
    file.write('\n--- Training Scores ---\n')
    for metrics_name, score in history_dict:
        if not metrics_name.startswith('val_'):
            file.write(f'{metrics_name}: {score[-1]:.4f}\n')
    file.write('\n--- Validation Scores ---\n')
    for metrics_name, score in history_dict:
        if metrics_name.startswith('val_'):
            file.write(f'{metrics_name}: {score[-1]:.4f}\n')

    # classification report
    file.write('\n--- Classification Report ---\n')
    file.write(classification_report(
        y_true, y_pred,
        target_names = [str(x) for x in label_classes],
        zero_division = 0
    ))

    # confusion matrix
    file.write('\n--- Confusion Matrix ---\n')
    cm = confusion_matrix(y_true, y_pred)
    file.write(str(cm))
with open('training_log_c2f.txt', 'r') as file:     # print to terminal
    print(file.read())

# save model
model.export('mnist_c2f_model')