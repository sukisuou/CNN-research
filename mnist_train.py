# basic modelling
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Rescaling
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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

# modelling
def create_model():
    model = tf.keras.models.Sequential([
        Rescaling(1/255.0, input_shape = x_train.shape[1:]),

        # first windowed layer
        Conv2D(32, (3, 3), name = 'first_layer'),
        LeakyReLU(alpha = 0.01),
        MaxPooling2D((2, 2)),
        Dropout(0.1),

        # second windowed layer
        Conv2D(64, (3, 3), name = 'second_layer'),
        LeakyReLU(alpha = 0.01),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # average pooling instead of flatten
        GlobalAveragePooling2D(),

        # dense layer for feature learning
        Dense(256, name = 'dense_layer'),
        LeakyReLU(alpha = 0.01),
        Dropout(0.4),

        # output layer
        Dense(len(label_classes), activation = 'softmax', name = 'output_layer')
    ])

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
    patience = 7,
    restore_best_weights = True
)

# create model
model = create_model()
model.summary()

# train model 
print('Training model...')
history = model.fit(
    x_train, y_train,
    epochs = 50,
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
with open('training_log.txt', 'w') as file:     # save in a file
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
with open('training_log.txt', 'r') as file:     # print to terminal
    print(file.read())

# save model
model.export('mnist_model')