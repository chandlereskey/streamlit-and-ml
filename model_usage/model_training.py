import numpy as np
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input


# Loading data from the 'mnist_compressed.npz' file
# This dataset comes from https://www.kaggle.com/datasets/martininf1n1ty/mnist100
data = np.load("./model_usage/mnist_compressed.npz")

print(data)

# Reading variables containing the data
X_test, y_test, X_train, y_train = data['test_images'], data['test_labels'], data['train_images'], data['train_labels']
print(type(X_train), type(X_train[0]))
def convert_to_black_and_white_and_normalize(np_array):
    max_val = np_array.max()
    threshold = max_val // 2
    # convert to only black and white
    b_w = np.where(np_array > threshold, 255, 0)

    # convert all 255 -> 1
    normalized = np.where(b_w == 255, 1, 0)
    # print(normalized.max())
    return normalized

# convert np arrays to black and white with thresholding
X_train_updated = []
for number_representation in X_train:
    X_train_updated.append(convert_to_black_and_white_and_normalize(number_representation))

X_test_updated = []
for number_representation in X_test:
    X_test_updated.append(convert_to_black_and_white_and_normalize(number_representation))

X_train_updated = np.array(X_train_updated)
X_test_updated = np.array(X_test_updated)
X_train_updated = X_train_updated.reshape(len(X_train_updated), 28, 56, 1)
X_test_updated = X_test_updated.reshape(len(X_test_updated), 28, 56, 1)

print(X_train_updated.shape)
print(X_train_updated[0].max())

# cv2.imwrite('sample2.png', np.where(X_train[0] == 0, 0, 255))
# Model Creation
inputs = Input((28, 56, 1))
#convolutional layers
conv_1 = Conv2D(32, kernel_size=(3,3), activation='relu')(inputs)
maxp_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3,3), activation='relu')(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3,3), activation='relu')(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2,2))(conv_3)

flatten = Flatten()(maxp_3)

# fully connected layers
dense = Dense(128, activation='relu')(flatten)

dropout = Dropout(0.3)(dense)

output = Dense(100, activation='softmax', name='number')(dropout)

model = Model(inputs=[inputs], outputs=output)

model.compile(loss=['SparseCategoricalCrossentropy'], optimizer='adam', metrics=['accuracy'])

history = model.fit(x=X_train_updated, y=[y_train], batch_size=32, epochs=10, validation_split=0.2)

# save the model
model.save('./model_usage/number_predictor.keras')