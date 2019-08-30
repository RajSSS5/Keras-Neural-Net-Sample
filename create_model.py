import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import *
import tensorflow as tf

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define the model
model = Sequential()

# Add layers
# 50 -> 25 -> 10 -> 1 significantly outperforms sample!
# 6% error to .0001% error
model.add(Dense(50, input_dim=9, activation='relu', name='Layer_1'))
model.add(Dense(25, activation='relu', name='Layer_2'))
model.add(Dense(10, activation='relu', name='Layer_3'))
model.add(Dense(1, activation='linear', name='Output_Layer'))
model.compile(loss='mean_squared_error', optimizer='adam')

RUN_NAME = "CustomModel1"
# Create TensorBoard Logger
logger = keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(RUN_NAME),
    write_graph=True,
)


# Train
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    # verbose=True,
    callbacks=[logger]
)

model.save("trained_model.h5")
print("Model Saved!")

# Test/Evaluate
test_data_df = pd.read_csv("sales_data_test_scaled.csv")

X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

test_error = model.evaluate(X_test, Y_test, verbose=0 )

# print('The MSE is for the test data is {}'.format(test_error) )

# # Export for Google Cloud
# model_builder = tf.saved_model.builder.SavedModelBuilder("exported_model")
#
# inputs = {
#     'input': tf.saved_model.utils.build_tensor_info(model.input)
# }
# outputs = {
#     'earnings': tf.saved_model.utils.build_tensor_info(model.output)
# }
#
# # To run prediction function on TF
# signature_def = tf.saved_model.signature_def_utils.build_signature_def(
#     inputs=inputs,
#     outputs=outputs,
#     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
# )
#
# # Save structure/weights
# model_builder.add_meta_graph_and_variables(
#     K.get_session(),
#     tags=[tf.saved_model.tag_constants.SERVING],
#     signature_def_map={
#         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
#     }
# )
#
#
# model_builder.save()