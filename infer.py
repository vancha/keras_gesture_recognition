import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
image_size = (180, 180)

img_loc = sys.argv[1] #"verify/verify_peace/out10.jpg"
img = keras.preprocessing.image.load_img(
    img_loc, target_size=image_size
)
model = load_model('my_model.h5')

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(
    f"{img_loc} is %.2f percent horn and %.2f percent peace."
    % (100 * (1 - score), 100 * score)
)

if __name__ == "main":
  print("test");
