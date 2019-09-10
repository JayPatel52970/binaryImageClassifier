import cv2
import tensorflow as tf

def prepare(filePath):
    IMG_SIZE=100
    img_array=cv2.imread(filePath)
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)

model = tf.kears.models.load_model("JayCNN.model")

prediction = model.predict([prepare("test4.jpg")])
if (("0" in str(prediction))==True):
    print("Image Contains Gold")
else:
    print("Image Contains Silver")