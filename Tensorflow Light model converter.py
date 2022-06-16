from keras.models import load_model
import tensorflow as tf
model = load_model('./Models/model.h5', compile=False)
export_path='./Models/pb'
model.save(export_path, save_format="tf")
converter = tf.lite.TFLiteConverter.from_saved_model(export_path)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('./Models/tflite/model.tflite','wb').write(tflite_model)