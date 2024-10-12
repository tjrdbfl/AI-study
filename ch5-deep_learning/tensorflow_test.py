# 텐서플로 버전과 동작 학인
import tensorflow as tf

print(tf.__version__)
print("Keras version:", tf.keras.__version__)
a=tf.random.uniform([2,3],0.1)
print(a)
print(type(a))
