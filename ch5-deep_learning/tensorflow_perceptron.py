import tensorflow as tf

# OR 데이터 구축
x = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=tf.float32)  # 입력 데이터
y = tf.constant([[-1], [1], [1], [1]], dtype=tf.float32)  # 정답 라벨

w = tf.Variable([[1.0], [1.0]], dtype=tf.float32)  # 4개의 특징에 대해 1개의 가중치
b = tf.Variable(-0.5, dtype=tf.float32)  # 바이어스

# 행렬 곱셈 및 연산 수행
s = tf.add(tf.matmul(x, w), b)  # x와 w의 행렬 곱셈 후 바이어스 더함
o = tf.sign(s)  # 활성화 함수 적용 (sign)

print(o)
