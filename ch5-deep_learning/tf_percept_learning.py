# 텐서플로 프로그래밍 : 퍼셉트론 학습
import tensorflow as tf

# OR 데이터 구축
x=tf.constant([[0,0,0,0],[0,0,1,0],[1,0,0,0],[1,0,1,0]], dtype=tf.float32)
y=tf.constant([[-1],[1],[1],[1]], dtype=tf.float32)

# 가중치 초기화
# w 의 모양 : [2,1]
# -0.5~0.5 의 난수 생성
# bias를 크기가 1인 배열인 0으로 초기화
w=tf.Variable(tf.random.uniform([4,1],-0.5,0.5))
b=tf.Variable(tf.zeros([1]))

# 옵티마이저
opt=tf.keras.optimizers.SGD(learning_rate=0.1)

# 전방 계산
def forward():
    s=tf.add(tf.matmul(x,w),b)
    o=tf.tanh(s)
    return o

# 손실 함수 정의
def loss():
    o=forward()
    return tf.reduce_mean((y-o)**2)

# 500 세대까지 학습(100세대마다 학습 정보 출력)
for i in range(500):
    opt.minimize(loss,var_list=[w,b])
    if(i%100==0):
        print('loss at epoch',i,'=',loss().numpy())
        
# 학습된 퍼셉트론으로 OR 데이터를 예측
o=forward()
print(o)

