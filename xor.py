# xor.py
import random

# seed is set for Python 3.6.5 (for other versions other seed might be required)
random.seed(33)

w11 = random.uniform(-0.2, 0.2)
w12 = random.uniform(-0.2, 0.2)
w21 = random.uniform(-0.2, 0.2)
w22 = random.uniform(-0.2, 0.2)
b1 = random.uniform(-0.2, 0.2)
b2 = random.uniform(-0.2, 0.2)
u = random.uniform(-0.2, 0.2)
v = random.uniform(-0.2, 0.2)
b3 = random.uniform(-0.2, 0.2)

trainingData = []


def generateTrainingData():
    for i in range(10000):
        x1 = random.randint(0, 1)
        x2 = random.randint(0, 1)
        trainingData.append([x1, x2])


def activation(x):
    if x < 0:
        return 0.1 * x
    return x


def deriv_activation(x):
    if x < 0:
        return 0.1
    return 1


def truth(x1, x2):
    if x1 == x2:
        return 0
    return 1


def train():
    global w11, w12, w21, w22, b1, b2, u, v, b3
    EPOCH = 10
    n = 0.1

    for i in range(EPOCH):
        for j in range(len(trainingData)):
            x1 = trainingData[j][0]
            x2 = trainingData[j][1]

            s1 = b1 + w11 * x1 + w21 * x2
            s2 = b2 + w12 * x1 + w22 * x2

            y1 = activation(s1)
            y2 = activation(s2)

            s3 = b3 + u * y1 + v * y2
            y = activation(s3)

            t = truth(x1, x2)
            db3 = (y - t) * deriv_activation(s3)
            du = (y - t) * deriv_activation(s3) * y1
            dv = (y - t) * deriv_activation(s3) * y2
            db1 = (y - t) * deriv_activation(s3) * u * deriv_activation(s1)
            dw11 = (y - t) * deriv_activation(s3) * u * deriv_activation(s1) * x1
            dw21 = (y - t) * deriv_activation(s3) * u * deriv_activation(s1) * x2
            db2 = (y - t) * deriv_activation(s3) * v * deriv_activation(s2)
            dw12 = (y - t) * deriv_activation(s3) * v * deriv_activation(s2) * x1
            dw22 = (y - t) * deriv_activation(s3) * v * deriv_activation(s2) * x2

            b3_new = b3 - n * db3
            u_new = u - n * du
            v_new = v - n * dv
            b1_new = b1 - n * db1
            b2_new = b2 - n * db2
            w11_new = w11 - n * dw11
            w21_new = w21 - n * dw21
            w12_new = w12 - n * dw12
            w22_new = w22 - n * dw22

            b3 = b3_new
            u = u_new
            v = v_new
            b1 = b1_new
            b2 = b2_new
            w11 = w11_new
            w12 = w12_new
            w21 = w21_new
            w22 = w22_new


def compute(x1, x2):
    s1 = b1 + w11 * x1 + w21 * x2
    s2 = b2 + w12 * x1 + w22 * x2
    y1 = activation(s1)
    y2 = activation(s2)
    s3 = b3 + u * y1 + v * y2
    y = activation(s3)
    y = 0 if y < 0.5 else 1
    return y


def validate():
    print("compute(0,0) = 0 ? " + str(compute(0, 0)))
    print("compute(0,1) = 1 ? " + str(compute(0, 1)))
    print("compute(1,0) = 1 ? " + str(compute(1, 0)))
    print("compute(1,1) = 0 ? " + str(compute(1, 1)))


generateTrainingData()
train()
validate()
