import numpy as np

from neural.utils import mse, mse_prime, sigmoid, sigmoid_prime, tanh, tanh_prime
from neural.activation_layer import ActivationLayer
from neural.fc_layer import FCLayer
from neural.network import Network

#
# Datas
#
xt = [0, 0, 0, 1, 1, 0, 1, 1]
yt = [0, 1, 1, 0]

x_train = np.array(xt).reshape(4, 1, 2)
y_train = np.array(yt).reshape(4, 1, 1)

net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))
net.use(mse, mse_prime)

net.fit(x_train, y_train, epochs = 1000, learning_rate = 0.1)
out = net.predict(x_train)
print(out)





