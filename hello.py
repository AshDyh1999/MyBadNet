import numpy as np
# print(np.float32(np.random.rand(2, 5)))
x_data = np.float32(np.random.rand(2, 5)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300
print(x_data)
print(np.dot([0.100, 0.200], x_data))
a = 8
if a == 8:
    print("8")
b = np.random.rand(5)
print(b)