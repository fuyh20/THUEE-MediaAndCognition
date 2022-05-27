import numpy as np
import matplotlib.pyplot as plt

def update(X, y ,w, b, lr):
    cnt = 0
    flag = 1
    while flag:
        flag = 0
        for i in range(100):
            if y[i] * (w.dot(X[i]) + b) <= 0 :
                w += lr * y[i] * X[i]
                b += lr * y[i]
                flag = 1
                cnt += 1
    print(f"{w[0]:4.2}x_1+{w[1]:4.2}x_2+{b:4.2}=0")
    print(f"The number of iterations is {cnt}")

    res_x = np.linspace(0, 10, 500)
    res_y = - (w[0] * res_x + b) / w[1]

    plt.plot(res_x,res_y)


if __name__ == '__main__':
    x_n = np.random.uniform(0, 5, [50, 2])
    x_p = np.random.uniform(5, 10, [50, 2])
    
    X = np.append(x_n, x_p).reshape((100,2))
    y = np.array([-1 if i < 50 else 1 for i in range(100)])
    
    plt.scatter(x_n[:,0], x_n[:,1], marker = 'o', color = 'blue')
    plt.scatter(x_p[:,0], x_p[:,1], marker = 'o', color = 'green')
    
    lr = 0.1
    w = np.zeros(2)
    b = 0
    update(X, y, w, b, lr)
     
    w = np.ones(2)
    b = 1
    update(X, y, w, b, lr)

    plt.ylim(0, 10)
    plt.show()