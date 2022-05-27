import numpy as np
from utils import load_dataset, extract_features, hog_feature
import scipy.io as io
import matplotlib.pyplot as plt
from svm_classifier import LinearSVM


# load train val test data
class_names = ['angry', 'disgust', 'fear',
               'happy', 'neutral', 'sad', 'surprise']
FER_path = './dataset/fer'
X_train, Y_train = load_dataset(FER_path, class_names, 'train')
X_val, Y_val = load_dataset(FER_path, class_names, 'val')
X_test, Y_test = load_dataset(FER_path, class_names, 'test')
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# change this when you work on different questions
question = 'deep'  # choose from pixel, hog, deep


if question == 'pixel':
    # reshape 2D images to 1D
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

elif question == 'hog':
    feature_fns = [hog_feature]
    X_train = extract_features(X_train, feature_fns)
    X_val = extract_features(X_val, feature_fns)
    X_test = extract_features(X_test, feature_fns)

elif question == 'deep':
    X_train = io.loadmat('./dataset/train_features.mat')['features']
    X_val = io.loadmat('./dataset/val_features.mat')['features']
    X_test = io.loadmat('./dataset/test_features.mat')['features']


# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train, axis=0, keepdims=True)
X_train -= mean_feat
X_val -= mean_feat
X_test -= mean_feat
# Preprocessing: Divide by standard deviation
std_feat = np.std(X_train, axis=0, keepdims=True)
X_train /= std_feat
X_val /= std_feat
X_test /= std_feat

# augmented vector
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])


lrs = [10 ** i for i in range(-8, 2)]
regs = [10 ** i for i in range(-5, 5)]
num_iters = [100, 300, 500, 1000, 1500, 2000, 2500, 3000]

train_accs = []
val_accs = []

# for lr in lrs:
#     """可以换成其他超参数进行比较"""
#     svm = LinearSVM()
#     print(f"learning_rate: {lr},reg: {10}, num_iters: {2000}")
#     loss_svm = svm.train(
#         X_train, Y_train, learning_rate=lr, reg=10, num_iters=2000)

#     # evaluate the performance on both the training and validation set
#     y_train_pred = svm.predict(X_train)
#     train_acc = np.mean(Y_train == y_train_pred)
#     print('training accuracy: %f' % (train_acc, ))

#     y_val_pred = svm.predict(X_val)
#     val_acc = np.mean(Y_val == y_val_pred)
#     print('validation accuracy: %f' % (val_acc, ))

#     train_accs.append(train_acc)
#     val_accs.append(val_acc)

# """作出不同超参数对应准确率的图形"""
# plt.scatter(lrs, train_accs, label="train")
# plt.scatter(lrs, val_accs, label="validation")
# plt.xscale("log")
# plt.xlabel("learning_rate")
# plt.ylabel("accuracy")
# plt.text(1.e-4, 0.6, "reg=10, num_iters = 2000")
# plt.legend()
# plt.show()


# evaluate the performance on both the training and validation set
svm = LinearSVM()
loss_svm = svm.train(X_train, Y_train, learning_rate=1.e-4, reg=10, num_iters=2500)
iters = np.linspace(1, 2500, 2500)
plt.plot(iters, loss_svm)
plt.xlabel("iters")
plt.ylabel("loss")
y_train_pred = svm.predict(X_train)
print('training accuracy: %f' % (np.mean(Y_train == y_train_pred), ))

y_val_pred = svm.predict(X_val)
print('validation accuracy: %f' % (np.mean(Y_val == y_val_pred), ))


# evaluate the performance on test set
final_test = True
if final_test:
    y_test_pred = svm.predict(X_test)
    print('test accuracy: %f' % (np.mean(Y_test == y_test_pred), ))
    for i, class_name in enumerate(class_names):
        class_i_Y = Y_test[Y_test==i]
        class_i_y_test_pred = y_test_pred[Y_test==i]
        acc = np.mean((class_i_Y==class_i_y_test_pred)) 
        print(f"class {class_name} accurary: {acc}")

plt.show()