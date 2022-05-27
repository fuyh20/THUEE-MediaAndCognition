from dataset import tiny_caltech35
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import BasicBlock, base_model, ResNet
import torchvision.models as models
import matplotlib.pyplot as plt
import utils


def main(config):
    transform_train = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = tiny_caltech35(transform=transform_train, used_data=['train'])
    # train_dataset = tiny_caltech35(transform=transform_train, used_data=['train', 'val'])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    # model = base_model(class_num=config.class_num)
    model = ResNet(BasicBlock, [2, 2, 2, 2], 7)
    
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=1e-4, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # you may need train_numbers and train_losses to visualize something
    train_numbers, train_losses = train(config, train_loader, model, optimizer, scheduler, criterion)

    plt.plot(train_numbers, train_losses)
    plt.xlabel("train_numbers")
    plt.ylabel("train_losses")
    plt.show()

    # you can use validation dataset to adjust hyper-parameters

    # train_accuracy = test(config, train_loader, model)
    print('===========================')
    val_accuracy = test(config, val_loader, model)
    print("val accuracy:{}%".format(val_accuracy * 100))
    # test_accuracy = test(config, test_loader, model)
    # print("test accuracy:{}%".format(test_accuracy * 100))


def train(config, data_loader, model, optimizer, scheduler, criterion):
    model.train()
    model.cuda()
    train_losses = []
    train_numbers = []
    counter = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += data.shape[0]
            accuracy = ((label == output.argmax(dim=1)).sum()).cpu().numpy() * 1.0 / output.shape[0]
            if batch_idx % 20 == 0:
                print('Train Epoch: {} / {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch, config.epochs, batch_idx * len(data), len(data_loader.dataset),
                                          100. * batch_idx / len(data_loader), loss.item(), accuracy.item()))
                train_losses.append(loss.item())
                train_numbers.append(counter)
            
        scheduler.step()
        torch.save(model.state_dict(), './model.pth')
    return train_numbers, train_losses


def test(config, data_loader, model):
    model.eval()
    model.cuda()
    correct = 0
    feature = torch.Tensor([]).cuda()
    labels = torch.Tensor([]).cuda()
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)

            """store data for PCA analyse """
            if(config.pca):
                feature = torch.cat((feature, model.feature), 0)
                labels = torch.cat((labels, label), 0)
            
            pred = output.argmax(dim=1)
            correct += ((pred == label).sum()).cpu().numpy()

    """PCA analyse and visualize"""
    if(config.pca):    
        utils.PCA_draw(feature, labels)
        plt.show()

    accuracy = correct * 1.0 / len(data_loader.dataset)
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[112, 112])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 25])
    parser.add_argument('--pca', action='store_true', default=False)

    config = parser.parse_args()
    main(config)