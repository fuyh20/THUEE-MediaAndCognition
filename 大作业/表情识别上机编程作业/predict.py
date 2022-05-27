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
    transform_test = transforms.Compose([
    transforms.Resize(config.image_size, interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    model= ResNet(BasicBlock, [2, 2, 2, 2], 7)
    model.load_state_dict(torch.load('./model.pth'))
    
    val_accuracy = test(val_loader, model)
    test_accuracy = test(test_loader, model)

    print('===========================')
    print("val accuracy:{}%".format(val_accuracy * 100))
    print("test accuracy:{}%".format(test_accuracy * 100))


def test(data_loader, model):
    model.eval()
    model.cuda()
    correct = 0
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += ((pred == label).sum()).cpu().numpy()


    accuracy = correct * 1.0 / len(data_loader.dataset)
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[112, 112])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=7)
    config = parser.parse_args()
    main(config)