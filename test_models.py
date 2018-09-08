import os
import sys
import time
import torch
import torchvision
import torchvision.transforms as transforms

from mobilenet_models.mobilenet import MobileNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_model(model_str, path_comps):
    print("---------------------------------------------------------------")
    print("|",model_str,'|', end='')

    model_dict = torch.load(os.path.join(*path_comps))
    model = MobileNet(model_dict['block_cfg'],
                      conv1_filter_count=model_dict['cfc'],
                      linear_filter_count=model_dict['lfc'])
    model.load_state_dict(model_dict['state_dict'])
    param_size = count_parameters(model)
    print('\t%d\t   | %.2f    |' % (param_size,model_dict['acc'] ) )
    return model


def test_model(net,testloader,device):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print("{0:.0f}% tested".format(100*total/10000), end='')
            print('| Accuracy: %.3f%% (%d/%d)'
                  % (100.*correct/total, correct, total))
        end = time.time()
        print('Time for testing 10000 pictures: %.2f seconds' % (end-start))
        return 100.*correct/total


def test_data_prep():
    print('==> Preparing test data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return testloader


def main():
    device_cpu = torch.device("cpu")

    print("---------------------------------------------------------------")
    print('|  MODEL VARIANTS              | # OF PARAMETERS   | ACCURACY |')

    vanilla_model = load_model('Unmodified mobilenet     (1)',
                               ['mobilenet_models','ckpts','vanilla_mobilenet.t7'])
    pruned_accurate_model = load_model('Pruned accurate mobilenet(2)',
                                ['mobilenet_models','ckpts', 'pruned_accurate_mobilenet.t7'])
    print("---------------------------------------------------------------")
    test_loader = test_data_prep()
    last_sel = -1
    while last_sel != 0:
        print('select model variant for testing: [1/2] or 0 to quit:', end="", flush=True)
        last_sel = sys.stdin.readline()
        try:
            last_sel = int(last_sel)
        except ValueError:
            continue
        if last_sel == 0:
            break
        if last_sel == 1:
            test_model(vanilla_model, test_loader, device_cpu)
        elif last_sel == 2:
            test_model(pruned_accurate_model, test_loader, device_cpu)


if __name__ == '__main__':
    main()
