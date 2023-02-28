import numpy as np
import torch
import torchinfo
import torchvision
import tqdm
import torchviz

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Lenet5(torch.nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()

        self.feature_exctractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6,
                            kernel_size=5, padding=2),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16,
                            kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=16, out_channels=120,
                            kernel_size=5, stride=1),
            torch.nn.Tanh()
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.feature_exctractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


def accuracy_fn(y_pred, y_true):
    y_pred_tags = y_pred.argmax(dim=1)
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc*100


def train(model, train_loader, test_loader, epochs=50):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model_name = 'pytorch_lenet5.pth'
    histories = {'loss': [], 'accuracy': []}

    for epoch in tqdm.tqdm(range(1, epochs+1)):
        model.train()
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            pred = model(data)
            train_loss = loss(pred, labels)
            train_accuracy = accuracy_fn(pred, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        model.eval()
        for data, labels in tqdm.tqdm(test_loader):
            data = data.to(device)
            labels = labels.to(device)

            with torch.inference_mode():
                pred = model(data)
                test_loss = loss(pred, labels)
                test_accuracy = accuracy_fn(pred, labels)
                histories['loss'].append(test_loss.item())
                histories['accuracy'].append(test_accuracy.item())

        if epoch % 5 == 0:
            print()
            print(f'Epoch: {epoch:02d}')
            print(
                f'Loss: {train_loss:.8f},Test Loss: {test_loss:.8f}')
            print(
                f'Accuracy: {train_accuracy:.8f}%,Test Accuracy: {test_accuracy:.8f}%')
            print(
                f'Loss: {np.mean(histories["loss"]):.8f}, Accuracy: {np.mean(histories["accuracy"])}')

        if epoch % 10 == 0:
            print(f'saving model...')
            torch.save(model.state_dict(), model_name)

    return model


if __name__ == '__main__':
    mnist_trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_data_loader = torch.utils.data.DataLoader(
        mnist_trainset, batch_size=64, shuffle=True)
    mnist_testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_data_loader = torch.utils.data.DataLoader(
        mnist_testset, batch_size=64, shuffle=True)

    model = Lenet5().to(device)
    batch_size = 64
    torchinfo.summary(model, input_size=(batch_size, 1, 28, 28))

    train(model, train_data_loader, test_data_loader)
