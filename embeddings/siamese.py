from torchsummary import summary
from siamese_network import SiameseNetwork, ConvAutoencoder

import torch
from data_loaders import get_dataloaders
import training

def main():
    # https://nextjournal.com/gkoehler/pytorch-mnist
    n_epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.02
    momentum = 0.55
    log_interval = 10

    torch.backends.cudnn.enabled = False
    #random_seed = 1
    #torch.manual_seed(random_seed)



    # Load dataset
    train_loader, test_loader = get_dataloaders(batch_size_train, batch_size_test)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    model = ConvAutoencoder().to('cuda')
    summary(model, (1, 28, 28))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer.zero_grad()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(n_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        training.train_loop(train_loader, model, loss_fn, optimizer)
        training.test_loop(test_loader, model, loss_fn)
    print("Done!")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()