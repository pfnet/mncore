import torch
import torchvision.transforms as transforms
from torchvision import datasets


def mnist_loaders(batch_size, eval_batch_size):
    # MLSDK requires input to be passed as a dictionary
    def list_to_dict(batch):
        batch = torch.utils.data.default_collate(batch)
        return {"x": batch[0], "t": batch[1]}

    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # Hack to avoid downloading a broken data from the first mirror.
    # TODO(hamaji): Remove this hack after updating torchvision.
    datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]
    train_dataset = datasets.MNIST(
        "/tmp", train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=list_to_dict,
    )
    eval_dataset = datasets.MNIST(
        "/tmp",
        train=False,
        transform=transform,
        download=True,
    )
    # In evaluation, using ``drop_last`` breaks its validity, so we instead use batch
    # size that divides the number of validation images (10,000) to correctly evaluate
    # the model.
    assert len(eval_dataset) % eval_batch_size == 0
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=list_to_dict,
    )
    return train_loader, eval_loader


class MNCoreClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 256)
        self.linear2 = torch.nn.Linear(256, 10)

    def forward(self, x, t, **args):
        x_reshaped = x.reshape(x.size(0), -1)
        x1 = self.linear1(x_reshaped)
        x2 = torch.nn.functional.relu(x1)
        y = self.linear2(x2)
        loss = torch.nn.functional.cross_entropy(y, t)
        if self.training:
            return {"loss": loss}
        else:
            return {"y": y, "loss": loss}
