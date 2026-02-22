#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T


def get_dataloaders(batch_size: int, num_workers: int):
    # CIFAR-10 normalization commonly used for CIFAR training
    # (These models were trained in CIFAR-land; no 224 resize.)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train(model, train_loader, test_loader, device, epochs, lr, weight_decay, out_dir: Path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_path = out_dir / "best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs} | test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

    print(f"Best test_acc={best_acc:.4f} saved to {best_path}")
    return best_path


def export_onnx(model, device, onnx_path: Path, opset: int = 12):
    model.eval().to(device)

    # Dummy input: (batch, channels, height, width) for CIFAR
    dummy = torch.randn(1, 3, 32, 32, device=device)

    # Stable names so Java side knows what to feed/fetch
    input_names = ["input"]
    output_names = ["logits"]

    dynamic_axes = {
        "input": {0: "batch"},
        "logits": {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported ONNX -> {onnx_path} (opset={opset})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cifar10_resnet20",
                        help="torch.hub entry name, e.g. cifar10_resnet20, cifar10_resnet56, cifar10_vgg16_bn, ...")
    parser.add_argument("--pretrained", action="store_true", help="load pretrained CIFAR-10 weights from torch.hub")
    parser.add_argument("--epochs", type=int, default=0, help="0 = no training, just export. >0 = fine-tune then export")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out", default="./out_cifar_onnx")
    parser.add_argument("--onnx", default="cifar10_model.onnx")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load CIFAR-10 pretrained model via torch.hub.
    # Repo explicitly supports torch.hub loading for CIFAR-10 models.  :contentReference[oaicite:2]{index=2}
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", args.model, pretrained=args.pretrained)
    model.to(device)

    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    if args.epochs > 0:
        best_path = train(model, train_loader, test_loader, device, args.epochs, args.lr, args.weight_decay, out_dir)
        # Reload best weights before exporting
        model.load_state_dict(torch.load(best_path, map_location=device))

    # Quick accuracy report (after optional training)
    acc = evaluate(model, test_loader, device)
    print(f"Final test_acc={acc:.4f}")

    onnx_path = out_dir / args.onnx
    export_onnx(model, device, onnx_path, opset=12)


if __name__ == "__main__":
    main()