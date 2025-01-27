import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import pandas as pd

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = '42'


class MyNAG(Optimizer):
    def __init__(self,
                 params,
                 lr=0.01,
                 momentum=0.9):
        if lr < 0.0:
            raise ValueError(f"Некорректное значение lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Некорректное значение momentum: {momentum}")

        defaults = dict(lr=lr,
                        momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = grad.clone().detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)

                nesterov_grad = grad.add(buf, alpha=momentum)
                p.data.add_(nesterov_grad, alpha=-lr)

        return loss


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class FlexibleResNet(nn.Module):
    def __init__(self, block_type, num_blocks_list, num_classes=2,
                 activation_name="relu", base_channels=64):
        super().__init__()
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }
        self.activation = act_map.get(activation_name, nn.ReLU)
        self.in_planes = base_channels

        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.act1 = self.activation()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_type, base_channels, num_blocks_list[0], 1)
        self.layer2 = self._make_layer(block_type, base_channels * 2, num_blocks_list[1], 2)
        self.layer3 = self._make_layer(block_type, base_channels * 4, num_blocks_list[2], 2)
        self.layer4 = self._make_layer(block_type, base_channels * 8, num_blocks_list[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        final_planes = base_channels * 8 * (block_type.expansion if hasattr(block_type, 'expansion') else 1)
        self.fc = nn.Linear(final_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for s in strides:
            layers.append(block(self.in_planes, planes, s, self.activation))
            self.in_planes = planes * block.expansion if hasattr(block, 'expansion') else planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Individual:
    def __init__(self, genes):
        self.genes = genes  # [block_type, n1, n2, n3, n4, activation, base_ch]
        self.fitness = 0.0


class GeneticOptimizer:
    def __init__(self, pop_size=15, n_generations=10, cx_prob=0.5, mut_prob=0.2,
                 devices=[0], n_epochs=5):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.devices = devices
        self.n_epochs = n_epochs
        self.hall_of_fame = None

    def _create_individual(self):
        genes = [
            random.randint(0, 1),  # block_type (0=Basic, 1=Bottleneck)
            random.randint(2, 6),  # n1
            random.randint(2, 6),  # n2
            random.randint(2, 6),  # n3
            random.randint(2, 6),  # n4
            random.choice(["relu", "tanh", "sigmoid"]),  # activation
            random.choice([32, 64, 128])  # base_channels
        ]
        return Individual(genes)

    def initialize_population(self):
        return [self._create_individual() for _ in range(self.pop_size)]

    def evaluate_population(self, population, train_loader, val_loader):
        with ProcessPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = []
            for i, ind in enumerate(population):
                device = self.devices[i % len(self.devices)]
                futures.append(executor.submit(
                    evaluate_individual,
                    ind.genes,
                    train_loader,
                    val_loader,
                    device,
                    self.n_epochs
                ))

            for future, ind in zip(futures, population):
                ind.fitness = future.result()

        # Update hall of fame
        current_best = max(population, key=lambda x: x.fitness)
        if not self.hall_of_fame or current_best.fitness > self.hall_of_fame.fitness:
            self.hall_of_fame = current_best

    def _select_parent(self, population):
        tournament = random.sample(population, 3)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1, parent2):
        if random.random() > self.cx_prob:
            return parent1, parent2

        cx_point = random.randint(1, len(parent1.genes) - 1)
        child1 = Individual(parent1.genes[:cx_point] + parent2.genes[cx_point:])
        child2 = Individual(parent2.genes[:cx_point] + parent1.genes[cx_point:])
        return child1, child2

    def _mutate(self, individual):
        for i in range(len(individual.genes)):
            if random.random() < self.mut_prob:
                if i == 0:
                    individual.genes[i] = random.randint(0, 1)
                elif 1 <= i <= 4:
                    individual.genes[i] = random.randint(2, 6)
                elif i == 5:
                    individual.genes[i] = random.choice(["relu", "tanh", "sigmoid"])
                elif i == 6:
                    individual.genes[i] = random.choice([32, 64, 128])
        return individual

    def evolve(self, train_loader, val_loader):
        population = self.initialize_population()

        for gen in range(self.n_generations):
            self.evaluate_population(population, train_loader, val_loader)

            fitnesses = [ind.fitness for ind in population]
            print(f"\nGeneration {gen + 1}/{self.n_generations}")
            print(f"Max Fitness: {max(fitnesses):.2f}")
            print(f"Avg Fitness: {np.mean(fitnesses):.2f}")
            print(f"Min Fitness: {min(fitnesses):.2f}")

            new_pop = []
            while len(new_pop) < self.pop_size:
                parent1 = self._select_parent(population)
                parent2 = self._select_parent(population)
                child1, child2 = self._crossover(parent1, parent2)

                for child in [child1, child2]:
                    if len(new_pop) >= self.pop_size:
                        break
                    self._mutate(child)
                    new_pop.append(child)

            population = new_pop

        return self.hall_of_fame


def evaluate_individual(genes, train_loader, val_loader, device, n_epochs=5):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    block_type = BasicBlock if genes[0] == 0 else Bottleneck
    num_blocks = genes[1:5]
    activation = genes[5]
    base_ch = genes[6]

    model = FlexibleResNet(
        block_type=block_type,
        num_blocks_list=num_blocks,
        activation_name=activation,
        base_channels=base_ch
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = MyNAG(model.parameters(), lr=0.01)

    best_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc

    return best_acc


def create_dataframe(data_dir):
    data = []
    for split in ['train', 'test']:
        for label in ['benign', 'malignant']:
            folder = os.path.join(data_dir, split, label)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'path': os.path.join(folder, fname),
                        'label': 0 if label == 'benign' else 1,
                        'split': split
                    })
    return pd.DataFrame(data)


if __name__ == '__main__':
    DATA_DIR = "data"
    BATCH_SIZE = 64
    DEVICES = [0, 1, 2]
    N_GENERATIONS = 10
    POP_SIZE = 15

    df = create_dataframe(DATA_DIR)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        ImageDataset(train_df, transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        ImageDataset(val_df, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    ga = GeneticOptimizer(
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        devices=DEVICES,
        n_epochs=50
    )

    best = ga.evolve(train_loader, val_loader)

    print("\nBest Architecture:")
    print(f"Block Type: {'Bottleneck' if best.genes[0] else 'BasicBlock'}")
    print(f"Num Blocks: {best.genes[1:5]}")
    print(f"Activation: {best.genes[5]}")
    print(f"Base Channels: {best.genes[6]}")
    print(f"Validation Accuracy: {best.fitness:.2f}%")