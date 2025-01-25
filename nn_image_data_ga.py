import torch.multiprocessing as mp

import os
import multiprocessing
import random
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms

from torchvision import transforms, models
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

WORKER_DEVICE = None

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["PYTHONHASHSEED"] = '42'

def create_dataframe(data_dir):
    data = []
    for split in ['train', 'test']:
        for label in ['benign', 'malignant']:
            folder = os.path.join(data_dir, split, label)
            for filename in os.listdir(folder):
                if filename.endswith(('.png', '.jpg', '.jpeg')):  # Убедимся, что это изображение
                    filepath = os.path.join(folder, filename)
                    data.append({
                        'path': filepath,
                        'label': 0 if label == 'benign' else 1,
                        'split': split
                    })
    return pd.DataFrame(data)

data_dir = "data"
dataframe = create_dataframe(data_dir)
print(dataframe.head())

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx]['label']
        index = self.dataframe.index[idx]
        image = torch.load(f'processed/{index}.pt')

        return image, label
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_dataloader(dataframe, transform, batch_size=32, shuffle=True):
    dataset = ImageDataset(dataframe, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_df = dataframe[dataframe['split'] == 'train'].copy()
test_df = dataframe[dataframe['split'] == 'test'].copy()

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

train_loader = get_dataloader(train_df, transform=transform, batch_size=64, shuffle=True)
val_loader   = get_dataloader(val_df,   transform=transform, batch_size=64, shuffle=False)
test_loader  = get_dataloader(test_df,  transform=transform, batch_size=64, shuffle=False)

def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Статистика
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, activation=nn.ReLU):
        super(BasicBlock, self).__init__()
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
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
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
        super(Bottleneck, self).__init__()
        self.activation = activation()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
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
    def __init__(self, block_type, num_blocks_list, 
                 num_classes=2, 
                 activation_name="relu",
                 base_channels=64):
        """
        block_type: класс блока (BasicBlock или Bottleneck)
        num_blocks_list: список [n1, n2, n3, n4] (сколько блоков в каждом из 4-х stage)
        activation_name: "relu" / "tanh" / "sigmoid"
        base_channels: кол-во каналов в первой свёртке
        """
        super(FlexibleResNet, self).__init__()

        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }
        self.activation = act_map.get(activation_name, nn.ReLU)

        self.in_planes = base_channels
        self.block = block_type

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.act1 = self.activation()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.block, base_channels,      num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(self.block, base_channels*2,    num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(self.block, base_channels*4,    num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(self.block, base_channels*8,    num_blocks_list[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        final_planes = (base_channels*8) * (block_type.expansion if hasattr(block_type, 'expansion') else 1)
        self.fc = nn.Linear(final_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []

        strides = [stride] + [1]*(num_blocks-1)
        for s in strides:
            layers.append(block(self.in_planes, planes, s, activation=self.activation))
            if hasattr(block, 'expansion'):
                self.in_planes = planes * block.expansion
            else:
                self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def init_block_type():
    # 0 -> BasicBlock, 1 -> Bottleneck
    return random.randint(0,1)

def init_num_blocks():
    return random.randint(2,6)

def init_activation():
    return random.choice(["relu", "tanh", "sigmoid"])

def init_base_channels():
    return random.choice([32, 64, 128])

def init_individual():
    """
    Примерная структура особи (индивида):
      [block_type,
       n_blocks_1, n_blocks_2, n_blocks_3, n_blocks_4,
       activation_name,
       base_channels]
    """
    return [
        init_block_type(),
        init_num_blocks(), init_num_blocks(), init_num_blocks(), init_num_blocks(),
        init_activation(),
        init_base_channels()
    ]

toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_individual(ind):
    # ind = [block_type, n1, n2, n3, n4, act_name, base_ch]
    block_type = BasicBlock if ind[0] == 0 else Bottleneck
    num_blocks_list = [ind[1], ind[2], ind[3], ind[4]]
    act_name = ind[5]
    base_ch = ind[6]
    return block_type, num_blocks_list, act_name, base_ch

def init_worker(devs):
    global WORKER_DEVICE
    pid = multiprocessing.current_process()._identity[0] - 1
    WORKER_DEVICE = devs[pid]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = '42'

def evaluate_on_device(ind):
    global TRAIN_LOADER, VAL_LOADER, WORKER_DEVICE
    device = torch.device(f'cuda:{WORKER_DEVICE}')

    block_type, num_blocks_list, act_name, base_ch = decode_individual(ind)

    model = FlexibleResNet(block_type, num_blocks_list, 2, act_name, base_ch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, nesterov=True, momentum=0.9)

    best_val_acc = 0
    
    num_epochs = 25
    for epoch in range(num_epochs):
        train_model(model, criterion, optimizer, TRAIN_LOADER, device)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in VAL_LOADER:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        val_acc = 100.0*correct/total if total else 0
        best_val_acc = max(best_val_acc, val_acc)
    
    print(best_val_acc)
    return (best_val_acc,)

def cx_one_point(ind1, ind2):
    cxpoint = random.randint(1, len(ind1)-1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2

def mut_individual(ind, indpb=0.2):
    """
    С вероятностью indpb мутируем каждый ген
    """
    for i in range(len(ind)):
        if random.random() < indpb:
            if i == 0:
                ind[i] = init_block_type()
            elif 1 <= i <= 4:
                ind[i] = init_num_blocks()
            elif i == 5:
                ind[i] = init_activation()
            elif i == 6:
                ind[i] = init_base_channels()
    return (ind,)

toolbox.register("mate", cx_one_point)
toolbox.register("mutate", mut_individual, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

devices = [0, 1, 2]
TRAIN_LOADER, VAL_LOADER = train_loader, val_loader

def evaluate_deap(ind):
    return evaluate_on_device(ind)

def run_ga(n_gen=10, pop_size=8):
    global TRAIN_LOADER, VAL_LOADER, devices

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # сохранить лучшее решение

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("mean", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    n_devices = len(devices)
    
    pool = multiprocessing.Pool(
        processes=n_devices,
        initializer=init_worker,
        initargs=(devices,)
    )
    
    toolbox.register("map", pool.map)
    
    toolbox.register("evaluate", evaluate_deap)

    pop, logbook = algorithms.eaSimple(
        population=pop,
        toolbox=toolbox,
        cxpb=0.5,   # вероятность скрещивания
        mutpb=0.3,  # вероятность мутации
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    pool.close()
    pool.join()

    print("\nЛучший индивид:")
    print(hof[0])
    print("Лучший fitness (val_acc) =", hof[0].fitness.values[0])

    # Декодируем в удобочитаемые параметры
    best_block_type, best_blocks_list, best_act, best_base_ch = decode_individual(hof[0])
    print(f"Тип блока: {best_block_type.__name__}, "
          f"Блоки: {best_blocks_list}, "
          f"Активация: {best_act}, "
          f"Base Channels: {best_base_ch}")

    return pop, hof, logbook

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    final_pop, hall_of_fame, logs = run_ga(n_gen=10, pop_size=15)
    print(hall_of_fame[0])
    
    with open('output.txt', 'w') as f:
        f.write(str(hall_of_fame))
        f.write('\n\n\n')
        f.write(str(logs))