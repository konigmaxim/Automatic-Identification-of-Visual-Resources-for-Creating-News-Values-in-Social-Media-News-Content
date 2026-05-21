import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

base_directory = "/Users/konigmaxim/Desktop/Курсовая 2026"
path_to_dataset = os.path.join(base_directory, "dataset.csv")
images_directory = os.path.join(base_directory, "images")

batch_size = 8
epochs = 30
learning_rate = 1e-4
patience = 5
n_folds = 5
label_columns = ["Image_Pos", "Image_Neg", "Image_Pers", "Image_Super", "Image_Elite", "Image_AA"]
num_classes = len(label_columns)

# СЧИТЫВАНИЕ ДАННЫХ (ровно 10 столбцов)
raw_df = pd.read_excel(path_to_dataset, header=0)
raw_df.columns = [
    "Source", "Link", "Headline", "Image_Pos", "Image_Neg", "Image_Pers", "Image_Super", "Image_Elite", "Image_AA"
]
for col in label_columns:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0).astype(int)

image_paths = [os.path.join(images_directory, f"image{i + 1}.png") for i in range(len(raw_df))] #путь к изображению
raw_df["image_path"] = image_paths

existing_rows = [] #проверка, что везде есть изображения
for _, row in raw_df.iterrows():
    if os.path.exists(row["image_path"]):
        existing_rows.append(row)
df = pd.DataFrame(existing_rows).reset_index(drop=True)

print(f"Всего изображений: {len(df)}")
print("Распределение по новостным ценностям:")
print(df[label_columns].sum())

# ТРАНСФОРМАЦИИ, ТЕНЗОР
#с аугментацией для обучения
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#без аугментации для валидации
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Датасет с конвертером
class NewsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        labels = row[label_columns].values.astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)

#модель ResNet50 без заморозки
def create_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

#Добавление весов для баланса
def get_pos_weights(dataframe):
    pos_weights = []
    for col in label_columns:
        n_pos = (dataframe[col] == 1).sum()
        n_neg = (dataframe[col] == 0).sum()
        weight = n_neg / n_pos if n_pos > 0 else 1.0
        pos_weights.append(weight)
    return torch.tensor(pos_weights)

#обучение одного из фолдов
def train_fold(train_df, val_df, fold_idx, device):
    print(f"\nФолд {fold_idx+1} из {n_folds}")

    train_dataset = NewsDataset(train_df, transform=train_transforms)
    val_dataset = NewsDataset(val_df, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = create_model().to(device)

    pos_weight = get_pos_weights(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_macro_f1 = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        #Обучение на аугментированном
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        #Валидация
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print(f"Эпоха {epoch+1}: Ошибка на обучении={avg_train_loss:.4f}, ошибка на валидации={avg_val_loss:.4f} Макро F1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Ранняя остановка после {epoch+1} эпох")
                break

    model.load_state_dict(best_state)

    #Финальная валидация с порогом 0.5
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\nМатрица ошибок по классам")
    for i, col in enumerate(label_columns):
        tp = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 1))
        fp = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 0))
        fn = np.sum((all_preds[:, i] == 0) & (all_labels[:, i] == 1))
        tn = np.sum((all_preds[:, i] == 0) & (all_labels[:, i] == 0))
        print(f"{col:10} | Верно распознано:{tp:3} | Ошибочно распознано:{fp:3} | Ошибочно не распознано:{fn:3} | Верно не распознано:{tn:3}")

    return best_macro_f1, model, best_state

#КРОСС-ВАЛИДАЦИЯ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    fold_f1, _, _ = train_fold(train_df, val_df, fold, device)
    fold_scores.append(fold_f1)

print("\nИтоги кросс-валидации:")
print(f"Mакро F1 за фолд: {[round(x,4) for x in fold_scores]}")
mean_f1 = np.mean(fold_scores)
print(f"Среднее Макро F1: {mean_f1:.4f}")

#ФИНАЛЬНОЕ ОБУЧЕНИЕ
final_model = create_model().to(device)
pos_weight_full = get_pos_weights(df).to(device)
criterion_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
optimizer_full = torch.optim.Adam(final_model.parameters(), lr=learning_rate)

train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
train_df_final = df.iloc[train_idx].reset_index(drop=True)
val_df_final = df.iloc[val_idx].reset_index(drop=True)

train_dataset_final = NewsDataset(train_df_final, transform=train_transforms)
val_dataset_final = NewsDataset(val_df_final, transform=val_transforms)
train_loader_final = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True)
val_loader_final = DataLoader(val_dataset_final, batch_size=batch_size, shuffle=False)

best_f1 = 0.0
best_state_final = None
no_improve = 0

# Списки для графиков обучения
final_train_losses = []
final_val_losses = []
final_val_f1s = []

for epoch in range(epochs):
    final_model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader_final, desc=f"Epoch {epoch+1} train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer_full.zero_grad()
        outputs = final_model(images)
        loss = criterion_full(outputs, labels)
        loss.backward()
        optimizer_full.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader_final)

    final_model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader_final:
            images, labels = images.to(device), labels.to(device)
            outputs = final_model(images)
            loss = criterion_full(outputs, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader_final)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Сохарнение истории для графиков
    final_train_losses.append(avg_train_loss)
    final_val_losses.append(avg_val_loss)
    final_val_f1s.append(macro_f1)

    print(f"Эпоха {epoch+1}: ошибка на обучении={avg_train_loss:.4f}, ошибка на валидации={avg_val_loss:.4f}, Макро F1={macro_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_state_final = final_model.state_dict().copy()
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Ранняя остановка после {epoch+1} эпох")
            break

final_model.load_state_dict(best_state_final)

# Финальная оценка (порог 0.5)
final_model.eval()
all_preds, all_labels = [], []
all_probs = []
with torch.no_grad():
    for images, labels in val_loader_final:
        images = images.to(device)
        outputs = final_model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs)
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

print("\nИтоговый результат без оптимизации порогов:")
print(classification_report(all_labels, all_preds, target_names=label_columns, zero_division=0))

#Оптимизатор порогов
print("\nОптимизатор порогов")
best_thresholds = {}
for i, col in enumerate(label_columns):
    best_f1_class = 0
    best_th = 0.5
    for thresh in np.arange(0.1, 0.95, 0.05):
        pred_class = (all_probs[:, i] > thresh).astype(int)
        f1 = f1_score(all_labels[:, i], pred_class, zero_division=0)
        if f1 > best_f1_class:
            best_f1_class = f1
            best_th = thresh
    best_thresholds[col] = best_th
    print(f"{col}: лучший порог = {best_th:.2f}, (F1={best_f1_class:.4f})")

y_pred_opt = np.zeros_like(all_probs)
for i, col in enumerate(label_columns):
    y_pred_opt[:, i] = (all_probs[:, i] > best_thresholds[col]).astype(int)

print("\nФинальный результат:")
print(classification_report(all_labels, y_pred_opt, target_names=label_columns, zero_division=0))

#Сохранение.
MODEL_PATH = os.path.join(base_directory, "final_model_balanced.pth")
torch.save(final_model.state_dict(), MODEL_PATH)





