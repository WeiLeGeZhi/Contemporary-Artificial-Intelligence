import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from model import Multimodel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image


def load_data(row):
    data_path = './data'
    guid = int(row['guid'])
    txt_file_path = os.path.join(data_path, f"{guid}.txt")
    img_file_path = os.path.join(data_path, f"{guid}.jpg")

    with open(txt_file_path, 'r', encoding='latin-1') as txt_file:
        txt_content = txt_file.read()

    img_tensor = load_image(img_file_path)

    return pd.Series({'guid': guid, 'txt': txt_content, 'fig': img_tensor, 'tag': row['tag']})


def get_dataframe(path):
    df = pd.read_csv(path, sep=',')
    frame = df.apply(load_data, axis=1)
    return frame


def test_acc(model, test_data, mode, if_test):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img, txt, labels in test_data:
            outputs = model(img, txt, mode)
            labels = labels.long()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            if not if_test:
                val_loss = criterion(outputs, labels)
            else:
                val_loss = 0.0

    accuracy = accuracy_score(all_labels, all_preds)
    return val_loss, accuracy, all_preds


class MyDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_data = self.data.iloc[index]['fig']
        txt_data = self.data.iloc[index]['docvec']
        label = self.data.iloc[index]['label']

        if self.transform:
            img_data = self.transform(img_data)

        return img_data, txt_data, label


if __name__ == '__main__':
    train_path = './train.txt'
    test_path = './test_without_label.txt'
    train_and_valid = get_dataframe(train_path)
    test = get_dataframe(test_path)
    label_encoder = LabelEncoder()
    train_and_valid['label'] = label_encoder.fit_transform(train_and_valid['tag'])
    combined_corpus = pd.concat([train_and_valid['txt'], test['txt']], axis=0)
    vocab_size = 1500
    max_len = 50
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(combined_corpus)
    train_seq = tokenizer.texts_to_sequences(train_and_valid['txt'])
    test_seq = tokenizer.texts_to_sequences(test['txt'])
    train_seq = pad_sequences(train_seq, maxlen=max_len, padding='post')
    test_seq = pad_sequences(test_seq, maxlen=max_len, padding='post')
    train_and_valid['docvec'] = [seq for seq in train_seq]
    test['docvec'] = [seq for seq in test_seq]
    test['label'] = -1

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_set, valid_set = train_test_split(train_and_valid, test_size=0.2, random_state=42)

    train_dataset = MyDataset(train_set, transform=transform)
    valid_dataset = MyDataset(valid_set, transform=transform)
    test_dataset = MyDataset(test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Multimodel.MultiModel(len(tokenizer.word_index) + 1, 100, 100, 128, 3)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()

        for img, txt, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(img, txt, 0)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_loss, accuracy, val_preds = test_acc(model, valid_loader, 0, False)
        # print(val_preds)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Accuracy: {accuracy}')

    txt_only_loss, txt_only_accuracy, txt_only_preds = test_acc(model, valid_loader, 1, False)
    print(
        f'Text Only Loss: {txt_only_loss.item()}, Accuracy: {txt_only_accuracy}')
    print(txt_only_preds)
    img_only_loss, img_only_accuracy, img_only_preds = test_acc(model, valid_loader, 2, False)
    print(
        f'Image Only Loss: {img_only_loss.item()}, Accuracy: {img_only_accuracy}')

    test_loss, test_accuracy, test_preds = test_acc(model, test_loader, 0, True)
    # print(test_preds)
    test['tag'] = label_encoder.inverse_transform(test_preds)
    columns_to_save = ['guid', 'tag']
    test[columns_to_save].to_csv('test_without_label.txt', index=False, header=True, sep=',')
