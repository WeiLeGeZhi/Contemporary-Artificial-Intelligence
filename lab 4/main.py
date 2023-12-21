import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
import argparse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models import LSTM, RNN, GRU
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        description = torch.tensor(self.data.iloc[index]['input'], dtype=torch.float32)
        diagnosis = torch.tensor(self.data.iloc[index]['output'], dtype=torch.float32)
        return description, diagnosis


def convert_int_list(int_list):
    str_list = list(map(str, int_list))
    space_separated_str = ' '.join(str_list)
    return space_separated_str


def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure


def sequence_to_text(sequence, tokenizer):
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
    output_text = ' '.join([reverse_word_index.get(i, '') for i in sequence])
    if len(output_text) == 1:
        output_text = output_text.join('<OOV>')
    return output_text


def div_str_and_vec(df):
    df_vec = pd.DataFrame()
    df_vec['input'] = df['input']
    df_vec['output'] = df['output']
    df_vec = df_vec.reset_index(drop=True)
    df = df.drop(columns=['index', 'input', 'output'])
    return df_vec, df


def evaluate(model, data_loader):
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            targets = targets.float()

            outputs_int = torch.round(outputs).long().tolist()
            targets_int = targets.long().tolist()

            all_outputs.extend(outputs_int)
            all_targets.extend(targets_int)

    return all_outputs, all_targets


def test_model(model, data_loader, std_diagnosis, text_tokenizer):
    outputs, targets = evaluate(model, data_loader)
    assert len(outputs) == len(targets) == len(std_diagnosis)
    rouge = []
    all_loss = []
    for i in range(len(outputs)):
        this_loss = np.linalg.norm(np.array(outputs[i]) - np.array(targets[i]))
        all_loss.append(this_loss)
        rouge_output = sequence_to_text(outputs[i],text_tokenizer)
        rouge_valid = calculate_rouge(std_diagnosis[i], rouge_output)
        rouge.append(rouge_valid)
    rouge_avg = sum(rouge) / len(rouge)
    loss_avg = sum(all_loss) / len(all_loss)
    return rouge_avg, loss_avg


parser = argparse.ArgumentParser(description="Command")
parser.add_argument('--model', default="RNN", type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--optimizer', default="Adam", type=str)
args = parser.parse_args()


train_size = 15000
valid_size = 3000
vocab_size = 1500
max_len = 50
train_and_valid = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
combined_data = pd.concat([train_and_valid, test])
combined_texts = combined_data['description'] + ' ' + combined_data['diagnosis']
text_tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
text_tokenizer.fit_on_texts(combined_texts)
train_desc_seq = text_tokenizer.texts_to_sequences(train_and_valid['description'])
train_diag_seq = text_tokenizer.texts_to_sequences(train_and_valid['diagnosis'])
test_desc_seq = text_tokenizer.texts_to_sequences(test['description'])
test_diag_seq = text_tokenizer.texts_to_sequences(test['diagnosis'])
train_desc_seq = pad_sequences(train_desc_seq, maxlen=max_len, padding='post')
train_diag_seq = pad_sequences(train_diag_seq, maxlen=max_len, padding='post')
test_desc_seq = pad_sequences(test_desc_seq, maxlen=max_len, padding='post')
test_diag_seq = pad_sequences(test_diag_seq, maxlen=max_len, padding='post')
train_and_valid['input'] = [seq for seq in train_desc_seq]
train_and_valid['output'] = [seq for seq in train_diag_seq]
test['input'] = [seq for seq in test_desc_seq]
test['output'] = [seq for seq in test_diag_seq]
train_indices, valid_indices = random_split(range(len(train_and_valid)), [train_size, valid_size])
train_indices = list(train_indices)
valid_indices = list(valid_indices)
train = train_and_valid.iloc[train_indices].reset_index(drop=True)
valid = train_and_valid.iloc[valid_indices].reset_index(drop=True)

train_vec, train = div_str_and_vec(train)
valid_vec, valid = div_str_and_vec(valid)
test_vec, test = div_str_and_vec(test)

train_dataset = MyDataset(train_vec)
valid_dataset = MyDataset(valid_vec)
test_dataset = MyDataset(test_vec)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if args.model == 'RNN':
    model = RNN.SimpleRNN(input_size=max_len, hidden_size=64, output_size=max_len)
elif args.model == 'LSTM':
    model = LSTM.SimpleLSTM(input_size=max_len, hidden_size=64, output_size=max_len)
elif args.model == 'GRU':
    model = GRU.SimpleGRU(input_size=max_len, hidden_size=64, output_size=max_len)
else:
    raise ValueError('Unknown model')


if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise ValueError("Unknown optimizer")
loss_function = nn.MSELoss()

valid_losses = []
valid_rouges = []

num_epochs = args.epoch
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.float()
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    valid_rouge, valid_loss = test_model(model, valid_loader, valid['diagnosis'], text_tokenizer)
    valid_losses.append(valid_loss)
    valid_rouges.append(valid_rouge)

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {valid_loss}, rouge: {valid_rouge}')


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), valid_losses, marker='o')
plt.title(f'{args.model} Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
# plt.savefig(f'{args.model}Loss.png')
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), valid_rouges, marker='o', color='r')
plt.title(f'{args.model} Validation ROUGE')
plt.xlabel('Epoch')
plt.ylabel('ROUGE Score')
plt.grid(True)
plt.savefig(f'./result/{args.model}Result.png')
    
test_rouge, test_loss = test_model(model, test_loader, test['diagnosis'], text_tokenizer)
print(f'Test Loss: {test_loss}, ROUGE Score on Test Set: {test_rouge}')
