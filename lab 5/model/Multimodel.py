import torch
import torch.nn as nn
from model import AlexNet, TextRNN


class MultiModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, single_outputdim, hidden_size, num_classes):
        super(MultiModel, self).__init__()
        self.img_model = AlexNet.AlexNet(single_outputdim, 3)
        self.txt_model = TextRNN.TextRNN(vocab_size, embedding_dim, hidden_size, single_outputdim)
        self.act = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(single_outputdim * 2, single_outputdim)
        self.fc2 = nn.Linear(single_outputdim, num_classes)

    def forward(self, img_data, txt_data, mode):
        img_out = self.img_model(img_data)
        txt_out = self.txt_model(txt_data)
        if mode == 1:
            img_out *= 0
        elif mode == 2:
            txt_out *= 0
        concated_out = torch.concat((img_out, txt_out), dim=1)
        out = self.fc1(self.act(concated_out))
        final_out = self.fc2(out)
        return final_out
