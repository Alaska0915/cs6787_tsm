import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

class Student(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, dropout):
        super(Student, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 30),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(30, num_classes)
        )
        
    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(out_1)
        out_3 = torch.mean(out_2, dim=(2, 3))  
        out_4 = self.classifier(out_3)
        
        return out_4  

class Teacher(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv, num_classes, dropout):
        super(Teacher, self).__init__()
        self.num_conv = num_conv
        self.conv_1 = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3)
        self.activation = nn.LeakyReLU()
        self.drop_1 = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            *[
                Conv_Forward(out_channels, out_channels, dropout=dropout)
                for i in range(num_conv)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 30),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(30, num_classes)
        )

    def forward(self, x):
        out_1 = self.drop_1(self.activation(self.conv_1(x)))
        out_2 = self.fc(out_1)
        out_3 = torch.mean(out_2, dim=(2, 3))
        out_4 = self.classifier(out_3)
        return out_4
    

class Conv_Forward(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Conv_Forward, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc(x)
    
