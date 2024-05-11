import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
    

def train(student, teacher, dataloader, val_dataloader, optimizer, scheduler, loss_calculator, epochs, device):
    best_accuracy = 0
    best_epoch = 0

    teacher.eval()

    start_time = time.time()

    for epoch in tqdm(range(1, epochs + 1)):
        # train one epoch
        train_step(student, teacher, dataloader, optimizer, loss_calculator, epoch, device)

        duration = time.time() - start_time

        # validate the network
        accuracy = measure_accuracy(student, val_dataloader, device)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

        # learning rate schenduling
        scheduler.step()

        # save check point
        if (epoch % 10 == 0) or (epoch == epochs):
            print("Epoch %d took %f s and has accuracy %f"%(epoch, duration, accuracy))
            torch.save({'epoch': epoch,
                        'time': duration, 
                        'state_dict': student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss_log': loss_calculator.loss_log},
                        os.path.join('model', 'cifar10_pretrained_distill', 'epoch_%d.pth'%epoch))

    print("Finished Training in %f, Best Accuracy: %f (at %d epochs)"%(duration, best_accuracy, best_epoch))
    return student

def train_step(student, teacher, dataloader, optimizer, loss_calculator, epoch, device):
    student.train()

    for i, info in enumerate(dataloader):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = student(info['img'].to(device))

        with torch.no_grad():
            teacher_outputs = teacher(info['pixel_values'].to(device)).logits

        loss = loss_calculator(outputs          = outputs,
                               labels           = info['label'].to(device),
                               teacher_outputs  = teacher_outputs)
        loss.backward()
        optimizer.step()
    return

def measure_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i, info in dataloader:
            outputs = model(info['img'].to(device))
            _, predicted = torch.max(outputs.data, 1)

            total += info['label'].size(0)
            correct += (predicted == info['label'].to(device)).sum().cpu().item()

    print("Accuracy of the network on the 10000 test images: %f %%"%(100 * correct / total))

    return correct / total