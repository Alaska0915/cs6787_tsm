import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
import numpy as np 

def train(student, teacher, dataloader, val_dataloader, optimizer, loss_calculator, epochs, device):
    best_accuracy = 0
    best_epoch = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    # start_time = time.time() 
    # predict(teacher, dataloader, val_dataloader, device)
    # inference_duration = time.time() - start_time

    teacher.eval()
    t_duration = 0 # only training
    total_duration = 0 # entire loop minus logging

    for epoch in tqdm(range(1, epochs + 1)):

        start_time = time.time()
        # train one epoch
        loss_kd = train_step(student, dataloader,optimizer, loss_calculator, device)

        t_duration += time.time() - start_time

        # validate the network
        # accuracy = measure_accuracy(student, val_dataloader, device)
        loss, accuracy = evaluate_model(val_dataloader, student, loss_fn, device)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

        # learning rate schenduling
        # scheduler.step()

        total_duration += (time.time() - start_time)

        if (epoch < 5) or (epoch % 5 == 0) or (epoch == epochs):
            print("Epoch %d took %f s and has accuracy %f"%(epoch, total_duration, accuracy))

        # save check point
        if (epoch % 10 == 0) or (epoch == epochs):
            torch.save({'epoch': epoch,
                        'training time': t_duration, 
                        'total time': total_duration, 
                        'kd loss' : loss_kd,
                        'loss': loss,
                        'accuracy': accuracy,
                        'state_dict': student.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        # 'scheduler': scheduler.state_dict()},
                        # 'loss_log': loss_calculator.loss_log},
                        os.path.join('model', 'cifar10_pretrained', 'epoch_%d.pth'%epoch))

    print("Finished Training in %f, Best Accuracy: %f (at %d epochs)"%(total_duration, best_accuracy, best_epoch))
    return student

def train_step(student, dataloader, optimizer, loss_calculator, device):
    student.train()
    # train_file_path = 'pretrained_predictions_train_{}.txt'
    teacher = torch.load(os.path.join('inference', 'train.pt'))
    
    for i, info in enumerate(dataloader):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = student(info[0].to(device))

        teacher_outputs = teacher[i]
        # with torch.no_grad():
        #     teacher_outputs = teacher(info['pixel_values'].to(device)).logits

        loss = loss_calculator.forward(outputs = outputs,
                            labels = info[1].to(device),
                            teacher_outputs = teacher_outputs)
        loss.backward()
        optimizer.step()

    return loss.item()

# def measure_accuracy(model, dataloader, device):
#     model.eval()

#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for i, info in dataloader:
#             outputs = model(info['img'].to(device))
#             _, predicted = torch.max(outputs.data, 1)

#             total += info['label'].size(0)
#             correct += (predicted == info['label'].to(device)).sum().cpu().item()

#     print("Accuracy of the network on the 10000 test images: %f %%"%(100 * correct / total))

#     return correct / total

def evaluate_model(dataloader, model, loss_fn, device):
    loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for i, info in enumerate(dataloader):
            outputs = model(info[0].to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += info[1].size(0)
            correct += (predicted == info[1]).sum().item()
            
            loss += loss_fn(outputs, info[1]).item()

    loss = loss / len(dataloader.dataset)
    accuracy = correct/total
    return (loss, accuracy)
