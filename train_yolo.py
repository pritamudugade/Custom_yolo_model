
import yolo_model


# train_yolo.py

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from yolo_model import YOLOv3
from dataset import CustomDataset
from yolo_loss import YOLOv3Loss


import os

def create_directory_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Specify the path to the 'labels' directory
labels_dir = '/path/to/IntruderDetectionYOLOv8.v1i.yolov8/train/labels'

try:
    # Attempt to create the 'labels' directory if it does not exist
    create_directory_if_not_exists(labels_dir)
except PermissionError as e:
    print(f"PermissionError: {e}")
    print("Failed to create the 'labels' directory. Make sure you have the necessary permissions.")
    exit(1)

def main():
    # Configuration
    num_classes = 4  # Replace with the number of classes in your dataset
    batch_size = 64
    img_size = 416
    epochs = 100
    learning_rate = 0.001
    step_size = 30
    gamma = 0.1


    

    # Dataset and DataLoader
    train_dataset = CustomDataset(annotation_file=r"IntruderDetectionYOLOv8.v1i.yolov8\train\labels", img_dir=r"IntruderDetectionYOLOv8.v1i.yolov8\train\images", num_classes=num_classes, img_size=img_size, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # YOLOv3 Model
    model = YOLOv3(num_classes)
    model.train()

    # Loss Function
    criterion = YOLOv3Loss(num_classes)

    # Optimizer and Learning Rate Scheduler
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training Loop
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Update Learning Rate Scheduler
        scheduler.step()

        # Print Training Progress
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {average_loss:.4f}")

    # Save the Trained Model


    import os

    # Specify the absolute path where you want to save the file
    #save_path = os.path.join(r"c:\Users\DELL\Downloads", 'yolov3_model.pth')
    #torch.save(model.state_dict(), save_path)
    

    # ... (other code) ...

    # Save the Trained Model
    save_path = os.path.join(r"c:\Users\DELL\Downloads", 'yolov3_model.pth')
    torch.save(model.state_dict(), save_path)


    # torch.save(r"C:\Windows\System32\Custom_model", 'yolov3_model.pth')