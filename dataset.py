
import os
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ToTensor


class CustomDataset(data.Dataset):
    def __init__(self, image_folder, label_folder, num_classes, img_size=416, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.num_classes = num_classes
        self.img_size = img_size
        self.transform = transform

        

        # Get a list of image and label file names
        image_files = os.listdir(self.image_folder)
        label_files = os.listdir(self.label_folder)

        # Filter out files with unsupported image file extensions
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in supported_extensions]

        # Match image files with their corresponding label files
        self.samples = []
        for image_file in image_files:
            image_path = os.path.join(self.image_folder, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(self.label_folder, label_file)
            if os.path.exists(label_path):
                self.samples.append((image_path, label_path))

    def _read_annotations(self, label_file):
        samples = []
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split(' ')
            x_center, y_center, width, height, class_id = map(float, line)
            targets.append([x_center, y_center, width, height, class_id])

        return targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Load and preprocess the image and targets for the given index
        img_path, label_path = self.samples[index]

        # Load the image using PIL
        img = Image.open(img_path)

        # Read the annotation file to get the targets
        targets = self._read_annotations(label_path)

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        # Convert the targets to a tensor
        targets = torch.tensor(targets)

        return img, targets


image_folder =  r"C:\Windows\System32\Custom_model\IntruderDetectionYOLOv8.v1i.yolov8\train\images"
label_folder = r"C:\Windows\System32\Custom_model\IntruderDetectionYOLOv8.v1i.yolov8\train\labels"

num_classes = 4

transform = ToTensor()

# Create an instance of the CustomDataset
dataset = CustomDataset(image_folder=image_folder, label_folder=label_folder, num_classes=num_classes, transform=transform)