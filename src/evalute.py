import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.model import SimpleAnimalCNN
from src.dataset import AnimalNumpyDataset
from torch.utils.data import DataLoader
from torchvision import transforms


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5
CLASS_NAMES = ['Cat', 'Dog', 'Bird', 'Fish', 'Turtle'] 


model = SimpleAnimalCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
test_dataset = AnimalNumpyDataset('data/processed/test_img.npy', 'data/processed/test_lbl.npy', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())


print("\nBÁO CÁO KẾT QUẢ TEST:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))