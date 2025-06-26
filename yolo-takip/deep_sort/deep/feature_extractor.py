import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class Extractor:
    def __init__(self, model_path, use_cuda=True):
        from deep_sort.deep.reid_model import Net

        self.net = Net(reid=True)
        self.device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')

        state = torch.load(model_path, map_location=self.device)
        # Hem 'net_dict' hem de doğrudan state_dict yüklenebilir
        if isinstance(state, dict) and 'net_dict' in state:
            self.net.load_state_dict(state['net_dict'])
        else:
            self.net.load_state_dict(state)

        self.net.to(self.device)
        self.net.eval()

        self.size = (64, 128)
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __call__(self, im, boxes):
        crops = []
        for box in boxes:
            x, y, w, h = box
            x1 = int(max(x - w / 2, 0))
            y1 = int(max(y - h / 2, 0))
            x2 = int(min(x + w / 2, im.shape[1]))
            y2 = int(min(y + h / 2, im.shape[0]))
            crop = im[y1:y2, x1:x2]
            crop = Image.fromarray(crop).convert('RGB')
            crops.append(self.transform(crop).unsqueeze(0))

        if not crops:
            return np.array([])

        imgs = torch.cat(crops).to(self.device)
        with torch.no_grad():
            feats = self.net(imgs).cpu().numpy()
        return feats
