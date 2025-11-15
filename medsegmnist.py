import os
import gdown

_all_ = ["MedSegMNIST", "medsegmnist"]

class MedSegMNIST:
    def __init__(self):
        self.root = None
        self.split = None
        self.download = False
        self.transform = None

    def BrainTumor(self):
        self.root = os.path.join("data", "MedSegMNIST", "BrainTumor")
        self.split = "train"
        self.download = True
        self.transform = None

        if self.download:
            url = "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j"
            output = os.path.join(self.root, "brain_tumor_data.zip")
            os.makedirs(self.root, exist_ok=True)
            gdown.download(url, output, quiet=False)

        print(f"Dataset prepared at {self.root} with split '{self.split}'")