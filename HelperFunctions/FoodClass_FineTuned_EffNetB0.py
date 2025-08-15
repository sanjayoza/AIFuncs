import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import pytorch_lightning as pl


class FineTuned(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3, layers_to_unfreeze=None, dropout_rate=0.3):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained EfficientNet
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Freeze all backbone layers initially
        for param in base_model.features.parameters():
            param.requires_grad = False

        # Feature extractor
        self.backbone = base_model.features

        # Add global pooling explicitly
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Extra dropout before classifier
        self.pre_classifier_dropout = nn.Dropout(p=dropout_rate)

        # Replace classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, C, 1, 1) → (B, C)
            nn.Dropout(p=dropout_rate),
            nn.Linear(base_model.classifier[1].in_features, num_classes)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.train_acc_epoch = []
        self.train_loss_epoch = []
        self.val_acc_epoch = []
        self.val_loss_epoch = []
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        # Fine-tuning control
        self.layers_to_unfreeze = layers_to_unfreeze or []
        self._unfreeze_selected_layers()

    def _unfreeze_selected_layers(self):
        print("Unfreezing selected layers:")
        for name, module in self.backbone.named_children():
            if name in self.layers_to_unfreeze:
                print(f"  - Layer {name} is now trainable")
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)               # EfficientNet features
        x = self.global_pool(x)            # Global average pooling
        x = self.pre_classifier_dropout(x) # Extra dropout
        x = self.classifier(x)             # Classifier
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return {"loss": loss, "acc": acc}

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics["train_loss"].item()
        train_acc = self.trainer.callback_metrics["train_acc"].item()
        self.train_loss_epoch.append(train_loss)
        self.train_acc_epoch.append(train_acc)
        print(f"Epoch {self.current_epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        val_acc = self.trainer.callback_metrics["val_acc"].item()
        self.val_loss_epoch.append(val_loss)
        self.val_acc_epoch.append(val_acc)
        print(f"Epoch {self.current_epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def getTrainingHistory(self):
        return {
            'train_loss': self.train_loss_epoch,
            'val_loss': self.val_loss_epoch,
            'train_acc': self.train_acc_epoch,
            'val_acc': self.val_acc_epoch
        }