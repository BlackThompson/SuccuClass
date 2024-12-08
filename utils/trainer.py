import torch
import logging


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=100,
    checkpoint_path="best_model.pth",
):
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                log_msg = (
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                    f"Train Loss: {running_loss/10:.4f}, "
                    f"Train Accuracy: {100 * correct/total:.2f}%"
                )
                logging.info(log_msg)
                running_loss = 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        log_msg = (
            f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )
        logging.info(log_msg)

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved with validation accuracy: {val_accuracy:.2f}%")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
