import json
import matplotlib.pyplot as plt

with open("checkpoints/training_metrics.json", "r") as f:
    metrics = json.load(f)

epochs = range(1, len(metrics["train_loss"]) + 1)

plt.figure()
plt.plot(epochs, metrics["train_loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.savefig("train_loss.png", dpi=200)
plt.show()

plt.figure()
plt.plot(epochs, metrics["val_accuracy"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy")
plt.tight_layout()
plt.savefig("val_accuracy.png", dpi=200)
plt.show()