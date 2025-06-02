import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import itertools
from PIL import Image
import torchvision.transforms as T

from dataset import AGDataset
from model import AGClassifier

def unnormalize(tensor, mean, std):
    """
    Undo Normalize so that tensor is back in [0,1] for display.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0.0, 1.0)

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class "AG"
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, y_prob)[:2])

    return y_true, y_pred, y_prob, acc, prec, rec, f1, roc_auc

def show_combined_results(model, dataset, device, classes, num_examples):
    """
    Create a single figure with:
      - Text box of metrics
      - ROC curve and Confusion Matrix side by side
      - Example images (raw vs. processed + predicted) in subsequent rows
    """
    # 1) Prepare test loader
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    # 2) Evaluate to get metrics
    y_true, y_pred, y_prob, acc, prec, rec, f1, roc_auc = evaluate_model(model, test_loader, device)

    # 3) Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # 4) Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # 5) Prepare figure with GridSpec
    total_rows = 2 + num_examples  # row0: text, row1: ROC+CM, rows2..: examples
    fig = plt.figure(constrained_layout=True, figsize=(10, 2.5 * total_rows))
    gs = fig.add_gridspec(total_rows, 3)

    # Row 0: metrics text across both columns
    ax_text = fig.add_subplot(gs[0, 0])
    metrics_text = (
        f"Test Accuracy:  {acc*100:.2f}%\n"
        f"Test Precision: {prec*100:.2f}%\n"
        f"Test Recall:    {rec*100:.2f}%\n"
        f"Test F1 Score:  {f1*100:.2f}%\n"
        f"Test AUC:       {roc_auc*100:.2f}%"
    )
    ax_text.text(0.01, 0.5, metrics_text, fontsize=12, va='center')
    ax_text.axis('off')

    # Row 1, col 0: ROC curve
    ax_roc = fig.add_subplot(gs[0, 1])
    ax_roc.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    ax_roc.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')

    # Row 1, col 1: Confusion matrix
    ax_cm = fig.add_subplot(gs[0, 2])
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.set_title('Confusion Matrix')
    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_xticklabels(classes)
    ax_cm.set_yticklabels(classes)
    # Annotate cells
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax_cm.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    ax_cm.set_ylabel('True label')
    ax_cm.set_xlabel('Predicted label')

    # 6) Example classifications
    transform = dataset.transform
    mean = transform.transforms[-1].mean
    std = transform.transforms[-1].std

    indices = np.random.choice(len(dataset), num_examples, replace=False)
    for row_idx, idx in enumerate(indices):
        img_path, true_label = dataset.samples[idx]
        # Raw image
        orig = Image.open(img_path).convert('RGB')

        # Preprocessed + normalized tensor
        img_tensor, _ = dataset[idx]
        img_tensor = img_tensor.to(device).unsqueeze(0)
        output = model(img_tensor)
        pred_label = output.argmax(dim=1).item()

        # Unnormalize for display
        img_vis = unnormalize(img_tensor.cpu().squeeze().clone(), mean, std)
        img_np = img_vis.permute(1, 2, 0).numpy()

        ax_raw = fig.add_subplot(gs[1 + row_idx, 0])
        ax_proc = fig.add_subplot(gs[1 + row_idx, 1])

        ax_raw.imshow(orig)
        ax_raw.set_title(f"True: {classes[true_label]}")
        ax_raw.axis('off')

        ax_proc.imshow(img_np)
        ax_proc.set_title(f"Pred: {classes[pred_label]}")
        ax_proc.axis('off')

    plt.show()

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "data/processed"
    ckpt_list = glob.glob(os.path.join("checkpoints", "*.pth"))
    checkpoint_path = max(ckpt_list, key=os.path.getmtime)
    classes = ['Normal', 'Atrophic Gastritis']

    # Load test dataset
    test_dataset = AGDataset(root_dir=os.path.join(data_root, "test"))

    # Load model
    model = AGClassifier(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Show combined results
    show_combined_results(model, test_dataset, device, classes, num_examples=4)

if __name__ == "__main__":
    main()
