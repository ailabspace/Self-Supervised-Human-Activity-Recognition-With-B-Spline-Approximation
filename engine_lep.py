import math

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.tensorboard import SummaryWriter

from utils import *
from tqdm import trange
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def generate_cm(cm, labels, work_dir):
    print_log("Generating Confusion matrix..", work_dir)
    sns.set(font_scale=1.2)
    plt.figure(figsize=(24, 20))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.1, square=True, cbar_kws={"shrink": 0.75},
                     annot_kws={"size": 6})
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "cm.png"))
    print_log("Confusion matrix generated.", work_dir)


def evaluate(model, classifier, test_loader, amp = False, segments = None, mask_ratio = .0, best_acc = 0, work_dir = "./output_dir/", avg_pool = False):
    test_pred = []
    labels_test = []
    batch_accuracies = []
    with torch.no_grad():
        for batch, (data, label, index) in enumerate(test_loader):
            data = data.float().cuda()

            with torch.amp.autocast('cuda', enabled=amp):
                latent = model(data, segments, mask_ratio, avg_pool)

                latent = latent.view(latent.shape[0], -1)
                predicted = classifier(latent)
                batch_accuracies.append(compute_batch_acc(predicted, label.cuda()))

                predicted = torch.softmax(predicted, dim=1)
                _, predicted = torch.max(predicted, 1)

                test_pred.append(predicted.cpu())
                labels_test.append(label.cpu())

        test_pred = torch.cat(test_pred, dim=0)
        test_pred = test_pred.numpy()
        labels_test = torch.cat(labels_test, dim=0)
        labels_test = labels_test.numpy()

        # accuracy = accuracy_score(labels_test, test_pred)
        # accuracy = balanced_accuracy_score(labels_test, test_pred)
        accuracy = 0
        for batch_accuracy in batch_accuracies:
            accuracy += batch_accuracy[0]
        accuracy /= len(batch_accuracies)

        precision = precision_score(labels_test, test_pred, average='macro')
        recall = recall_score(labels_test, test_pred, average='macro')
        f1 = f1_score(labels_test, test_pred, average='macro')

        print_log(f"Accuracy: {accuracy:.1f}%", work_dir)
        print_log(f"Precision: {precision*100:.1f}%", work_dir)
        print_log(f"Recall: {recall*100:.1f}%", work_dir)
        print_log(f"F1 Score: {f1*100:.1f}%", work_dir)

        if best_acc < accuracy:
            cm = confusion_matrix(labels_test, test_pred)
        else:
            cm = None

    return cm, accuracy, precision, recall, f1

def train_lep(train_loader, test_loader, model, classifier, classifier_optimizer, classifier_criterion, scheduler_classifier, e, segments, mask_ratio, amp = False, work_dir = "./output_dir/", labels = None, avg_pool = False, per_class_acc = False):
    writer = SummaryWriter()
    best_acc = 0
    best_e = -1

    test_accuracies = []
    test_precision = []
    test_recall = []
    test_f1 = []

    con_mat = None

    start_time = time.time()
    for epoch in trange(e):

        l_cap = []
        l = []
        for batch, (data, label, index) in enumerate(train_loader):
            data = data.float().cuda()

            with torch.amp.autocast('cuda', enabled=amp):
                latent = model(data, segments, mask_ratio, avg_pool)

                latent = latent.view(latent.shape[0], -1)
                predicted = classifier(latent)

                ls = label.cuda()
                # ls = label.cuda().repeat_interleave(2)

                loss = classifier_criterion(predicted, ls)
                writer.add_scalar(f"LEP Training Loss", loss, epoch)

            predicted = torch.softmax(predicted, dim=1)
            _, predicted = torch.max(predicted, 1)
            l.extend(ls.cpu())
            l_cap.extend(predicted.detach().cpu())
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

        writer.flush()

        cm, acc, prec, rec, f1 = evaluate(model, classifier, test_loader, amp, segments, mask_ratio, best_acc, work_dir=work_dir, avg_pool=avg_pool)

        if acc>best_acc:
            best_acc = acc
            con_mat = cm
            best_e = epoch
            classifier_state_dict = classifier.state_dict()

        test_accuracies.append(acc)
        test_precision.append(prec)
        test_recall.append(rec)
        test_f1.append(f1)

        print_log(f"Epoch {epoch + 1} Loss: {loss}", work_dir)
        scheduler_classifier.step()

    save_weights(classifier_state_dict, work_dir, "best_lep")

    print_log(f"Best Accuracy at epoch {best_e}: {best_acc}", work_dir)

    print_log(f"Best Test Accuracy: {test_accuracies[best_e] :.1f}%", work_dir)
    print_log(f"Best Test Precision: {test_precision[best_e] * 100:.1f}%", work_dir)
    print_log(f"Best Test Recall: {test_recall[best_e] * 100:.1f}%", work_dir)
    print_log(f"Best Test F1 Score: {test_f1[best_e] * 100:.1f}%", work_dir)


    if per_class_acc:
        per_class_acc = con_mat.diagonal() / con_mat.sum(axis = 1)
        for i, acc in enumerate(per_class_acc):
            print_log(f"{labels[i]}: Accuracy = {acc:.2f}", work_dir)


    print_log(f"Time taken for LEP: {compute_duration(start_time, time.time())}s", work_dir)

    generate_cm(con_mat, labels, work_dir)

    writer.close()
    torch.cuda.empty_cache()

def val_lep(test_loader, model, classifier, segments, mask_ratio, weights_path, amp=False, work_dir="./output_dir/", labels = None, avg_pool = False, per_class_acc = False):
    classifier = load_weights(classifier, weights_path)

    with torch.no_grad():
        cm, acc, prec, rec, f1 = evaluate(model, classifier, test_loader, amp, segments, mask_ratio, work_dir=work_dir, avg_pool=avg_pool)

    print_log(f"Best Test Accuracy: {acc :.1f}%", work_dir)
    print_log(f"Best Test Precision: { prec * 100:.1f}%", work_dir)
    print_log(f"Best Test Recall: {rec * 100:.1f}%", work_dir)
    print_log(f"Best Test F1 Score: {f1 * 100:.1f}%", work_dir)

    if per_class_acc:
        per_class_acc = cm.diagonal() / cm.sum(axis = 1)
        for i, acc in enumerate(per_class_acc):
            print_log(f"{labels[i]}: Accuracy = {acc:.2f}", work_dir)

    generate_cm(cm, labels, work_dir)