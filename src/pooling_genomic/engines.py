import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pooling_genomic.models import get_diffpool_aux_losses


def train_epoch_clf(
    model,
    train_loader,
    optimizer,
    loss_fn=None,
    device: str = "cpu",
    print_period: int = 5,
    scheduler=None,
    epoch=None,
    lambda_l1_node_importances: float = None,
    lambda_link_pred: float = 0.0,
    lambda_entropy: float = 0.0,
):
    model.train()

    if loss_fn is None:
        loss_fn = CrossEntropyLoss()
    loss_fn.to(device=device)

    iters = len(train_loader)
    correct, total, train_loss, train_steps = 0, 0, 0, 0
    for i, batch in enumerate(train_loader, 0):
        x, t = batch
        x, t = x.to(device), t.to(device)

        optimizer.zero_grad()
        y_predicted = model(x)
        loss = loss_fn(y_predicted, t)
        if lambda_l1_node_importances is not None:
            for ni in model[0].node_importances:
                loss += lambda_l1_node_importances * torch.sum(torch.abs(ni)) / ni.size()[0]

        aux_loss = get_diffpool_aux_losses(model, lambda_link_pred, lambda_entropy)
        loss = loss + aux_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            class_predicted = torch.argmax(y_predicted, dim=1)
            correct += (class_predicted == t).type(torch.float).sum().item()
            total += t.shape[0]

            train_loss += loss.detach().cpu().numpy()
            train_steps += 1

        if scheduler is not None:
            scheduler.step(epoch + i / iters)

        # print statistics
        if i % print_period == 0:
            with torch.no_grad():
                print("Batch: {}".format(i))
                print("Mean batch loss: {}".format(loss.item()))
                print("Batch Accuracy: {}".format(correct / total))
                if lambda_l1_node_importances is not None:
                    for ni in model[0].node_importances:
                        l1 = torch.sum(torch.abs(ni)) / ni.size()[0]
                        max_ni = torch.max(torch.abs(ni))
                        min_ni = torch.min(torch.abs(ni))
                        mean_ni = torch.mean(torch.abs(ni))
                        print(f"Num nodes: {ni.size()[0]}")
                        print(f"L1 nodes: {l1}")
                        print(f"Max: {max_ni} Min: {min_ni} Mean: {mean_ni}")
                    print()
            # break  # remove this break

    train_loss = train_loss / train_steps
    train_accuracy = correct / total
    metrics = {"loss": train_loss, "accuracy": train_accuracy}
    return model, metrics



def evaluate_clf(
    model,
    validation_loader,
    loss_fn=None,
    device: str = "cpu",
    return_outputs: bool = False,
):
    model.eval()

    if loss_fn is None:
        loss_fn = CrossEntropyLoss()
    loss_fn.to(device=device)

    class_predictions = []
    outputs = []
    labels = []
    iters = len(validation_loader)
    with torch.no_grad():
        total_loss = 0
        for i, batch in enumerate(validation_loader, 0):
            x, t = batch
            x, t = x.to(device), t.to(device)

            y_predicted = model(x)
            loss = loss_fn(y_predicted, t)

            outputs.append(y_predicted.detach().cpu().numpy())
            class_predicted = torch.argmax(y_predicted, dim=1)
            class_predictions.append(class_predicted.detach().cpu().numpy())
            labels.append(t.detach().cpu().numpy())
            total_loss += loss.detach().cpu().numpy()
            # break  # remove this break

    outputs = np.concatenate(outputs, axis=0)
    class_predictions = np.concatenate(class_predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    metrics = {
        "loss": total_loss / iters,
        "balanced_accuracy": balanced_accuracy_score(labels, class_predictions),
        "accuracy": accuracy_score(labels, class_predictions)
    }

    if return_outputs:
        return metrics, (outputs, labels)

    return metrics


def train_cohort_tumor_clf(
    model,
    train_loader,
    optimizer,
    loss_fn=None,
    device: str = "cpu",
    print_period: int = 5,
    scheduler=None,
    epoch=None,
    lambda_l1_node_importances: float = None,
    lambda_link_pred: float = 0.0,
    lambda_entropy: float = 0.0,
):
    model.train()

    if loss_fn is None:
        loss_fn = CrossEntropyLoss()
    loss_fn.to(device=device)

    iters = len(train_loader)
    correct, total, train_loss, train_steps = 0, 0, 0, 0
    correct_type = 0
    for i, batch in enumerate(train_loader, 0):
        x, t = batch
        x, t = x.to(device), [t[0].to(device), t[1].to(device)]

        optimizer.zero_grad()
        y_predicted = model(x)
        loss = loss_fn(y_predicted, t)
        if lambda_l1_node_importances is not None:
            for ni in model[0].node_importances:
                loss += lambda_l1_node_importances * torch.sum(torch.abs(ni)) / ni.size()[0]

        aux_loss = get_diffpool_aux_losses(model, lambda_link_pred, lambda_entropy)
        loss = loss + aux_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            class_predicted = torch.argmax(y_predicted[0], dim=1)
            correct += (class_predicted == t[0]).type(torch.float).sum().item()
            type_predicted = (y_predicted[1] > 0).to(torch.long)
            correct_type += (type_predicted == t[1]).type(torch.float).sum().item()
            total += t[0].shape[0]

            train_loss += loss.detach().cpu().numpy()
            train_steps += 1

        if scheduler is not None:
            scheduler.step(epoch + i / iters)

        # print statistics
        if i % print_period == 0:
            with torch.no_grad():
                print("Batch: {}".format(i))
                print("Mean batch loss: {}".format(loss.item()))
                print("Cohort Batch Accuracy: {}".format(correct / total))
                print("Type Batch Accuracy: {}".format(correct_type / total))
                if lambda_l1_node_importances is not None:
                    for ni in model[0].node_importances:
                        l1 = torch.sum(torch.abs(ni)) / ni.size()[0]
                        max_ni = torch.max(torch.abs(ni))
                        min_ni = torch.min(torch.abs(ni))
                        mean_ni = torch.mean(torch.abs(ni))
                        print(f"Num nodes: {ni.size()[0]}")
                        print(f"L1 nodes: {l1}")
                        print(f"Max: {max_ni} Min: {min_ni} Mean: {mean_ni}")
                    print()
        # break

    train_loss = train_loss / train_steps
    train_accuracy = correct / total
    train_accuracy_type = correct_type / total
    metrics = {"loss": train_loss, "accuracy": train_accuracy, "accuracy_type": train_accuracy_type}
    return model, metrics


def evaluate_cohort_tumor_clf(
    model,
    validation_loader,
    loss_fn=None,
    device: str = "cpu",
    return_outputs: bool = False,
):
    model.eval()

    if loss_fn is None:
        loss_fn = CrossEntropyLoss()
    loss_fn.to(device=device)

    cohort_predictions = []
    cohort_outputs = []
    cohort_labels = []

    type_predictions = []
    type_outputs = []
    type_labels = []

    iters = len(validation_loader)
    with torch.no_grad():
        total_loss = 0
        for i, batch in enumerate(validation_loader, 0):
            x, t = batch
            x, t = x.to(device), [t[0].to(device), t[1].to(device)]

            y_predicted = model(x)
            loss = loss_fn(y_predicted, t)

            cohort_outputs.append(y_predicted[0].detach().cpu().numpy())
            class_predicted = torch.argmax(y_predicted[0], dim=1)
            cohort_predictions.append(class_predicted.detach().cpu().numpy())
            cohort_labels.append(t[0].detach().cpu().numpy())

            type_outputs.append(y_predicted[1].detach().cpu().numpy())
            type_predicted = (y_predicted[1] > 0).detach().cpu().to(torch.long).numpy()
            type_predictions.append(type_predicted)
            type_labels.append(t[1].detach().cpu().numpy())

            total_loss += loss.detach().cpu().numpy()
            # break  # remove this break

    cohort_outputs = np.concatenate(cohort_outputs, axis=0)
    cohort_predictions = np.concatenate(cohort_predictions, axis=0)
    cohort_labels = np.concatenate(cohort_labels, axis=0)

    type_outputs = np.concatenate(type_outputs, axis=0)
    type_predictions = np.concatenate(type_predictions, axis=0)
    type_labels = np.concatenate(type_labels, axis=0)

    metrics = {
        "loss": total_loss / iters,
        "balanced_accuracy": balanced_accuracy_score(cohort_labels, cohort_predictions),
        "accuracy": accuracy_score(cohort_labels, cohort_predictions),
        "accuracy_type": accuracy_score(type_labels, type_predictions)
    }

    outputs = {
        'cohort_predictions': cohort_predictions,
        'cohort_outputs': cohort_outputs,
        'cohort_labels': cohort_labels,
        'type_predictions': type_predictions,
        'type_outputs': type_outputs,
        'type_labels': type_labels
    }

    if return_outputs:
        return metrics, outputs

    return metrics