import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch.autograd import Variable


def relu_hook_function(module, grad_in, grad_out):
    if isinstance(module, torch.nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.),)
    

def guided_backprop_saliency_dl(dataloader, model, device='cpu', max_batches=None):
    """Compute the guided backpropagation saliency values.
    The reLUs in the model must be set as Modules.
    """
    model.eval()

    for i, module in enumerate(model.modules()):
        if isinstance(module, torch.nn.ReLU):
            print(f"Registering relu hook to: {model.named_modules()}")
            module.register_backward_hook(relu_hook_function)

    images_grads = 0
    for i, batch in enumerate(dataloader, 0):
        if max_batches is not None:
            if i == max_batches:
                break
        
        x, t = batch
        x, t = x.to(device), t.to(device)

        X_var = Variable(x, requires_grad=True)
        y_var = Variable(t)

        scores = model(X_var)
        print(t)
        print(scores.argmax(dim=1))
        labels_scores = scores.gather(1, y_var.view(-1, 1)).squeeze()
        loss = -torch.sum(torch.log(labels_scores))
        loss.backward()
        
        images_grads += X_var.grad.data.abs()
        
    return images_grads


def cohort_and_tumor_saliency_analysis(
    dataloader, model, device='cpu', max_batches=None, which_y: int = 0
):
    """Compute the guided backpropagation saliency values for the cohort and tumor model.
    The reLUs in the model must be set as Modules.
    """
    model.eval()

    for i, module in enumerate(model.modules()):
        if isinstance(module, torch.nn.ReLU):
            print(f"Registering relu hook to: {model.named_modules()}")
            module.register_backward_hook(relu_hook_function)

    images_grads = 0
    for i, batch in enumerate(dataloader, 0):
        if max_batches is not None:
            if i == max_batches:
                break
        
        x, t = batch
        x, t = x.to(device), [t[0].to(device), t[1].to(device)]

        X_var = Variable(x, requires_grad=True)
        y_var = Variable(t[which_y])
        # print(y_var)
        scores = model(X_var)
        # print(scores)
        scores = scores[which_y]
        # print(scores)

        if which_y is None:
            pass
        elif which_y == 0:
            labels_scores = scores.gather(1, y_var.view(-1, 1)).squeeze()
            # print(labels_scores)
            loss = -torch.sum(torch.log(labels_scores))  # shouldn't I be adding a sigmoid here?
            loss.backward()
        elif which_y == 1:
            # scores = scores[y_var == 0]  # get the ones marked as tumor
            # print(scores)
            # loss = torch.log(1 - torch.sigmoid(scores[y_var == 0])) + torch.log(torch.sigmoid(scores[y_var == 1]))
            loss = y_var * torch.log(torch.sigmoid(scores)) + (1 - y_var) * torch.log(1 - torch.sigmoid(scores))
            # print(loss)
            loss = -torch.mean(loss)
            # print(loss)
            loss.backward()
            

        images_grads += X_var.grad.data.abs()
        
    return images_grads


def compare_with_random_features(dataset, genes_rank, n_examples=None):
    print(f"Shape genes_rank: {genes_rank.shape}")
    max_nf, step_nf = 100, 1
    Xs, ys = [], []
    if n_examples is None:
        n_examples = len(dataset)
    for i in range(min(n_examples, len(dataset))):
        Xs.append(dataset[i][0].numpy())
        ys.append(dataset[i][1])

    scores = []
    X_ds, y_ds = np.array(Xs), np.array(ys)
    print("Using rank:")
    for n_f in range(1, max_nf, step_nf):
        X_train, X_test, y_train, y_test = train_test_split(X_ds[:, genes_rank[:n_f]], y_ds, test_size=0.2, random_state=123)
        # model = LogisticRegression(penalty='l2', max_iter=1000)
        model = KNeighborsClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        # print(f"No Features: {n_f} Balanced Accuracy: {balanced_accuracy} Accuracy: {accuracy}")
        scores.append({
            'n_features': n_f, 'balanced_accuracy': balanced_accuracy, 'accuracy': accuracy
        })
    scores_rank = pd.DataFrame.from_records(scores)
    scores_rank.to_csv('knn_ranked_features.csv')

    scores = []
    rng = np.random.default_rng(seed=123)
    genes_rank_random = rng.permutation(genes_rank)
    print("Permuted:")
    for n_f in range(1, max_nf, step_nf):
        X_train, X_test, y_train, y_test = train_test_split(X_ds[:, genes_rank_random[:n_f]], y_ds, test_size=0.2, random_state=123)
        # model = LogisticRegression(penalty='l2', max_iter=1000)
        model = KNeighborsClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        # print(f"No Features: {n_f} Balanced Accuracy: {balanced_accuracy} Accuracy: {accuracy}")
        scores.append({
            'n_features': n_f, 'balanced_accuracy': balanced_accuracy, 'accuracy': accuracy
        })
    scores_rank_random = pd.DataFrame.from_records(scores)
    scores_rank_random.to_csv('knn_ranked_random_features.csv')        
