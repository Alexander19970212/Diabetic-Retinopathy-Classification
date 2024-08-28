import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # is need to train on 'hachiko'

from sklearn.metrics import confusion_matrix
import torch

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score #, kappa
from sklearn.metrics import roc_auc_score

from transformers import TrainingArguments
from transformers import Trainer

import matplotlib.pyplot as plt
import numpy as np
import evaluate

from tqdm import tqdm
import json

# Load accuracy evaluator
accuracy = evaluate.load("accuracy")

# Define function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, file_name,
                          num_classes = 5,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    classes = range(num_classes)

    # Set the title if not provided
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    np.set_printoptions(precision=2)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    # Normalize confusion matrix if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    # Create the plot
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
                              
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Save the plot as an image
    plt.savefig(file_name)
    plt.show()

# Define function to calculate per class accuracy
def calculate_per_class_accuracy(confusion_matrix):
    """
    Calculate per-class accuracy from a confusion matrix.

    Args:
    - confusion_matrix (numpy.ndarray): The confusion matrix.

    Returns:
    - per_class_accuracy (list): List of per-class accuracy values.
    """
    num_classes = confusion_matrix.shape[0]
    per_class_accuracy = []

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, :]) - TP
        FP = np.sum(confusion_matrix[:, i]) - TP
        TN = np.sum(confusion_matrix) - (TP + FP + FN)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        per_class_accuracy.append(accuracy)

    return per_class_accuracy

# Define function to compute metrics
def get_compute_metrics(num_classes=5, save_cm=False, cm_path=None):

    """
    Get the function to compute evaluation metrics.

    Args:
    - pretrained_model_name (str): Name of the pretrained model.
    - dataset_name (str): Name of the dataset.
    - save_cm (bool): Whether to save the confusion matrix plot.

    Returns:
    - compute_metrics (function): Function to compute evaluation metrics.
    """
    
    def compute_metrics(eval_pred):
        predictions_proba, labels = eval_pred
        predictions = np.argmax(predictions_proba, axis=1)
        predictions = np.clip(predictions, 0, num_classes)
        result_accuracy = accuracy.compute(predictions=predictions, references=labels)

        if predictions_proba.shape[1] > 1:  # Check if we have more than one class
            predictions_proba = torch.nn.functional.softmax(torch.tensor(predictions_proba), dim=-1).numpy()

        cm = confusion_matrix(labels, predictions)
        perclass_acc = calculate_per_class_accuracy(cm)

        result = {
                'accuracy': np.mean([result_accuracy['accuracy']]),
                'kappa': np.mean([cohen_kappa_score(labels, predictions, weights = "quadratic")]),
                'f1': np.mean([f1_score(labels, predictions, average='weighted')]),
                # 'roc_auc': np.mean([roc_auc_score(labels, predictions_proba, multi_class='ovr')]),
                'roc_auc': np.mean([roc_auc_score(labels, predictions_proba, multi_class='ovr')]),
                'class_0' : perclass_acc[0],
                'class_1' : perclass_acc[1],
                'class_2' : perclass_acc[2],
                'class_3' : perclass_acc[3],
                'class_4' : perclass_acc[4],
                }
        
        if save_cm:
           plot_confusion_matrix(labels, predictions, file_name=cm_path, num_classes=num_classes, normalize=True,
                           title='Normalized confusion matrix')

        return result

    return compute_metrics

# Define function to define collate function
def get_collate_fn(with_embedings=False):
    def collate_fn(batch):
        collated_batch = {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }
        if with_embedings:
            collated_batch['embedings'] = torch.stack([x['embedings'] for x in batch])
        
        return collated_batch
    
    return collate_fn

def build_trainer(model, train_dataset, valid_dataset, args, train_mode=True):
    # arguments for training
    training_args = TrainingArguments(
        output_dir=args.output_dir, #"./SSiT-base",
        evaluation_strategy=args.evaluation_strategy, #"steps",
        logging_steps=args.logging_steps, #50,

        save_steps=args.save_steps, #50,
        eval_steps=args.eval_steps, #50,
        save_total_limit=args.save_total_limit, #3,
        
        report_to=args.report_to, #"wandb",  # enable logging to W&B
        run_name=args.run_name, #r_name,  # name of the W&B run (optional)
        
        remove_unused_columns=False,
        dataloader_num_workers = args.dataloader_num_workers, #16,
        lr_scheduler_type = args.lr_scheduler_type, #'constant_with_warmup', # 'constant', 'cosine'
        
        learning_rate=args.learning_rate, #2e-5,
        # label_smoothing_factor = 0.6,
        per_device_train_batch_size=args.batch_size, #64,
        gradient_accumulation_steps=args.gradient_accumulation_steps, #4,
        per_device_eval_batch_size=args.batch_size, #64,
        num_train_epochs=args.num_train_epochs, #15,
        warmup_ratio=args.warmup_ratio, #0.02,
        
        metric_for_best_model=args.metric_for_best_model, #"kappa", # select the best model via metric kappa
        greater_is_better = True,
        load_best_model_at_end=True,
        
        push_to_hub=False
    )

    collate_fn = get_collate_fn(with_embedings=False)
    if train_mode:
        compute_metrics_f = get_compute_metrics(args.num_classes)
    else:
        cm_path = f'{args.plots_path}/{args.run_name}.png'
        compute_metrics_f = get_compute_metrics(args.num_classes, save_cm=True, cm_path=cm_path)

    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_f,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

    return trainer

def train(model, train_dataset, valid_dataset, test_dataset, args):
    
    trainer = build_trainer(model, train_dataset, valid_dataset, args, train_mode=True)
    train_results = trainer.train()

    if args.save_hgf_model:
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        model.save_pretrained(f"{args.saved_model_dir}/{args.run_name}", from_pt=True)

    if args.test_after_train:
        metrics = trainer.evaluate(test_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    return model

def test(model, test_dataset, train_dataset, valid_dataset, args, device):
    """
    trainer = build_trainer(model, train_dataset, valid_dataset, args, train_mode=False)
    metrics = trainer.evaluate(test_dataset)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    """
    
    cm_path = f'{args.plots_path}/{args.run_name}.png'
    compute_metrics_f = get_compute_metrics(args.num_classes, save_cm=True, cm_path=cm_path)

    n_chunks = len(test_dataset)//args.batch_size
    if len(test_dataset)%args.batch_size > 0:
        n_chunks += 1

    splited_dataframe = np.array_split(np.arange(len(test_dataset)), n_chunks)

    all_logits = []
    all_labels = []

    model.model.eval()
    with torch.no_grad():
        for batch_ids in tqdm(splited_dataframe):
            batch = test_dataset[batch_ids]
            pixel_values = torch.stack([x for x in batch['pixel_values']]).to(device)
            labels = torch.tensor([x for x in batch['label']])
    
            logits = model.forward(pixel_values=pixel_values)["logits"]
            all_logits.append(logits.to('cpu'))
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = compute_metrics_f((logits, labels))
    print(metrics)
    
    with open(f'{args.plots_path}/{args.run_name}.json', 'w') as fp:
        json.dump(metrics, fp)
