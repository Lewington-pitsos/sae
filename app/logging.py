import wandb
import torch
import time

import random
import torch.nn as nn
from app.tok import load_tokenizer
from sklearn.metrics import f1_score

def log_model_parameters_info(model: nn.Module, skip_wandb: bool = False):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    ratio = trainable_params / non_trainable_params if non_trainable_params != 0 else float('inf')

    print(f"Total parameters: {total_params:,} (roughly {total_params:.2e})")
    print(f"Trainable parameters: {trainable_params:,} (roughly {trainable_params:.2e})")
    print(f"Non-trainable parameters: {non_trainable_params:,} (roughly {non_trainable_params:.2e})")
    print(f"Ratio (Trainable/Non-trainable): {ratio:,} (roughly {ratio:.2e})")

    if not skip_wandb:
        wandb.log({"total_parameters": total_params, "trainable_parameters": trainable_params, "non_trainable_parameters": non_trainable_params, "trainable_parameter_ratio": ratio})

class MetricsLogger():
    def __init__(self, model_type, skip_wandb=False, name=None):
        if not skip_wandb:
            self.batch1_table = wandb.Table(columns=["epoch", "idx", "input_text", "label", "prediction", "logits"])
        else:
            self.batch1_table = None

        self.tokenizer = load_tokenizer(model_type)
        self.train_losses = []
        self.train_acc = []
        self.train_f1 = []
        self.test_losses = []
        self.test_acc = []
        self.test_f1 = []
        self.epoch = 0
        self.skip_wandb = skip_wandb
        self.start = time.time()
        self.name = name

    def init(self, *args, **kwargs):
        if not self.skip_wandb:
            wandb.init(reinit=True, name=self.name, *args, **kwargs)

    def step_epoch(self):
        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        avg_train_accuracy = sum(self.train_acc) / len(self.train_acc)
        avg_train_f1 = sum(self.train_f1) / len(self.train_f1)
        self.train_losses = []
        self.train_acc = []
        self.train_f1 = []
        print(f"Epoch {self.epoch + 1}, Train Loss: {avg_train_loss}, Train Acc: {avg_train_accuracy}, Train F1 {avg_train_f1}")

        if len(self.test_losses) > 0:
            avg_test_loss = sum(self.test_losses) / len(self.test_losses)
            avg_test_accuracy = sum(self.test_acc) / len(self.test_acc)
            avg_test_f1 = sum(self.test_f1) / len(self.test_f1)
            self.test_losses = []
            self.test_acc = []
            self.test_f1 = []
            print(f"Epoch {self.epoch + 1}, Test Loss: {avg_test_loss}, Test Acc: {avg_test_accuracy}, Test F1: {avg_test_f1}")

            if not self.skip_wandb:
                wandb.log({"train_loss": avg_train_loss, "train_accuracy": avg_train_accuracy, "train_f1": avg_train_f1})
                wandb.log({"test_loss": avg_test_loss, "test_accuracy": avg_test_accuracy, "test_f1": avg_test_f1})

        self.epoch += 1

    def log_train_batch(self, loss, labels, outputs, lr):
        self.train_losses.append(loss)
        pred_classes = torch.argmax(outputs, dim=-1)
        acc = ((pred_classes == labels) * 1.0).mean().item()
        self.train_acc.append(acc)
        f1 = f1_score(labels.cpu().numpy(), pred_classes.cpu().numpy(), average='macro')
        self.train_f1.append(f1)
        if not self.skip_wandb:
            wandb.log({"train_batch_loss": loss, "batch_learning_rate": lr, "train_batch_accuracy": acc, "train_batch_f1": f1})

    def log_test_batch(self, batch_idx, loss, labels, outputs, input_ids):
        self.test_losses.append(loss)
        pred_classes = torch.argmax(outputs, dim=-1)
        self.test_acc.append(((pred_classes == labels) * 1.0).mean().item())
        self.test_f1.append(f1_score(labels.cpu().numpy(), pred_classes.cpu(), average='macro'))

        if batch_idx == 0 and not self.skip_wandb:
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids[:4]]

            predictions = torch.argmax(outputs, dim=-1)
            
            for i in range(min(len(input_texts), 4)):
                self.batch1_table.add_data(self.epoch, i, input_texts[i], labels[i], predictions[i], outputs[i])

    def finalize(self):
        elapsed_seconds = time.time() - self.start
        print(f"Training took {elapsed_seconds:.2f} seconds")
        if not self.skip_wandb:
            wandb.log({"elapsed_seconds": elapsed_seconds})
            wandb.log({f"batch_1": self.batch1_table})
            wandb.finish()

    def log_label_ratio(self, dataset, name):
        labels = []
        to_sample = min(300, len(dataset))
        indices = random.sample(range(len(dataset)), to_sample)
        for i in indices:
            labels.append(dataset.get_label(i))
        
        true_count = sum(1 for item in labels if item == 1)
        false_count = len(labels) - true_count
        print(f"{name} True labels: {true_count}, False labels: {false_count}")
        print(f"{name} Ratio (True/False): {true_count/false_count:.2f}")

        if not self.skip_wandb:
            wandb.log({f"{name}_data_true_count": true_count, f"{name}_data_false_count": false_count, f"{name}_data_TF_ratio": true_count/false_count})

    def log_model_params(self, model):
        log_model_parameters_info(model, self.skip_wandb)


