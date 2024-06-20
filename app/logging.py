import wandb
import torch

from transformers import GPT2Tokenizer

from app.viz import log_model_parameters_info

class MetricsLogger():
    def __init__(self, skip_wandb=False):
        if not skip_wandb:
            self.batch1_table = wandb.Table(columns=["epoch", "idx", "input_text", "label", "prediction", "logits"])
        else:
            self.batch1_table = None

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []
        self.epoch = 0
        self.skip_wandb = skip_wandb

    def init(self, *args, **kwargs):
        if not self.skip_wandb:
            wandb.init(*args, **kwargs)

    def step_epoch(self):
        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        avg_train_accuracy = sum(self.train_acc) / len(self.train_acc)
        self.train_losses = []
        self.train_acc = []
        print(f"Epoch {self.epoch + 1}, Train Loss: {avg_train_loss}")

        avg_test_loss = sum(self.test_losses) / len(self.test_losses)
        avg_test_accuracy = sum(self.test_acc) / len(self.test_acc)
        print(f"Epoch {self.epoch + 1}, Test Loss: {avg_test_loss}")

        if not self.skip_wandb:
            wandb.log({"train_loss": avg_train_loss, "train_accuracy": avg_train_accuracy})
            wandb.log({"test_loss": avg_test_loss, "test_accuracy": avg_test_accuracy})


        self.epoch += 1

    def log_train_batch(self, loss, labels, outputs, lr):
        self.train_losses.append(loss)
        self.train_acc.append((torch.argmax(outputs, dim=-1) == labels).sum().item())
        if not self.skip_wandb:
            wandb.log({"train_batch_loss": loss, "batch_learning_rate": lr})

    def log_test_batch(self, batch_idx, loss, labels, outputs, input_ids):
        self.test_losses.append(loss)
        self.test_acc.append((torch.argmax(outputs, dim=-1) == labels).sum().item())

        if batch_idx == 0 and not self.skip_wandb:
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids[:4]]

            predictions = torch.argmax(outputs, dim=-1)
            
            for i in range(min(len(input_texts), 4)):
                self.batch1_table.add_data(self.epoch, i, input_texts[i], labels[i], predictions[i], outputs[i])

    def finalize(self):
        if not self.skip_wandb:
            wandb.log({f"batch_1": self.batch1_table})
            wandb.finish()

    def log_label_ratio(self, dataset, name):
        labels = []
        for i in range(min(300, len(dataset))):
            labels.append(dataset.get_label(i))
        
        true_count = sum(1 for item in labels if item == 1)
        false_count = len(labels) - true_count
        print(f"{name} True labels: {true_count}, False labels: {false_count}")
        print(f"{name} Ratio (True/False): {true_count/false_count:.2f}")

        if not self.skip_wandb:
            wandb.log({f"{name}_data_true_count": true_count, f"{name}_data_false_count": false_count, f"{name}_data_TF_ratio": true_count/false_count})

    def log_model_params(self, model):
        log_model_parameters_info(model, self.skip_wandb)
