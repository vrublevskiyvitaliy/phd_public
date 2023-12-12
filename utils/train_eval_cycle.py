import evaluate
import torch
from tqdm.auto import tqdm

def train_eval(model, optimizer, lr_scheduler,  train_dataloader, eval_dataloader, num_train_epochs, num_training_steps, device):
  progress_bar = tqdm(range(num_training_steps))

  for epoch in range(num_train_epochs):
      print(f"Epoch {epoch}")
      model.train()
      for batch in train_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
          loss.backward()

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)

      accuracy_metric = evaluate.load("accuracy")
      f1_metric = evaluate.load("f1")
      model.eval()
      for batch in eval_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          with torch.no_grad():
              outputs = model(**batch)

          logits = outputs.logits
          predictions = torch.argmax(logits, dim=-1)
          accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
          f1_metric.add_batch(predictions=predictions, references=batch["labels"])

      acc = accuracy_metric.compute()
      f1 = f1_metric.compute()
      print(f"Accuracy {acc['accuracy']}")
      print(f"F1 {f1['f1']}")

def train_eval_test(model, optimizer, lr_scheduler,  train_dataloader, eval_dataloader, test_dataloader, num_train_epochs, num_training_steps, device):
  progress_bar = tqdm(range(num_training_steps))

  for epoch in range(num_train_epochs):
      print(f"Epoch {epoch}")
      model.train()
      for batch in train_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
          loss.backward()

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)

      ev_accuracy_metric = evaluate.load("accuracy")
      ev_f1_metric = evaluate.load("f1")
      model.eval()
      for batch in eval_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          with torch.no_grad():
              outputs = model(**batch)

          logits = outputs.logits
          predictions = torch.argmax(logits, dim=-1)
          ev_accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
          ev_f1_metric.add_batch(predictions=predictions, references=batch["labels"])

      acc = ev_accuracy_metric.compute()
      f1 = ev_f1_metric.compute()
      print(f"Eval accuracy {acc['accuracy']:.2f}")
      print(f"Eval F1 {f1['f1']:.2f}")

      test_accuracy_metric = evaluate.load("accuracy")
      test_f1_metric = evaluate.load("f1")
      model.eval()
      for batch in test_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          with torch.no_grad():
              outputs = model(**batch)

          logits = outputs.logits
          predictions = torch.argmax(logits, dim=-1)
          test_accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
          test_f1_metric.add_batch(predictions=predictions, references=batch["labels"])

      acc = test_accuracy_metric.compute()
      f1 = test_f1_metric.compute()
      print(f"Test accuracy {acc['accuracy']:.2f}")
      print(f"Test F1 {f1['f1']:.2f}")
