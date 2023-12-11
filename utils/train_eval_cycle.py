import evaluate
from tqdm.auto import tqdm

def train_eval(model, optimizer, lr_scheduler,  train_dataloader, eval_dataloader, num_train_epochs, num_training_steps):
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