from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "facebook/bart-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save to disk
model.save_pretrained("./models/bart-large-mnli")
tokenizer.save_pretrained("./models/bart-large-mnli")
print(f"Model and tokenizer saved to ./models/bart-large-mnli")