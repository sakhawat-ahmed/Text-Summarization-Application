from transformers import BartTokenizer, BartForConditionalGeneration

class TextSummarizer:
    def __init__(self):
        model_path = "local_models/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained("/app/local_models/bart-large-cnn")
        self.model = BartForConditionalGeneration.from_pretrained("/app/local_models/bart-large-cnn")

    def summarize(self, text):
        inputs = self.tokenizer(
            [text],
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=200,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
