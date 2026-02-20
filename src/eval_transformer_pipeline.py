import evaluate
from tqdm import tqdm

rouge = evaluate.load("rouge")


def evaluate_gpt2_rouge(generator, texts, tokenizer, max_new_tokens=50):
    predictions = []
    references = []

    for text in tqdm(texts):

        words = text.split()
        if len(words) < 8:
            continue  # пропускаем слишком короткие тексты

        split_idx = (3 * len(words)) // 4

        prefix_text = " ".join(words[:split_idx])
        target_text = " ".join(words[split_idx:])

        # Генерация
        output = generator(
            prefix_text,
            max_new_tokens=2,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_full = output[0]["generated_text"]

        # убираем префикс 
        generated_cont = generated_full[len(prefix_text):].strip()

        predictions.append(generated_cont)
        references.append(target_text)

    results = rouge.compute(predictions=predictions, references=references)

    print("Metric: GPT2 ROUGE")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results
