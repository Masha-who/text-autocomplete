import torch
import evaluate

from tqdm import tqdm

rouge = evaluate.load("rouge")

def evaluate_rouge(model, loader, tokenizer, max_new_tokens=20):
    model.eval()

    predictions = []
    references = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader):
            x_batch = x_batch.to(device)  # [B, T]
            y_batch = y_batch.to(device)  # [B, T]

            T = x_batch.size(1)
            prefix_len = (3 * T) // 4
            gen_len = T - prefix_len
            gen_len = min(gen_len, max_new_tokens)

            for i in range(x_batch.size(0)):
                prefix_ids = x_batch[i][:prefix_len]          # 3/4 входа
                target_ids = y_batch[i][prefix_len:prefix_len + gen_len]  # оставшаяся 1/4 (по смещённому таргету)

                # генерируем gen_len (разницу) токенов
                generated_ids = model.generate(prefix_ids, max_new_tokens=gen_len)

                # сгенерированное продолжение
                gen_cont = generated_ids[-gen_len:]

                pred_text = tokenizer.decode(gen_cont.tolist(), skip_special_tokens=True)
                ref_text = tokenizer.decode(target_ids.tolist(), skip_special_tokens=True)

                predictions.append(pred_text)
                references.append(ref_text)

    results = rouge.compute(
        predictions=predictions,
        references=references
    )

    print('Metrics')
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results