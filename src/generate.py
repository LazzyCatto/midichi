import torch
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextStreamer
from utils.logger import setup_logger, get_logger


def generate(model_path: str, cards: str, max_length: int, temperature: float, top_p: float):
    logger = get_logger()
    
    logger.info(f'Загрузка предобученных моделей из \"{model_path}\"')
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    prompt = f"Расклад медичи: {cards}, значение : "
    logger.info(f'Промпт \"{prompt}\"')

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    with torch.no_grad():
        streamer = TextStreamer(tokenizer, skip_prompt=True)

        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length + len(input_ids[0]),
            num_return_sequences=1,
            temperature=temperature, 
            top_p=top_p, 
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Генерация комментариев GPT-2")
    parser.add_argument("--model-path", type=str, default="../models/gpt2-midich", help="Путь к модели")
    parser.add_argument("--cards", type=str, required=True, help="Получившийся расклад")
    parser.add_argument("--log-file", type=str, default='../logs/generate.log', help="Файл логирования")
    parser.add_argument("--max-length", type=int, default=100, help="Максимальная длинна текста")
    parser.add_argument("--temperature", type=float, default=0.7, help="Температура")
    parser.add_argument("--top-p", type=float, default=0.7, help="Top p")
    
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    generated_text = generate(args.model_path, args.cards, args.max_length, args.temperature, args.top_p)
    logger.info(f'значение расклада:\n\"{generated_text}\"')


if __name__ == '__main__':
    main()
