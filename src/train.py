import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
import torch
from datasets import load_dataset
from utils.logger import setup_logger, get_logger


class CustomLoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                self.logger.info(f"{key}: {value}")
        return control


def train(
        model_name: str,
        data_path: str,
        epochs: int,
        batch_size: int,
        max_length: int,
        test_size: float,
        model_path: str,
        use_gpu: bool
        ):
    logger = get_logger()
    custom_callback = CustomLoggingCallback(logger)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f'Обучение на \"{device}\"')

    logger.info(f'Загрузка предобработоного датасета из \"{data_path}\"')
    dataset = load_dataset('csv', data_files=data_path)['train']

    logger.info(f'Загрузка модели \"{model_name}\"')
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.gradient_checkpointing_enable()

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    
    logger.info('Токенизация')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=model_path,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_total_limit=2,
        fp16=True,
        use_cpu=not use_gpu
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    splited_dataset = tokenized_datasets.train_test_split(test_size=test_size)
    train_dataset = splited_dataset['train']
    eval_dataset = splited_dataset['test']


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[custom_callback]
    )

    logger.info('Обучение')
    trainer.train()

    logger.info(f'Сохранение модели в \"{model_path}\"')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f'Модель сохранена')


def main():
    parser = argparse.ArgumentParser(description='Обучение GPT-2 для генерации комментариев')
    parser.add_argument('--model-name', type=str, default='gpt2', help='Название модели для дообучения')
    parser.add_argument('--data-path', type=str, default='../data/processed.csv', help='Файл с предобработаными данными')
    parser.add_argument('--epochs', type=int, default=3, help='Количество эпох')
    parser.add_argument('--batch-size', type=int, default=4, help='Размер batch')
    parser.add_argument('--max-length', type=int, default=124, help='Максимальная длинна контекста')
    parser.add_argument('--test-size', type=float, default=0.1, help='Размер валидационной выборки')
    parser.add_argument('--model-path', type=str, default='../models/gpt2-midich', help='Куда сохранить модель')
    parser.add_argument('--gpu', type=bool, action=argparse.BooleanOptionalAction, default=True, help='Проводить обучение на GPU')
    parser.add_argument("--log-file", type=str, default='../logs/train.log', help="Файл логирования")
    args = parser.parse_args()

    setup_logger(args.log_file)
    train(
        args.model_name,
        args.data_path,
        args.epochs,
        args.batch_size,
        args.max_length,
        args.test_size,
        args.model_path,
        args.gpu
    )


if __name__ == '__main__':
    main()
