import argparse
import pandas as pd
from tqdm import tqdm
from utils.logger import setup_logger, get_logger

tqdm.pandas()

def prepare_row(row) -> str:
    return f"Расклад медичи: {row['cards']}, значение : {row['meaning']}"

def prepare_data(input_file: str, output_file: str):
    logger = get_logger()

    logger.info(f'Загрузка датасета из \"{input_file}\"')
    data = pd.read_csv(input_file, sep=";")
    
    logger.info('Обработка датасета')
    data = data.progress_apply(prepare_row, axis=1)
    data.drop_duplicates()

    logger.info(f'Сохраниение датасета в \"{output_file}\"')
    data.to_csv(output_file, index=False, header=['text'])
    logger.info(f'Датасет сохранен')


def main():
    parser = argparse.ArgumentParser(description="Подготовка данных")
    parser.add_argument("--input-file", type=str, default='../data/data.csv', help="Сырой датасет")
    parser.add_argument("--output-file", type=str, default='../data/processed.csv', help="Выходной файл")
    parser.add_argument("--log-file", type=str, default='../logs/prepare_data.log', help="Файл логирования")
    args = parser.parse_args()

    setup_logger(args.log_file)
    prepare_data(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
