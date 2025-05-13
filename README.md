# Midich

Консольная утилита для расклада карт по системе Мидичи. Позволяет наблюдать пошаговую визуализацию процесса, а также управлять цветами мастей и скоростью выполнения.

---

## Установка

Установите напрямую из PyPI:

```bash
pip install cat-midich
```

Или локально из исходников:

```bash
git clone https://github.com/LazzyCatto/midichi
cd midich
pip install .
```

---

## Использование

```bash
midich [опции]
```

### Доступные флаги:

| Флаг               | Описание                          | По умолчанию |
|--------------------|-----------------------------------|--------------|
| `--guided`         | Показ пошаговой визуализации      | `False`      |
| `--sleep`          | Задержка между шагами (в секундах)| `0.1`        |
| `--add-color`      | Цвет новой карты                  | `green`      |
| `--diamond-color`  | Цвет масти ♦                      | `red`        |
| `--hearts-color`   | Цвет масти ♥                      | `red`        |
| `--clubs-color`    | Цвет масти ♣                      | `white`      |
| `--spades-color`   | Цвет масти ♠                      | `white`      |

Пример:
```bash
midich --guided --sleep 0.2 --add-color yellow --diamond-color cyan --spades-color blue
```

---

## Классы

### CardSuit

`enum` с возможными мастями карт
- `CardSuit.DIAMOND` - буби ♦
- `CardSuit.HEARTS` - черви ♥
- `CardSuit.SPADES` - трефы ♣
- `CardSuit.CLUBS` - пики ♠

`__str__` и `__repr__` возвращают unicode символ мастей (♦, ♥, ♣, ♠)

### CardValue

`enum` с возможными значениями карт
- `CardValue.SIX` - "6"
- `CardValue.SEVEN` - "7"
- `CardValue.EIGHT` - "8"
- `CardValue.NINE` - "9"
- `CardValue.TEN` - "10"
- `CardValue.JACK` - "J"
- `CardValue.QUEEN` - "Q"
- `CardValue.KING` - "K"
- `CardValue.ACE` - "A"

`__str__` и `__repr__` возвращают символьное описание значения ("6", "7", "8", "9", "10", "J", "Q", "K", "A")

`@classmethod from_str(cls, new_value: str) -> "CardValue"` - возвращает `CardValue` из строки. В случае попытки создать `CardValue` из некорректного состояния - бросает исключение `InvalidCardValue`

### Card

Класс карты. Содержит поля
- `suit: CardSuit`
- `value: CardValue`

`@property rank(self) -> int` - возвращает числиное значение веса карты. Для всех числовых крат - значение сохранится, а все буквенные карты перейдут в соответствующие числа
- "J" -> 11
- "Q" -> 12
- "K" -> 13
- "A" -> 1

### Deck

Класс колоды. Изначально создается неперемешанная колода. Содержит поля
- `cards: list[Card]` - список карт

- `shuffle() -> None` - перемешивает колоду
- `pop() -> Card` - достает первую карту из колоды (в случае попытки достать карту из пустой колоды - бросает исключение `EmptyDeckAccess`)

`__str__` и `__repr__` выписывают все карты подряд через пробел

`__bool__` - проверяет колоду на наличие карт

---

## Методы

### расклад

`midich(guided: bool = False, **kwargs) -> list[Card]` - симуляция расклада с новой колодой
- `guided` - если `True` будет печатать подробный расклад, если `False` - только выдаст результат
- `kwargs`:
    - `sleep_time` - (необходимо, если `guided=True`) время ожидения между инструкциями
    - `suit_colors` - (необходимо, если `guided=True`) словарь с цветами мастей
    - `add_color` - (необходимо, если `guided=True`) цвет последней добавленной карты
В случае, есои не все необходимые параметры заданы - бросает исключение `GuidedMissingParameters`

### utils

`colored_card(card: Card, suit_colors: dict[str, str], attrs) -> str` - генерация цветного текста карты
- `card` - карта
- `suit_colors` - словарь с цветами мастей
- `attrs` - дополнительные тпараметры для `termcolor.colored`

`print_table(cards, sleep_time: float, suit_colors: dict[str, str], highlight_indices: tuple[int]=(), **kwargs) -> None` - генерация цветного текста списка карт
- `cards` - список карт
- `sleep_time` - время ожидания после написания карт
- `suit_colors` - словарь с цветами мастей
- `highlight_indices` - индексы карт, которые будут подчеркнуты
- `**kwargs` - дополнительные параметры
    - `add_color: str` - цвет последней карты

---

## В планах

- Генерация аналитики расклада с помощью лёгкой языковой модели (например, GPT-2)
- Возможность загрузки собственной колоды

---

## Автор

**Akio**