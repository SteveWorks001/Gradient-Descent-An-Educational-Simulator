# -*- coding: utf-8 -*-
"""
Учебная программа: градиентный спуск на одной функции потерь.

L(w) = (w - 3)²          dL/dw = 2(w - 3)          w_new = w - η · dL/dw

Никакого машинного обучения, датасетов и сторонних библиотек: только
стандартная библиотека Python (tkinter, math). Приложение рисует функцию
потенциальной энергии и показывает, как параметр w спускается по ней к
минимуму — ровно одним шагом на одну кнопку.
Язык интерфейса переключается на ходу: русский (по умолчанию) или английский.

Где что искать:
  0.  Все надписи (ru/en) ......... LANG / TEXTS / T(), справки GUIDE_RU и GUIDE_EN
  1.  Настройки и пределы ......... DEFAULT_*, ETA_*, W_SPAN, TOL_*
  2.  Примеры функций ............. EXAMPLES, loss_of(), grad_of()
  3.  Движок спуска ............... class GDStep / class Descent
  4.  Текст расчёта ............... calc_block(), what_happened(), status_of()
  5.  График ...................... class GraphView (tkinter Canvas)
  6.  Поля ввода .................. NumberControl (поле + ползунок)
  7.  Окно ........................ class GradientDescentApp
"""

import math
import tkinter as tk
from tkinter import ttk, font as tkfont

# ---------------------------------------------------------------
# 1. Настройки, пределы и цветовая палитра
# ---------------------------------------------------------------
DEFAULT_ETA = 0.10           # скорость обучения по умолчанию
# Верхний предел η выбран больше 1 намеренно: для L = (w-3)² множитель (1 - 2η)
# по модулю превышает 1 ровно при η > 1, и только тогда видно настоящую
# расходимость. Чтобы интерфейс при этом не разрастался, есть MAX_DIST.
ETA_MIN, ETA_MAX = 0.005, 1.20
W_SPAN = 6.0                 # начальный w можно выбрать в пределах min ± 6
MAX_STEPS = 100              # предел на всякий случай (для очень малых η)
MAX_DIST = 60.0              # дальше этого спуск считается ушедшим в разнобой

# Минимальный размер окна: ниже оконная сетка не вмещает три колонки, и нижняя
# строка уезжает за край. Перемер запросов сети в обоих языках и во всех состояниях
# спуска: при ширине 1320 px сетка просит 1287x666, запас по ширине 33 px.
MIN_W, MIN_H = 1320, 700
# По высоте 960 px панель расчёта = 825 px, а самый длинный блок (сходимость и
# разнобой, 37 перенесённых строк) просит 823 px — нижняя строка видна целиком.
DEFAULT_W, DEFAULT_H = 1460, 960

# Допуск, ниже которого считаем, что минимум достигнут.
TOL_DIST = 1e-3              # |w - w*| — расстояние до минимума
TOL_GRAD = 2e-3              # |dL/dw| — насколько плоское место

COL_TEXT = "#242424"
COL_BG = "#fbfbf7"                 # фон окна
COL_PAPER = "#fdfdf9"              # фон «бумажных» панелей
COL_CURVE = "#3f6f96"              # сама функция потерь
COL_POINT = "#c8651b"              # текущая точка
COL_POINT_FILL = "#fbeada"
COL_TRAIL = "#5b3a86"              # траектория спуска
COL_TRAIL_FILL = "#ece5f5"
COL_MIN = "#2f7d4f"                # минимум функции
COL_MIN_FILL = "#e2f0e6"
COL_ACTIVE = "#b07d10"             # янтарная подсветка шага
COL_DIM = "#91918a"
COL_FRAME = "#c8c8c0"
COL_LINE = "#8d949b"               # оси и деления
COL_POS = "#2f7d4f"
COL_NEG = "#b3261e"
COL_BAD_BG = "#fde8e6"
COL_BAD_FG = "#7a1a13"

# ---------------------------------------------------------------
# 2. Примеры: L(w), её производная и минимум
#    Формулы записаны математическими символами — они одинаковы на
#    обоих языках, поэтому не нуждаются в переводе.
# ---------------------------------------------------------------
FORMULA_MAIN = "L(w) = (w − 3)²"
FORMULA_DERIV = "dL/dw = 2(w − 3)"
MIN_NAME = "w*"

# loss_sub и grad_sub — те же формулы, но с {v} вместо w: в него подставляется
# конкретное число, и строка вида «L(0.5000) = (0.5000 − 3)² = 6.2500» собирается
# прямо из примера, без разветвлений по языку и по функции.
EXAMPLES = (
    {
        "id": "ex1",
        "min": 3.0,
        "formula": FORMULA_MAIN,
        "deriv": FORMULA_DERIV,
        "loss_sub": "({v} − 3)²",
        "grad_sub": "2({v} − 3)",
        "loss": lambda w: (w - 3.0) ** 2,
        "grad": lambda w: 2.0 * (w - 3.0),
        "w0": 0.5,
        "eta": DEFAULT_ETA,
    },
    {
        "id": "ex2",
        "min": -2.0,
        "formula": "L(w) = (w + 2)²",
        "deriv": "dL/dw = 2(w + 2)",
        "loss_sub": "({v} + 2)²",
        "grad_sub": "2({v} + 2)",
        "loss": lambda w: (w + 2.0) ** 2,
        "grad": lambda w: 2.0 * (w + 2.0),
        "w0": 2.5,
        "eta": DEFAULT_ETA,
    },
    {
        "id": "ex3",
        "min": 4.0,
        "formula": "L(w) = 0.5(w − 4)²",
        "deriv": "dL/dw = (w − 4)",
        "loss_sub": "0.5({v} − 4)²",
        "grad_sub": "({v} − 4)",
        "loss": lambda w: 0.5 * (w - 4.0) ** 2,
        "grad": lambda w: w - 4.0,
        "w0": 0.5,
        "eta": 0.40,
    },
)
EXAMPLE_KEYS = [ex["id"] for ex in EXAMPLES]
DEFAULT_EXAMPLE = EXAMPLES[0]

# Отдельная кнопка-демонстрация: та же функция, но η заведомо слишком велика.
# Множитель (1 − 2η) = −1.1: каждый прыжок через минимум больше предыдущего.
BIG_ETA = 1.05

# ---------------------------------------------------------------
# 3. Движок градиентного спуска: вся математика собрана здесь
# ---------------------------------------------------------------
class GDStep:
    """Один шаг спуска: все промежуточные числа остаются видимыми."""

    def __init__(self, index, w, loss, grad, eta, min_w):
        self.index = index          # номер шага, от нуля
        self.w = w                  # параметр ДО этого шага
        self.loss = loss            # L(w)
        self.grad = grad            # dL/dw в точке w
        self.eta = eta              # скорость обучения этого шага
        self.delta = -eta * grad    # Δw = -η · dL/dw
        self.w_new = w + self.delta
        self.dist = abs(w - min_w)          # |w - w*| до шага
        self.dist_new = abs(self.w_new - min_w)
        self.loss_new = 0.0                 # заполняет Descent
        self.grad_new = 0.0


class Descent:
    """Состояние спуска: параметр, скорость обучения и вся история шагов."""

    def __init__(self, example, w0, eta):
        self.example = example
        self.w0 = w0
        self.eta = eta
        self.steps = []                     # список GDStep
        self._reset(w0)

    def _reset(self, w):
        self.w = w
        self.loss = self.example["loss"](w)
        self.grad = self.example["grad"](w)
        self.steps = []

    def restart(self, w=None, eta=None):
        """Начать заново (с новой точки или с новой скоростью обучения)."""
        if w is None:
            w = self.w0
        if eta is not None:
            self.eta = eta
        self._reset(w)

    @property
    def min_w(self):
        return self.example["min"]

    def step(self):
        """Ровно одна итерация: w -> L(w) -> dL/dw -> Δw -> w_new."""
        if self.stuck():
            return None
        s = GDStep(len(self.steps), self.w, self.loss, self.grad, self.eta, self.min_w)
        s.loss_new = self.example["loss"](s.w_new)
        s.grad_new = self.example["grad"](s.w_new)
        self.steps.append(s)
        self.w, self.loss, self.grad = s.w_new, s.loss_new, s.grad_new
        return s

    def stuck(self):
        """Шаг делать незачем: минимум достигнут, уперлись в предел шагов или
        спуск ушёл в разнобой слишком далеко от минимума."""
        return (self.converged() or len(self.steps) >= MAX_STEPS
                or abs(self.w - self.min_w) > MAX_DIST)

    def converged(self):
        """Минимум достигнут в пределах допуска: близко и почти горизонтально."""
        return (abs(self.w - self.min_w) < TOL_DIST
                and abs(self.grad) < TOL_GRAD)

    def blown(self):
        """Спуск разсошёлся: параметр улетел далеко от минимума."""
        return abs(self.w - self.min_w) > MAX_DIST

    def last_step(self):
        return self.steps[-1] if self.steps else None

    def step_preview(self):
        """Тот же шаг, что выполнит next_step, но ничего не меняя: для таблицы."""
        return GDStep(len(self.steps), self.w, self.loss, self.grad, self.eta,
                      self.min_w)

    def positions(self):
        """w на каждом шаге, включая начальную точку."""
        return [s.w for s in self.steps] + [self.w]


# ---------------------------------------------------------------
# 4. Локализация: весь текст интерфейса живёт в TEXTS
#    (русский по умолчанию)
# ---------------------------------------------------------------
LANG = "ru"

# Настоящий математический минус: в формулах он должен выглядеть одинаково
# со всех сторон, а не спорить с ASCII-дефисом.
MINUS = "−"

GUIDE_RU = """1. ЧТО ТАКОЕ ОПТИМИЗАЦИЯ
Оптимизация — это поиск лучших значений параметров. «Лучше» означает, что
некоторая функция принимает наименьшее возможное значение. Мы перебираем числа,
каждый раз чуть-чуть двигаясь, и смотрим, стало ли лучше.

2. ФУНКЦИЯ ПОТЕРЬ
Функция потерь L — это «насколько плохо». В этой программе она всегда зависит от
одного-единственного параметра w. Например, L(w) = (w − 3)²: при w = 3 потери
равны нулю, а чем сильнее w отклонился от тройки, тем больше штраф. Обучение
нейросети устроено так же, только параметров там миллионы, а не один.

3. ПАРАМЕТР (ВЕС)
Параметр w — это число, которое мы подбираем. Всё, что умеет программа, — это
менять одно число и смотреть, как меняется L(w). Само по себе w ничего не
означает: важна только величина потерь, которую оно даёт.

4. ПРОИЗВОДНАЯ
Производная dL/dw отвечает на вопрос: «если я увеличу w на самую малую величину,
насколько вырастут потери?» Для L(w) = (w − 3)² это dL/dw = 2(w − 3). При w = 0.5
производная равна −5: потери уменьшаются, если w увеличивать.

5. ГРАДИЕНТ
Градиент — то же самое, что производная, когда параметр один. Когда параметров
много, градиент собирает все частные производные в один вектор и показывает
направление наиболее быстрого роста функции.

6. ПОЧЕМУ ГРАДИЕНТ ВЫЧИТАЮТ
w_new = w − η · dL/dw. Градиент показывает, куда расти. Значит, чтобы падать,
надо идти против него — то есть вычесть. Из-за этого минуса спуск и называется
спуском. Если бы мы прибавляли градиент, мы бы поднимались вверх по функции
потерь, то есть ухудшали результат.

7. СКОРОСТЬ ОБУЧЕНИЯ
η (эта) — размер шага. Она превращает направление в конкретное число:
Δw = −η · dL/dw. Маленькое η — много крохотных шагов, обучение идёт медленно.
Большое η — грубые прыжки, можно проскочить минимум. η подбирают по поведению:
здесь для этого есть ползунок и кнопка с заведомо большим значением.

8. ОДИН ШАГ ГРАДИЕНТНОГО СПУСКА
Пять действий, и они повторяются бесконечно:
    посчитать L(w);
    посчитать dL/dw;
    посчитать Δw = −η · dL/dw;
    обновить w;
    повторить.
Кнопка «Следующий шаг» делает ровно один такой цикл, чтобы его можно было
разобрать построчно.

9. СХОДИМОСТЬ
Сходимость — это когда дальнейшие шаги почти ничего не меняют: параметр
остановился возле минимума, а градиент стал близок к нулю. Здесь минимум
считается достигнутым, когда |w − w*| < 0.001 и |dL/dw| < 0.002. Про «почти»
честно сказано в статусе: точного равенства программа не обещает.

10. КОГДА η СЛИШКОМ БОЛЬШАЯ
Обновление w_new = w − η · dL/dw для этой функции дает w_new − 3 = (1 − 2η)(w − 3).
Множитель (1 − 2η) решает всё:
    0 < η < 0.5  — он положительный и меньше 1: параметры идут к минимуму ровно,
                   без перемахивания;
    η = 0.5      — множитель равен 0: один шаг, и мы точно в минимуме;
    0.5 < η < 1  — он отрицательный, по модулю меньше 1: точка перескакивает
                   минимум, но прыжки затухают и сходимость всё равно наступает;
    η = 1        — модуль равен 1: вечные прыжки симметрично вокруг минимума;
    η > 1        — модуль больше 1: каждый прыжок больше предыдущего, потери
                   растут, спуск разваливается.
Поэтому «градиентный спуск» вовсе не гарантия движения к минимуму: всё зависит
от выбранного размера шага. Кнопка «Слишком большая η» ставит η = 1.05 — и путь
параметра сразу уходит в расходящиеся прыжки.

Автор: SteveWorks001
"""

GUIDE_EN = """1. WHAT OPTIMIZATION IS
Optimization is the search for the best values of parameters. "Best" means some
function reaches its smallest possible value. We nudge the numbers over and over
and check whether things got better.

2. THE LOSS FUNCTION
The loss function L measures "how bad". In this program it always depends on a
single parameter w. Take L(w) = (w − 3)²: at w = 3 the loss is zero, and the
further w moves away from 3, the bigger the penalty. Training a neural network
works the same way, only there the parameters number in the millions.

3. THE PARAMETER (WEIGHT)
The parameter w is the number we are tuning. All this program does is change one
number and watch how L(w) reacts. The symbol w means nothing by itself; what
matters is the loss it produces.

4. THE DERIVATIVE
The derivative dL/dw answers the question: "if I increase w by a tiny amount,
how much does the loss grow?" For L(w) = (w − 3)² it is dL/dw = 2(w − 3).
At w = 0.5 the derivative is −5, so the loss goes down when w increases.

5. THE GRADIENT
The gradient is just the derivative when there is a single parameter. With many
parameters the gradient collects all partial derivatives into one vector and
points in the direction of steepest increase of the function.

6. WHY WE SUBTRACT THE GRADIENT
w_new = w − η · dL/dw. The gradient points uphill, so to go downhill we move
against it — which means subtracting. That minus sign is the reason the method
is called descent. Adding the gradient would climb the loss function and make
the result worse.

7. THE LEARNING RATE
η (eta) is the step size. It turns a direction into an actual number:
Δw = −η · dL/dw. A small eta means many tiny steps and slow progress. A large
eta means wild jumps and a risk of overshooting the minimum. You can pick it
with the slider, and there is a button with a deliberately too-large value.

8. ONE GRADIENT DESCENT STEP
Five actions, repeated over and over:
    compute L(w);
    compute dL/dw;
    compute Δw = −η · dL/dw;
    update w;
    repeat.
The "Next Step" button performs exactly one of those cycles, so that every line
of it can be inspected.

9. CONVERGENCE
Convergence is when further steps barely change anything: the parameter has
settled near the minimum and the gradient has become close to zero. Here the
minimum counts as reached when |w − w*| < 0.001 and |dL/dw| < 0.002. The word
"almost" is stated honestly in the status line — the program never claims
exact equality.

10. WHEN η IS TOO LARGE
For this function the update w_new = w − η · dL/dw gives
w_new − 3 = (1 − 2η)(w − 3). The factor (1 − 2η) decides everything:
    0 < eta < 0.5 — it is positive and below 1: the parameter walks to the
                    minimum smoothly, never passing it;
    eta = 0.5     — the factor is 0: one step lands exactly on the minimum;
    0.5 < eta < 1 — it is negative with magnitude below 1: the point overshoots,
                    but the jumps shrink and the descent still converges;
    eta = 1       — magnitude exactly 1: endless symmetric jumps;
    eta > 1       — magnitude above 1: every jump is bigger than the last one,
                    the loss grows and the descent falls apart.
So "gradient descent" is no guarantee of moving toward the minimum: everything
depends on the chosen step size. The "Too large η" button sets η = 1.05 and the
path immediately turns into diverging jumps.

Author: SteveWorks001
"""

TEXTS = {
    "ru": {
        "app_title": "Градиентный спуск",
        "subtitle": "учебный симулятор · как параметр спускается по функции потерь",
        "lang_label": "Язык",
        "lang_ru": "RU",
        "lang_en": "EN",

        "box_function": "Функция потерь",
        "box_controls": "Что можно менять",
        "box_examples": "Примеры",
        "lbl_w0": "Начальный w",
        "lbl_eta": "η — скорость обучения",

        "btn_step": "Следующий шаг",
        "btn_auto": "Автоматически",
        "btn_stop": "Остановить",
        "btn_reset": "Сброс",
        "btn_big_eta": "Слишком большая η",
        "btn_ex1": "1. (w − 3)²",
        "btn_ex2": "2. (w + 2)²",
        "btn_ex3": "3. 0.5(w − 4)²",

        "graph_title": "Функция потерь и путь параметра",
        "axis_w": "w",
        "axis_loss": "L(w)",
        "lbl_minimum": "минимум",
        "lbl_start": "старт",
        "lbl_current": "сейчас",
        "gradient_dir": "градиент",
        "update_dir": "шаг обновления",

        "tab_calc": "Расчёт шага",
        "tab_history": "История шагов",
        "tab_guide": "Справка",

        "head_step": "Шаг",
        "head_w": "w",
        "head_loss": "L(w)",
        "head_grad": "dL/dw",
        "head_delta": "Δw",
        "history_note": "Каждая строка — один полный шаг спуска. Последний шаг ещё не "
                        "применён: он показывает, куда параметр пойдёт дальше.",
        "history_empty": "История пуста: нажмите «Следующий шаг».",
        "history_rows": "Шагов сделано: {n}",

        "what_title": "Что произошло?",
        "cur_step": "Текущий шаг: {n}",

        "status_ready": "Готово. Меняйте w и η — и нажимайте «Следующий шаг».",
        "status_lang": "Язык интерфейса переключён.",
        "status_reset": "Спуск сброшен: начальная точка, η, счётчик и траектория как при запуске.",
        "status_err": "Поле «{field}»: «{raw}» — не число. Последние правильные значения сохранены.",
        "status_range": "Поле «{field}»: {raw} вне разрешённого предела. Последние правильные значения сохранены.",
        "status_example": "Пример «{name}»: {desc}",
        "status_step": "Шаг {n}: w = {w}, потери = {loss}, {dir}",
        "dir_down": "спуск к минимуму",
        "dir_up": "движение от минимума",
        "status_converged": "Сходимость: |w − w*| = {dist} < {tol_d} и |dL/dw| = {grad} < {tol_g}. Движение остановилось.",
        "status_max": "Достигнут предел в {n} шагов: при такой η спуск идёт очень медленно. Нажмите «Сброс» и возьмите η побольше.",
        "status_blown": "Спуск разсошёлся: w = {w} ушёл от минимума {m} дальше, чем на {dist}. Так ведёт себя слишком большая η. Нажмите «Сброс».",
        "status_big_eta": "η = {eta}: слишком большой шаг. Множитель (1 − 2η) по модулю больше 1, поэтому прыжки через минимум только растут.",
        "status_auto": "Автоматический режим: шаг каждые {ms} мс. Кнопка «Остановить» прерывает его.",

        "lbl_step": "ШАГ {n}",
        "lbl_pending": "ещё не сделан",
        "ln_param": "Параметр:",
        "ln_loss": "Потери:",
        "ln_grad": "Градиент:",
        "ln_eta": "Скорость обучения:",
        "ln_update": "Обновление:",
        "ln_new": "Новый параметр:",
        "ln_result": "Итог шага:",
        "tpl_named": "w = {v}",
        "tpl_loss_at": "L({w}) = {sub} = {v}",
        "tpl_grad_at": "dL/dw = {sub} = {v}",
        "tpl_eta": "η = {v}",
        "tpl_delta": "Δw = −η · dL/dw\n  Δw = −{e} · {g} = {v}",
        "tpl_w_new": "w_new = w − η · dL/dw\n  w_new = {w} − {e} · {g} = {v}",
        "tpl_after": "w = {v}        L(w) = {l}",
        "tpl_dist": "|w − w*| = |{w} − {m}| = {v}",

        "status_hdr": "Состояние:",
        "st_converged": "сошлось: параметр остановился у минимума (в пределах допуска)",
        "st_toward": "движемся к минимуму",
        "st_away": "движемся от минимума — шаг слишком велик",
        "st_idle": "шаг ещё не сделан",
        "st_blown": "спуск разсошёлся: каждый прыжок через минимум больше предыдущего",

        "what_start": "Нажмите «Следующий шаг»: будет рассчитана ровно одна итерация — "
                      "потери, градиент, обновление и новый параметр.",
        "what_neg": "Градиент = {g}: он отрицательный, значит функция падёт, если w "
                    "увеличивать.\nСпуск вычитает градиент:\n    w_new = w − η · dL/dw\n"
                    "Минус и отрицательный градиент дают плюс: w сдвинулся вправо, "
                    "к минимуму w* = {m}.\n{cmp}",
        "what_pos": "Градиент = {g}: он положительный, значит функция растёт при росте w.\n"
                    "Спуск вычитает градиент:\n    w_new = w − η · dL/dw\n"
                    "Поэтому w сдвинулся влево, к минимуму w* = {m}.\n{cmp}",
        "what_zero": "Градиент = {g}: настолько близок к нулю, что Δw = −η · dL/dw почти "
                     "исчезает. Параметр остановился — дальше шагу идти некуда.\n{cmp}",
        "cmp_closer": "Новое значение {wn} ближе к минимуму, чем {wo}: потери уменьшились "
                      "с {lo} до {ln}.",
        "cmp_farther": "Новое значение {wn} дальше от минимума, чем {wo}: потери выросли с "
                       "{lo} до {ln}. Это перемахивание через минимум — шаг слишком велик.",
        "cmp_same": "Расстояние до минимума почти не изменилось: {lo} → {ln}. Мы уже "
                    "почти у самой точки минимума.",

        "fn_deriv": "производная: {deriv}",
        "fn_min": "минимум при w* = {m}",

        "ex1_name": "(w − 3)²",
        "ex1_desc": "основной пример: минимум при w = 3, стартуем из 0.5",
        "ex2_name": "(w + 2)²",
        "ex2_desc": "тот же принцип, но минимум слева: w = −2",
        "ex3_name": "0.5(w − 4)²",
        "ex3_desc": "та же парабола только шире: минимум при w = 4",
        "big_eta_name": "Слишком большая η",
        "big_eta_desc": "устойчивость нарушается: вместо сходимости — расходящиеся прыжки",

        "guide": GUIDE_RU,
    },

    "en": {
        "app_title": "Gradient Descent",
        "subtitle": "an educational simulator · watching a parameter slide down a loss function",
        "lang_label": "Language",
        "lang_ru": "RU",
        "lang_en": "EN",

        "box_function": "Loss function",
        "box_controls": "What you can change",
        "box_examples": "Examples",
        "lbl_w0": "Initial w",
        "lbl_eta": "η — learning rate",

        "btn_step": "Next Step",
        "btn_auto": "Auto Run",
        "btn_stop": "Stop",
        "btn_reset": "Reset",
        "btn_big_eta": "Too large η",
        "btn_ex1": "1. (w − 3)²",
        "btn_ex2": "2. (w + 2)²",
        "btn_ex3": "3. 0.5(w − 4)²",

        "graph_title": "The loss function and the path of the parameter",
        "axis_w": "w",
        "axis_loss": "L(w)",
        "lbl_minimum": "minimum",
        "lbl_start": "start",
        "lbl_current": "now",
        "gradient_dir": "gradient",
        "update_dir": "update step",

        "tab_calc": "Step calculation",
        "tab_history": "Step history",
        "tab_guide": "Guide",

        "head_step": "Step",
        "head_w": "w",
        "head_loss": "L(w)",
        "head_grad": "dL/dw",
        "head_delta": "Δw",
        "history_note": "Each row is one complete descent step. The last row is not applied "
                        "yet: it shows where the parameter goes next.",
        "history_empty": "No steps yet — press \"Next Step\".",
        "history_rows": "Steps taken: {n}",

        "what_title": "What happened?",
        "cur_step": "Current step: {n}",

        "status_ready": "Ready. Change w or η, then press \"Next Step\".",
        "status_lang": "Interface language switched.",
        "status_example": "Example \"{name}\": {desc}",
        "status_reset": "Descent reset: starting point, η, counter and trajectory back to launch state.",
        "status_err": "Field \"{field}\": \"{raw}\" is not a number. The last valid values are kept.",
        "status_range": "Field \"{field}\": {raw} is outside the allowed range. The last valid values are kept.",
        "status_step": "Step {n}: w = {w}, loss = {loss}, {dir}",
        "dir_down": "moving down to the minimum",
        "dir_up": "moving away from the minimum",
        "status_converged": "Converged: |w − w*| = {dist} < {tol_d} and |dL/dw| = {grad} < {tol_g}. The parameter stopped.",
        "status_max": "The limit of {n} steps is reached: with this η the descent crawls. Press \"Reset\" and pick a larger η.",
        "status_blown": "The descent diverged: w = {w} is more than {dist} away from the minimum {m}. That is what an oversized η does. Press \"Reset\".",
        "status_big_eta": "η = {eta}: the step is too large. |1 − 2η| exceeds 1, so the jumps over the minimum keep growing.",
        "status_auto": "Auto run: one step every {ms} ms. \"Stop\" interrupts it.",

        "lbl_step": "STEP {n}",
        "lbl_pending": "not taken yet",
        "ln_param": "Parameter:",
        "ln_loss": "Loss:",
        "ln_grad": "Gradient:",
        "ln_eta": "Learning rate:",
        "ln_update": "Update:",
        "ln_new": "New parameter:",
        "ln_result": "Result of the step:",
        "tpl_named": "w = {v}",
        "tpl_loss_at": "L({w}) = {sub} = {v}",
        "tpl_grad_at": "dL/dw = {sub} = {v}",
        "tpl_eta": "η = {v}",
        "tpl_delta": "Δw = −η · dL/dw\n  Δw = −{e} · {g} = {v}",
        "tpl_w_new": "w_new = w − η · dL/dw\n  w_new = {w} − {e} · {g} = {v}",
        "tpl_after": "w = {v}        L(w) = {l}",
        "tpl_dist": "|w − w*| = |{w} − {m}| = {v}",

        "status_hdr": "Status:",
        "st_converged": "converged: the parameter settled next to the minimum (within tolerance)",
        "st_toward": "moving toward the minimum",
        "st_away": "moving away from the minimum — step too large",
        "st_idle": "no step taken yet",
        "st_blown": "the descent diverged: every jump over the minimum is bigger than the last",

        "what_start": "Press \"Next Step\": exactly one iteration will be worked out — loss, "
                      "gradient, update and the new parameter.",
        "what_neg": "Gradient = {g}: it is negative, which means the function falls when w "
                    "grows.\nDescent subtracts the gradient:\n    w_new = w − η · dL/dw\n"
                    "A minus times a negative gradient is a plus, so w moved to the right, "
                    "toward the minimum w* = {m}.\n{cmp}",
        "what_pos": "Gradient = {g}: it is positive, which means the function rises when w "
                    "grows.\nDescent subtracts the gradient:\n    w_new = w − η · dL/dw\n"
                    "So w moved to the left, toward the minimum w* = {m}.\n{cmp}",
        "what_zero": "Gradient = {g}: close enough to zero that Δw = −η · dL/dw almost "
                     "disappears. The parameter has stopped — there is nowhere left to go.\n{cmp}",
        "cmp_closer": "The new value {wn} sits closer to the minimum than {wo}: the loss fell "
                      "from {lo} to {ln}.",
        "cmp_farther": "The new value {wn} sits further from the minimum than {wo}: the loss "
                       "grew from {lo} to {ln}. That is overshooting — the step is too large.",
        "cmp_same": "The distance to the minimum barely changed: {lo} → {ln}. We are already "
                    "practically on top of it.",

        "fn_deriv": "derivative: {deriv}",
        "fn_min": "minimum at w* = {m}",

        "ex1_name": "(w − 3)²",
        "ex1_desc": "the main example: minimum at w = 3, starting from 0.5",
        "ex2_name": "(w + 2)²",
        "ex2_desc": "the same idea, but the minimum sits on the left: w = −2",
        "ex3_name": "0.5(w − 4)²",
        "ex3_desc": "the same parabola, only wider: minimum at w = 4",
        "big_eta_name": "Too large learning rate",
        "big_eta_desc": "stability breaks down: diverging jumps instead of convergence",

        "guide": GUIDE_EN,
    },
}


def T(key, **kw):
    """Текст текущего языка по ключу; подстановки {name} — только если переданы."""
    text = TEXTS[LANG][key]
    return text.format(**kw) if kw else text


def set_language(code):
    """Сменить язык интерфейса (обновление виджетов делает retranslate)."""
    global LANG
    LANG = code


def guide_text():
    """Справка на текущем языке (10 разделов)."""
    return TEXTS[LANG]["guide"]


# ---------------------------------------------------------------
# 5. Форматирование чисел: в формулах четыре знака, как в учебнике
# ---------------------------------------------------------------
def f4(v):
    """Число для формул: 0.5 -> «0.5000», мусор от float -> «0.0000».

    Знак — настоящий минус (U+2212), как в формулах задании: в формуле
    «2(w − 3)» уже стоит U+2212, и смешивать его с дефисом нельзя.
    """
    if abs(v) < 5e-9:
        v = 0.0
    return f"{v:.4f}".replace("-", MINUS)


def f2(v):
    if abs(v) < 5e-9:
        v = 0.0
    return f"{v:.2f}".replace("-", MINUS)


def fmt(v):
    """Число для полей ввода: 2.0 -> «2», 0.25 -> «0.25».

    Здесь остаётся обычный ASCII-дефис: это значение пользователь может
    напечатать в поле, и оно должно читаться обратно через float().
    """
    if abs(v) < 1e-9:
        return "0"
    if abs(v - round(v)) < 1e-10 and abs(v) < 1e15:
        return str(int(round(v)))
    return f"{v:.4g}"


def tick_text(v):
    """Подпись деления на осях графика: минус настоящий, число короткое."""
    return fmt(v).replace("-", MINUS)


def signed(v):
    """Отрицательное число в множителе берём в скобки: −(0.1000) · (−5.0000)."""
    return f"({f4(v)})" if v < -5e-9 else f4(v)


def arrow_for(delta):
    """Стрелка направления движения параметра по численной оси."""
    if delta > 1e-12:
        return "→"
    if delta < -1e-12:
        return "←"
    return "·"


# ---------------------------------------------------------------
# 6. Текст расчёта: явные подстановки, каждую строку можно проверить
# ---------------------------------------------------------------
def calc_block(descent):
    """Панель «Расчёт шага»: собранный шаг целиком, построчно и с подстановками.

    До первого нажатия показывается тот же самый расчёт — предстоящий шаг. Тогда
    кнопка «Следующий шаг» не сообщает ничего нового: числа уже на экране, и их
    можно проверить вручную до того, как они применятся.
    """
    ex = descent.example
    done = descent.last_step()
    s = done or descent.step_preview()
    head = T("lbl_step", n=s.index + 1)
    if done is None:
        head += "  (" + T("lbl_pending") + ")"
    loss_at = T("tpl_loss_at", w=f4(s.w), sub=ex["loss_sub"].format(v=f4(s.w)),
                v=f4(s.loss))
    grad_at = T("tpl_grad_at", w=f4(s.w), sub=ex["grad_sub"].format(v=f4(s.w)),
                v=f4(s.grad))
    after = ex["loss"](s.w_new)
    lines = [
        head, "",
        T("ln_param") + f"\n  {T('tpl_named', v=f4(s.w))}", "",
        T("ln_loss") + f"\n  {ex['formula']}\n  {loss_at}", "",
        T("ln_grad") + f"\n  {ex['deriv']}\n  {grad_at}", "",
        T("ln_eta") + f"\n  {T('tpl_eta', v=f4(s.eta))}", "",
        T("ln_update") + "\n  " + T("tpl_delta", e=f4(s.eta), g=signed(s.grad),
                                      v=f4(s.delta)), "",
        T("ln_new") + "\n  " + T("tpl_w_new", w=f4(s.w), e=f4(s.eta),
                                   g=signed(s.grad), v=f4(s.w_new)), "",
        T("ln_result") + "\n  " + T("tpl_after", v=f4(s.w_new), l=f4(after)),
        "  " + T("tpl_dist", w=f4(s.w_new), m=signed(descent.min_w),
                 v=f4(abs(s.w_new - descent.min_w))), "",
        T("status_hdr"), f"  {status_of(descent)}",
    ]
    return "\n".join(lines)


def status_of(descent):
    """Одна строка состояния: сошлось / к минимуму / от минимума."""
    if descent.converged():
        return T("st_converged")
    if descent.blown():
        return T("st_blown")
    s = descent.last_step()
    if s is None:
        return T("st_idle")
    return T("st_toward" if s.dist_new <= s.dist else "st_away")


def what_happened(descent):
    """Панель «Что произошло?» — объяснение именно этого шага."""
    s = descent.last_step()
    if s is None:
        return T("what_start")
    cmp_key = "cmp_same"
    if s.dist_new < s.dist - 1e-12:
        cmp_key = "cmp_closer"
    elif s.dist_new > s.dist + 1e-12:
        cmp_key = "cmp_farther"
    cmp_text = T(cmp_key, wn=f4(s.w_new), wo=f4(s.w),
                 lo=f4(s.loss), ln=f4(s.loss_new))
    if abs(s.grad) < TOL_GRAD:
        key = "what_zero"
    elif s.grad < 0:
        key = "what_neg"
    else:
        key = "what_pos"
    return T(key, g=f4(s.grad), m=f4(descent.min_w), cmp=cmp_text)


# ---------------------------------------------------------------
# 7. Поле ввода: число + ползунок
# ---------------------------------------------------------------
class NumberControl(ttk.Frame):
    """Одно число: подпись + поле ввода + ползунок. Поле и ползунок связаны.

    Печатаете в поле — ползунок едет следом; двигаете ползунок — число появляется
    в поле. Нечисловое поле ползунок не сдвигает: значение останется последним
    правильным, а поле подсветится красным.
    """

    def __init__(self, parent, label, lo, hi, on_change=None, digits=2):
        super().__init__(parent)
        self.lo, self.hi = lo, hi
        self.on_change = on_change
        self.digits = digits
        self.lbl_name = label
        self.var_text = tk.StringVar(value="0")
        self.var_scale = tk.DoubleVar(value=0.0)
        self._lock = False

        self.lbl = ttk.Label(self, text=label)
        self.lbl.pack(side="top", anchor="w")
        row = ttk.Frame(self)
        row.pack(fill="x")
        self.entry = ttk.Entry(row, width=9, textvariable=self.var_text, justify="right")
        self.entry.pack(side="left", padx=(0, 6))
        self.scale = ttk.Scale(row, from_=lo, to=hi, variable=self.var_scale,
                               command=self._from_scale)
        self.scale.pack(side="left", fill="x", expand=True)
        self.var_text.trace_add("write", self._from_entry)

    def set_label(self, text):
        self.lbl_name = text
        self.lbl.configure(text=text)

    def set_value(self, v):
        """Программная установка числа (пример, сброс) без вызова on_change."""
        self._lock = True
        self.var_text.set(fmt(v))
        self.var_scale.set(min(self.hi, max(self.lo, v)))
        self._lock = False

    def _from_entry(self, *_):
        if self._lock:
            return
        raw = self.var_text.get().strip().replace(",", ".")
        try:
            v = float(raw)
        except ValueError:
            if raw:
                self._notify(None, raw)
            return
        self._lock = True
        self.var_scale.set(min(self.hi, max(self.lo, v)))
        self._lock = False
        self._notify(v, raw)

    def _from_scale(self, *_):
        if self._lock:
            return
        value = round(self.var_scale.get(), 2)
        self._lock = True
        self.var_text.set(fmt(value))
        self._lock = False
        self._notify(value, fmt(value))

    def _notify(self, value, shown):
        if self.on_change:
            self.on_change(value, shown)


# ---------------------------------------------------------------
# 8. График на Canvas: он строится из формулы, а не из картинки
# ---------------------------------------------------------------
class GraphView(ttk.Frame):
    """Кривая L(w), минимум, траектория спуска и текущая точка с вертикальным guide."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.canvas = tk.Canvas(self, bg=COL_PAPER, highlightthickness=1,
                               highlightbackground=COL_FRAME, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    # ---------- границы рисунка ----------
    def _ranges(self, descent, ex):
        """Отображаемые интервалы по w и по L: чтобы кривая и весь путь влезали."""
        min_w = ex["min"]
        lo, hi = min_w - 1.6, min_w + 1.6
        lo = min(lo, descent.w0 - 0.8, descent.w)
        hi = max(hi, descent.w0 + 0.8, descent.w)
        for s in descent.steps:
            lo = min(lo, s.w, s.w_new)
            hi = max(hi, s.w, s.w_new)
        pad = (hi - lo) * 0.10
        lo, hi = lo - pad, hi + pad
        peak = max(ex["loss"](lo), ex["loss"](hi), 0.05)
        for w in descent.positions():
            peak = max(peak, ex["loss"](w))
        bottom = ex["loss"](min_w)
        top = peak + max(peak - bottom, 1.0) * 0.14
        base = bottom - (top - bottom) * 0.04
        return lo, hi, base, top

    def _ticks(self, lo, hi, want=6):
        """Красивые деления: шаг из {1, 2, 2.5, 5} · 10ⁿ."""
        raw = (hi - lo) / want
        if raw <= 0:
            return [lo]
        mag = 10.0 ** math.floor(math.log10(raw))
        step = mag
        for mult in (1, 2, 2.5, 5, 10):
            if mag * mult >= raw:
                step = mag * mult
                break
        first = math.ceil(lo / step) * step
        out = []
        v = first
        while v <= hi + step * 1e-6:
            out.append(round(v, 10))
            v += step
        return out

    def redraw(self):
        c = self.canvas
        c.delete("all")
        width = max(c.winfo_width(), 420)
        height = max(c.winfo_height(), 300)
        descent = self.app.descent
        ex = descent.example
        ml, mr, mb = 76, 28, 52               # поля под подписи осей
        title, formula = T("graph_title"), ex["formula"]
        ls = self.app.f_mid.metrics("linespace")
        # Верхнее поле по умолчанию вмещает ровно одну строку заголовка. Если обоим
        # текстам не хватает места хотя бы с одним пробелом между ними, формула
        # уходит на вторую строку, и поле растёт на межстрочный интервал шрифта.
        one_row = (self.app.f_mid.measure(title) + self.app.f_mid.measure(formula)
                   + ml + mr + self.app.f_mid.measure(" ") <= width)
        mt = 44 if one_row else 44 + ls
        plot_w = width - ml - mr
        plot_h = height - mt - mb
        w_lo, w_hi, l_lo, l_hi = self._ranges(descent, ex)

        def px(w):
            return ml + (w - w_lo) / (w_hi - w_lo) * plot_w

        def py(loss):
            return mt + (l_hi - loss) / (l_hi - l_lo) * plot_h

        # заголовок графика
        c.create_text(ml, 18, text=title, font=self.app.f_mid,
                      fill=COL_TEXT, anchor="w")
        c.create_text(width - mr, 18 if one_row else 18 + ls, text=formula,
                      font=self.app.f_mid, fill=COL_CURVE, anchor="e")

        # сетка и деления
        for v in self._ticks(l_lo, l_hi, 5):
            y = py(v)
            c.create_line(ml, y, width - mr, y, fill="#ededdf")
            c.create_text(ml - 10, y, text=tick_text(v), font=self.app.f_small,
                          fill=COL_LINE, anchor="e")
        for v in self._ticks(w_lo, w_hi, 7):
            x = px(v)
            c.create_line(x, mt, x, mt + plot_h, fill="#f2f2e6")
            c.create_text(x, mt + plot_h + 14, text=tick_text(v), font=self.app.f_small,
                          fill=COL_LINE)

        # оси
        c.create_line(ml, mt, ml, mt + plot_h, fill=COL_LINE, width=2)
        c.create_line(ml, mt + plot_h, width - mr, mt + plot_h, fill=COL_LINE, width=2)
        c.create_text(ml + 6, mt + plot_h + 30, text=T("axis_w"), font=self.app.f_midb,
                      fill=COL_TEXT, anchor="w")
        c.create_text(ml - 10, mt - 12, text=T("axis_loss"), font=self.app.f_midb,
                      fill=COL_TEXT, anchor="e")

        # кривая функции потерь
        n = 220
        pts = []
        for i in range(n + 1):
            w = w_lo + (w_hi - w_lo) * i / n
            pts.extend((px(w), py(ex["loss"](w))))
        c.create_line(pts, fill=COL_CURVE, width=3, smooth=True, capstyle="round")

        # минимум
        mx, my = px(ex["min"]), py(ex["loss"](ex["min"]))
        c.create_line(mx, mt, mx, mt + plot_h, fill=COL_MIN, dash=(3, 4))
        c.create_oval(mx - 7, my - 7, mx + 7, my + 7, fill=COL_MIN_FILL,
                      outline=COL_MIN, width=2)
        c.create_text(mx + 11, my - 16, text=f"{MIN_NAME} = {tick_text(ex['min'])}",
                      font=self.app.f_smallb, fill=COL_MIN, anchor="w")
        c.create_text(mx + 11, my + 2, text=T("lbl_minimum"), font=self.app.f_small,
                      fill=COL_MIN, anchor="w")

        # траектория: ломаная по точкам спуска
        pos = descent.positions()
        trail = []
        for w in pos:
            trail.extend((px(w), py(ex["loss"](w))))
        if len(trail) >= 4:
            c.create_line(trail, fill=COL_TRAIL, width=2, dash=(5, 3))
        for i, w in enumerate(pos):
            x, y = px(w), py(ex["loss"](w))
            if i == 0:
                c.create_oval(x - 5, y - 5, x + 5, y + 5, fill="white",
                              outline=COL_TRAIL, width=2)
                c.create_text(x - 9, y - 12, text=T("lbl_start"), font=self.app.f_small,
                              fill=COL_TRAIL, anchor="e")
            elif i == len(pos) - 1:
                continue
            else:
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill=COL_TRAIL_FILL,
                              outline=COL_TRAIL)

        # текущая точка + отвес к оси w
        cx, cy = px(descent.w), py(descent.loss)
        c.create_line(cx, cy, cx, mt + plot_h, fill=COL_LINE, dash=(2, 3))
        c.create_text(cx, mt + plot_h + 30, text=tick_text(descent.w), font=self.app.f_smallb,
                      fill=COL_POINT)
        pulse = self.app.pulse
        r = 11 if pulse else 9
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=COL_POINT_FILL,
                      outline=COL_ACTIVE if pulse else COL_POINT, width=3)
        c.create_text(cx + r + 8, cy - 12, text=f"w = {f4(descent.w)}",
                      font=self.app.f_smallb, fill=COL_POINT, anchor="w")
        c.create_text(cx + r + 8, cy + 6, text=f"L(w) = {f4(descent.loss)}",
                      font=self.app.f_small, fill=COL_POINT, anchor="w")

        # касательная: визуально показываем наклон = dL/dw
        self._tangent(descent, ex, px, py, plot_w, plot_h)

        # стрелка предстоящего шага + стрелки направления внизу
        self._direction(descent, ex, px, py, mt, plot_h)

    def _tangent(self, descent, ex, px, py, plot_w, plot_h):
        """Отрезок касательной в текущей точке: его наклон и есть dL/dw.

        Экранная ось Y смотрит вниз, поэтому экранный наклон берётся с обратным
        знаком относительно математического.
        """
        lo_w, hi_w, lo_l, hi_l = self._ranges(descent, ex)
        sx = plot_w / (hi_w - lo_w)          # пикселей на единицу w
        sy = plot_h / (hi_l - lo_l)          # пикселей на единицу L
        k = -descent.grad * sy / sx
        dx = 46.0 / math.sqrt(1 + k * k)
        cx, cy = px(descent.w), py(descent.loss)
        self.canvas.create_line(cx - dx, cy - k * dx, cx + dx, cy + k * dx,
                                fill=COL_ACTIVE, width=2, dash=(6, 3))

    def _direction(self, descent, ex, px, py, mt, plot_h):
        """Стрелки: куда указывает градиент и куда реально идёт обновление."""
        c = self.canvas
        if descent.stuck():
            return
        g = descent.grad
        eta = descent.eta
        nxt = descent.w - eta * g
        cx, cy = px(descent.w), py(descent.loss)
        nx, ny = px(nxt), py(ex["loss"](nxt))
        if abs(nx - cx) > 2 or abs(ny - cy) > 2:
            c.create_line(cx, cy, nx, ny, fill=COL_POINT, width=2, arrow="last",
                          arrowshape=(11, 13, 6))
            c.create_text((cx + nx) / 2, (cy + ny) / 2 - 14, text="w_new",
                          font=self.app.f_small, fill=COL_POINT)
        # две стрелки-пояснения в нижней части поля
        base_y = mt + plot_h - 26
        cx0 = px(descent.w)
        gdir = 1 if g > 0 else -1             # градиент показывает вверх по функции
        sx = abs(px(1.0) - px(0.0))           # пикселей на единицу w
        ulen = min(74.0, max(20.0, abs(eta * g) * sx))
        c.create_line(cx0 - gdir * ulen, base_y - 14, cx0 + gdir * ulen, base_y - 14,
                      fill=COL_CURVE, width=2, arrow="last", arrowshape=(9, 11, 5))
        c.create_text(cx0 + gdir * (ulen + 8), base_y - 14, text=T("gradient_dir"),
                      font=self.app.f_small, fill=COL_CURVE,
                      anchor="w" if gdir > 0 else "e")
        udir = -gdir                          # обновление идёт против градиента
        c.create_line(cx0 - udir * ulen, base_y + 8, cx0 + udir * ulen, base_y + 8,
                      fill=COL_POINT, width=2, arrow="last", arrowshape=(9, 11, 5))
        c.create_text(cx0 + udir * (ulen + 8), base_y + 8, text=T("update_dir"),
                      font=self.app.f_small, fill=COL_POINT,
                      anchor="w" if udir > 0 else "e")


# ---------------------------------------------------------------
# 8.5 Колонка, которую можно прокручивать по вертикали
# ---------------------------------------------------------------
class ScrollColumn(ttk.Frame):
    """Столбец оконной сетки с вертикальной прокруткой содержимого.

    В низком окне нижняя часть колонки не обрезается молча: появляется полоса
    прокрутки (и колесо мыши), и до последней строки можно доехать.
    """

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self._has_bar = False
        self.inner.bind("<Configure>", self._relayout, add="+")
        self.canvas.bind("<Configure>", self._relayout, add="+")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, self._on_wheel, add="+")

    def _relayout(self, *_):
        width = self.canvas.winfo_width()
        if width > 1:
            self.canvas.itemconfigure(self._win, width=width)
        region = self.canvas.bbox("all")
        self.canvas.configure(scrollregion=region or (0, 0, 1, 1))
        need = bool(region) and region[3] > self.canvas.winfo_height() + 1
        if need == self._has_bar:
            return
        self._has_bar = need
        if need:
            self.bar.pack(side="right", fill="y", before=self.canvas)
        else:
            self.bar.pack_forget()

    def _on_wheel(self, event):
        if not self._has_bar:
            return
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            widget = widget.master
        if widget is not self:
            return
        delta = getattr(event, "delta", 0) or (120 if getattr(event, "num", 0) == 4 else -120)
        self.canvas.yview_scroll(-1 if delta > 0 else 1, "units")


# ---------------------------------------------------------------
# 9. Окно приложения
# ---------------------------------------------------------------
class GradientDescentApp(tk.Tk):
    """Слева управление и состояние, в центре большой график, справа расчёт,
    история шагов и справка; внизу — «Что произошло?» и строка состояния."""

    AUTO_DELAY_MS = 700

    def __init__(self):
        super().__init__()
        self.title(T("app_title"))
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.minsize(MIN_W, MIN_H)
        self.geometry(f"{min(DEFAULT_W, max(MIN_W, screen_w - 60))}"
                      f"x{min(DEFAULT_H, max(MIN_H, screen_h - 60))}")

        # ---- состояние ----
        ex = DEFAULT_EXAMPLE
        self.example = ex
        self.descent = Descent(ex, ex["w0"], ex["eta"])
        self.pulse = False
        self._auto_job = None
        self._silent = False
        self._text_refs = []
        self._tabs = []

        self.var_lang = tk.StringVar(value=LANG)
        self.var_status = tk.StringVar()
        self.var_what = tk.StringVar()
        self.var_step = tk.StringVar()
        self.var_w = tk.StringVar()
        self.var_loss = tk.StringVar()
        self.var_grad = tk.StringVar()
        self.var_dist = tk.StringVar()
        self.var_state = tk.StringVar()
        self.var_auto = tk.StringVar(value=T("btn_auto"))
        self.var_fn = tk.StringVar(value=self.example["formula"])

        # ---- шрифты: 14–20 px для основного текста, крупнее для важных чисел ----
        base = tkfont.nametofont("TkDefaultFont").actual("family")
        self.f_title = tkfont.Font(family=base, size=17, weight="bold")
        self.f_sub = tkfont.Font(family=base, size=14)
        self.f_big = tkfont.Font(family=base, size=19, weight="bold")
        self.f_mid = tkfont.Font(family=base, size=14, weight="bold")
        self.f_midb = tkfont.Font(family=base, size=13, weight="bold")
        self.f_small = tkfont.Font(family=base, size=12)
        self.f_smallb = tkfont.Font(family=base, size=12, weight="bold")
        self.f_mono = tkfont.nametofont("TkFixedFont").copy()
        self.f_mono.configure(size=14)
        self.option_add("*Font", self.f_sub)

        self._build_header()
        self._build_left()
        self._build_center()
        self._build_right()
        self._build_status()

        self._fill_guide()
        self.render()
        self.set_status("status_ready")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- локализация ----------
    def _reg(self, widget, key, attr="text", **kw):
        """Надпись виджета берётся из TEXTS и переведётся при смене языка."""
        widget.configure(**{attr: T(key, **kw)})
        self._text_refs.append((widget, key, attr, kw))
        return widget

    def switch_language(self):
        code = self.var_lang.get()
        if code == LANG:
            return
        set_language(code)
        self.retranslate()
        self.set_status("status_lang")

    def retranslate(self):
        """Перевести статические надписи и заново нарисовать динамические части."""
        self.title(T("app_title"))
        for widget, key, attr, kw in self._text_refs:
            widget.configure(**{attr: T(key, **kw)})
        for tab, key in self._tabs:
            self.nb.tab(tab, text=T(key))
        self.ctrl_w0.set_label(T("lbl_w0"))
        self.ctrl_eta.set_label(T("lbl_eta"))
        self.var_auto.set(T("btn_auto") if self._auto_job is None else T("btn_stop"))
        self._set_headings()
        self._fill_guide()
        self.render()

    def set_status(self, key, **kw):
        self.var_status.set(T(key, **kw))

    # ---------- построение интерфейса ----------
    def _build_header(self):
        head = ttk.Frame(self)
        head.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(12, 2))
        self._reg(ttk.Label(head, font=self.f_title), "app_title").pack(side="left")
        self._reg(ttk.Label(head, font=self.f_sub, foreground="#4a4a4a"),
                  "subtitle").pack(side="left", padx=(18, 0))
        box = ttk.Frame(head)
        box.pack(side="right")
        self._reg(ttk.Label(box, font=self.f_sub), "lang_label").pack(side="left")
        for code, key in (("ru", "lang_ru"), ("en", "lang_en")):
            ttk.Radiobutton(box, text=T(key), value=code, variable=self.var_lang,
                            command=self.switch_language).pack(side="left", padx=(8, 0))
        self.columnconfigure(0, minsize=300, weight=0)
        self.columnconfigure(1, minsize=490, weight=1)
        self.columnconfigure(2, minsize=400, weight=0)
        self.rowconfigure(1, weight=1)

    def _build_left(self):
        column = ScrollColumn(self)
        column.grid(row=1, column=0, sticky="nsew")
        left = ttk.Frame(column.inner, padding=(12, 4))
        left.pack(fill="x")

        fn_box = ttk.LabelFrame(left, padding=8)
        self._reg(fn_box, "box_function")
        fn_box.pack(fill="x", pady=(0, 8))
        ttk.Label(fn_box, textvariable=self.var_fn, font=self.f_big,
                  foreground=COL_CURVE).pack(anchor="w")
        self._reg(ttk.Label(fn_box, font=self.f_small, foreground="#4a4a4a"),
                  "fn_deriv", deriv=self.example["deriv"]).pack(anchor="w", pady=(6, 0))
        self._reg(ttk.Label(fn_box, font=self.f_smallb, foreground=COL_MIN),
                  "fn_min", m=f4(self.example["min"])).pack(anchor="w", pady=(2, 0))

        ctrl_box = ttk.LabelFrame(left, padding=8)
        self._reg(ctrl_box, "box_controls")
        ctrl_box.pack(fill="x", pady=(0, 8))
        self.ctrl_w0 = NumberControl(ctrl_box, T("lbl_w0"),
                                     self.example["min"] - W_SPAN,
                                     self.example["min"] + W_SPAN,
                                     on_change=lambda v, raw: self.set_start("w0", v, raw))
        self.ctrl_w0.pack(fill="x", pady=(0, 8))
        self.ctrl_eta = NumberControl(ctrl_box, T("lbl_eta"), ETA_MIN, ETA_MAX,
                                      on_change=lambda v, raw: self.set_start("eta", v, raw))
        self.ctrl_eta.pack(fill="x")
        self.ctrl_w0.set_value(self.descent.w0)
        self.ctrl_eta.set_value(self.descent.eta)

        state_box = ttk.LabelFrame(left, padding=8)
        state_box.pack(fill="x", pady=(0, 8))
        state_box.columnconfigure(1, weight=1, minsize=118)
        rows = ("var_step", "var_w", "var_loss", "var_grad", "var_dist")
        for i, key in enumerate(rows):
            name_key = {"var_step": "head_step", "var_w": "head_w",
                        "var_loss": "head_loss", "var_grad": "head_grad",
                        "var_dist": "lbl_dist"}[key]
            if name_key == "lbl_dist":
                ttk.Label(state_box, text="|w − w*|", font=self.f_smallb).grid(
                    row=i, column=0, sticky="w")
            else:
                self._reg(ttk.Label(state_box, font=self.f_smallb), name_key).grid(
                    row=i, column=0, sticky="w")
            big = key in ("var_w", "var_loss")
            ttk.Label(state_box, textvariable=getattr(self, key), font=(
                self.f_mid if big else self.f_sub), foreground=COL_TEXT,
                justify="right").grid(row=i, column=1, sticky="e", pady=1)
        ttk.Separator(state_box, orient="horizontal").grid(row=5, column=0, columnspan=2,
                                                           sticky="ew", pady=6)
        self._reg(ttk.Label(state_box, font=self.f_smallb, foreground=COL_POINT),
                  "status_hdr").grid(row=6, column=0, columnspan=2, sticky="w")
        state_lbl = ttk.Label(state_box, textvariable=self.var_state, font=self.f_small,
                              foreground=COL_POINT, justify="left")
        state_lbl.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        _auto_wrap(state_lbl, min_width=200)

    def _build_center(self):
        center = ttk.Frame(self)
        center.grid(row=1, column=1, sticky="nsew", padx=(0, 6), pady=4)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        self.view = GraphView(center, self)
        self.view.grid(row=1, column=0, sticky="nsew")

        buttons = ttk.Frame(center)
        buttons.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.btn_step = self._reg(ttk.Button(buttons, command=self.next_step,
                                             padding=(14, 6)), "btn_step")
        self.btn_step.pack(side="left")
        self.btn_auto = ttk.Button(buttons, textvariable=self.var_auto,
                                   command=self.toggle_auto, padding=(12, 6))
        self.btn_auto.pack(side="left", padx=(8, 0))
        self._reg(ttk.Button(buttons, command=self.reset_all,
                             padding=(12, 6)), "btn_reset").pack(side="left", padx=(8, 0))

        ex_box = ttk.LabelFrame(center, padding=8)
        self._reg(ex_box, "box_examples")
        ex_box.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for i, key in enumerate(EXAMPLE_KEYS):
            self._reg(ttk.Button(ex_box, command=lambda k=key: self.load_example(k)),
                      f"btn_{key}").pack(side="left", padx=(0 if i == 0 else 6, 0))
        self._reg(ttk.Button(ex_box, command=self.load_big_eta),
                  "btn_big_eta").pack(side="left", padx=(10, 0))

    def _build_right(self):
        self.nb = ttk.Notebook(self)
        # rowspan=2: панель вкладок занимает и строку пояса «Что происходит»,
        # иначе последних строк блока расчёта не хватает по высоте.
        self.nb.grid(row=1, column=2, rowspan=2, sticky="nsew", padx=(0, 12), pady=4)

        calc_frame = ttk.Frame(self.nb, padding=6)
        self.txt_calc = self._text_panel(calc_frame, width=30)
        self.nb.add(calc_frame, text=T("tab_calc"))
        self._tabs.append((calc_frame, "tab_calc"))

        hist_frame = ttk.Frame(self.nb, padding=6)
        self.tree = ttk.Treeview(hist_frame, columns=("w", "loss", "grad", "delta"),
                                 show="tree headings", selectmode="browse", height=10)
        note = self._reg(ttk.Label(hist_frame, font=self.f_small, foreground="#4a4a4a",
                                   justify="left", anchor="w"), "history_note")
        _auto_wrap(note, min_width=330)
        self.var_hist = tk.StringVar()
        count = ttk.Label(hist_frame, textvariable=self.var_hist, font=self.f_smallb,
                          foreground=COL_TRAIL)
        bar = ttk.Scrollbar(hist_frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        note.pack(side="bottom", fill="x", pady=(8, 2))
        count.pack(side="bottom", anchor="w")
        self.tree.pack(side="top", fill="both", expand=True)
        self.nb.add(hist_frame, text=T("tab_history"))
        self._tabs.append((hist_frame, "tab_history"))

        guide_frame = ttk.Frame(self.nb, padding=6)
        self.txt_guide = self._text_panel(guide_frame, width=30)
        self.nb.add(guide_frame, text=T("tab_guide"))
        self._tabs.append((guide_frame, "tab_guide"))

        self._set_headings()

        what = ttk.LabelFrame(self, padding=10)
        self._reg(what, "what_title")
        what.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        what_label = ttk.Label(what, textvariable=self.var_what, font=self.f_sub,
                               justify="left")
        what_label.pack(fill="x")
        _auto_wrap(what_label, min_width=320)

    def _set_headings(self):
        self.tree.heading("#0", text=T("head_step"), anchor="center")
        for col, key in (("w", "head_w"), ("loss", "head_loss"),
                         ("grad", "head_grad"), ("delta", "head_delta")):
            self.tree.heading(col, text=T(key), anchor="e")
        self.tree.column("#0", width=46, minwidth=40, anchor="center", stretch=False)
        for col in ("w", "loss", "grad", "delta"):
            self.tree.column(col, width=78, minwidth=66, anchor="e", stretch=True)

    def _text_panel(self, parent, width):
        """Текстовая панель вкладки: перенос по словам, чтобы ни один результат
        не уезжал за правый край (горизонтальная прокрутка прятала бы числа)."""
        widget = tk.Text(parent, font=self.f_mono, bg=COL_PAPER, fg=COL_TEXT, wrap="word",
                         width=width, height=14, relief="flat", padx=12, pady=10,
                         highlightthickness=1, highlightbackground=COL_FRAME,
                         state="disabled")
        bar = ttk.Scrollbar(parent, command=widget.yview)
        widget.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        widget.pack(side="left", fill="both", expand=True)
        return widget

    def _build_status(self):
        bar = ttk.Frame(self)
        bar.grid(row=3, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 10))
        ttk.Label(bar, textvariable=self.var_status, font=self.f_small).pack(side="left")
        # Имя автора: place прямо по окну, а не pack в бар — полоса состояния бывает
        # шире окна, и тогда метка ушла бы за его правый край.
        ttk.Label(self, text="SteveWorks001", font=self.f_small,
                  foreground="#8a8a82").place(relx=1.0, rely=1.0, anchor="se",
                                              x=-12, y=-10)

    # ---------- числа: начальный w и скорость обучения ----------
    def set_start(self, which, value, shown):
        """Поле или ползунок изменили: проверяем число, затем начинаем спуск заново."""
        if self._silent:
            return
        field = T("lbl_w0") if which == "w0" else T("lbl_eta")
        if value is None:
            self.set_status("status_err", field=field, raw=shown)
            return
        if which == "w0":
            lo = self.example["min"] - W_SPAN
            hi = self.example["min"] + W_SPAN
            if not (lo <= value <= hi):
                self.set_status("status_range", field=field, raw=f"{fmt(value)} [{lo}; {hi}]")
                return
            self.descent.restart(w=value)
        else:
            if not (ETA_MIN <= value <= ETA_MAX):
                self.set_status("status_range", field=field,
                                raw=f"{fmt(value)} [{ETA_MIN}; {ETA_MAX}]")
                return
            self.descent.restart(eta=value)
        self.stop_auto()
        self.render()

    # ---------- режимы ----------
    def next_step(self):
        """Ровно одна итерация градиентного спуска."""
        d = self.descent
        if d.converged():
            self._say_converged()
            return False
        if d.blown():
            self.set_status("status_blown", w=f4(d.w), m=f4(d.min_w), dist=f4(MAX_DIST))
            return False
        if len(d.steps) >= MAX_STEPS:
            self.set_status("status_max", n=MAX_STEPS)
            return False
        s = d.step()
        self._blink()
        self.render()
        self.set_status("status_step", n=s.index + 1, w=f4(s.w_new), loss=f4(s.loss_new),
                        dir=T("dir_down" if s.dist_new <= s.dist else "dir_up"))
        if d.converged():
            self._say_converged()
        elif d.blown():
            self.set_status("status_blown", w=f4(d.w), m=f4(d.min_w), dist=f4(MAX_DIST))
        return True

    def _say_converged(self):
        d = self.descent
        self.set_status("status_converged", dist=f4(abs(d.w - d.min_w)),
                        grad=f4(abs(d.grad)), tol_d=f4(TOL_DIST), tol_g=f4(TOL_GRAD))

    def toggle_auto(self):
        if self._auto_job is None:
            self.start_auto()
        else:
            self.stop_auto()

    def start_auto(self):
        self.stop_auto()
        self.var_auto.set(T("btn_stop"))
        self.set_status("status_auto", ms=self.AUTO_DELAY_MS)
        self._auto_job = self.after(self.AUTO_DELAY_MS, self._auto_tick)

    def _auto_tick(self):
        self._auto_job = None
        if not self.next_step():
            self.var_auto.set(T("btn_auto"))
            return
        self._auto_job = self.after(self.AUTO_DELAY_MS, self._auto_tick)
        if self._auto_job is None:
            self.var_auto.set(T("btn_auto"))

    def stop_auto(self):
        if self._auto_job is not None:
            self.after_cancel(self._auto_job)
            self._auto_job = None
        self.var_auto.set(T("btn_auto"))

    def reset_all(self):
        """Вернуть всё к исходному состоянию выбранного примера."""
        self.stop_auto()
        ex = self.example
        self.descent.restart(w=ex["w0"], eta=ex["eta"])
        self._silent = True
        self.ctrl_w0.set_value(ex["w0"])
        self.ctrl_eta.set_value(ex["eta"])
        self._silent = False
        self.pulse = False
        self.render()
        self.set_status("status_reset")

    def _pick_example(self, key):
        for ex in EXAMPLES:
            if ex["id"] == key:
                return ex
        return DEFAULT_EXAMPLE

    def load_example(self, key):
        self.stop_auto()
        ex = self._pick_example(key)
        self.example = ex
        self.descent = Descent(ex, ex["w0"], ex["eta"])
        lo = ex["min"] - W_SPAN
        hi = ex["min"] + W_SPAN
        self._silent = True
        self.ctrl_w0.lo, self.ctrl_w0.hi = lo, hi
        self.ctrl_w0.scale.configure(from_=lo, to=hi)
        self.ctrl_w0.set_value(ex["w0"])
        self.ctrl_eta.set_value(ex["eta"])
        self._silent = False
        self._refresh_function_labels()
        self.render()
        self.set_status("status_example", name=T(f"{key}_name"), desc=T(f"{key}_desc"))

    def load_big_eta(self):
        """Демонстрация неустойчивости: основной пример и η за пределом устойчивости."""
        self.load_example(DEFAULT_EXAMPLE["id"])
        self.descent.restart(eta=BIG_ETA)
        self._silent = True
        self.ctrl_eta.set_value(BIG_ETA)
        self._silent = False
        self.render()
        self.set_status("status_big_eta", eta=f4(BIG_ETA))

    def _refresh_function_labels(self):
        """Подписи над функцией: формула одинакова на обоих языках, остальное — из TEXTS."""
        ex = self.example
        self.var_fn.set(ex["formula"])
        for widget, key, attr, kw in self._text_refs:
            if key in ("fn_deriv", "fn_min"):
                new_kw = dict(kw)
                if key == "fn_deriv":
                    new_kw["deriv"] = ex["deriv"]
                else:
                    new_kw["m"] = f4(ex["min"])
                widget.configure(**{attr: T(key, **new_kw)})
                kw.clear()
                kw.update(new_kw)

    # ---------- вывод панелей ----------
    def render(self):
        self.view.redraw()
        d = self.descent
        s = d.last_step()
        self.var_step.set(str(len(d.steps)))
        self.var_w.set(f4(d.w))
        self.var_loss.set(f4(d.loss))
        self.var_grad.set(f4(d.grad))
        self.var_dist.set(f4(abs(d.w - d.min_w)))
        self.var_state.set(status_of(d))
        self._set_text(self.txt_calc, calc_block(d))
        self._fill_history()
        self.var_what.set(what_happened(d))

    def _fill_history(self):
        d = self.descent
        self.tree.delete(*self.tree.get_children())
        rows = [(s.index, f4(s.w), f4(s.loss), f4(s.grad), f4(s.delta), "")
                for s in d.steps]
        # Последняя строка — предстоящий шаг: тот же расчёт, что покажет кнопка.
        if not d.stuck():
            s = d.step_preview()
            rows.append((s.index, f4(s.w), f4(s.loss), f4(s.grad), f4(s.delta),
                         "preview"))
        for idx, w, loss, grad, delta, tag in rows:
            self.tree.insert("", "end", text=str(idx), values=(w, loss, grad, delta),
                             tags=(tag,))
        self.tree.tag_configure("preview", foreground=COL_TRAIL)
        self.var_hist.set(T("history_rows", n=len(d.steps)) if d.steps
                          else T("history_empty"))

    def _blink(self):
        """Короткая подсветка текущей точки: 260 мс, без лишней мишуры."""
        self.pulse = True
        self.view.redraw()
        self.after(260, self._unpulse)

    def _unpulse(self):
        self.pulse = False
        self.view.redraw()

    def _set_text(self, widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _fill_guide(self):
        self._set_text(self.txt_guide, guide_text())

    def _on_close(self):
        self.stop_auto()
        self.destroy()


def _auto_wrap(label, min_width=280):
    """Переносить текст метки по её собственной фактической ширине.

    Метку ставят с fill="x" или sticky="ew": тогда её ширина равна месту родителя,
    а запрос никогда не больше этой ширины — оконная сетка не разрастается.
    Начальное wraplength нужно, чтобы до первого <Configure> метка не просила
    ширину всей строки сразу.
    """
    label.configure(wraplength=min_width)

    def on_configure(event):
        label.configure(wraplength=max(min_width, event.width))
    label.bind("<Configure>", on_configure, add="+")


def main():
    GradientDescentApp().mainloop()


if __name__ == "__main__":
    main()
