"""Расчёты для дизайна и анализа пилота (семинары С7 и С8)."""

import math
from statistics import NormalDist


def n_per_group(
    sigma: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> int:
    """Число клиентов на каждую группу; mde — абсолютный эффект.

    Для нескольких сравнений применяется поправка Бонферрони.
    """
    z = NormalDist().inv_cdf
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2)


def mde_for_n(
    sigma: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> float:
    """Минимальный абсолютный эффект при n клиентах в каждой группе."""
    z = NormalDist().inv_cdf
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return float((za + zb) * sigma * math.sqrt(2 / n))


def diff_ci(t, c, alpha: float = 0.05) -> dict:
    """Разность test − control, нормальный интервал и двустороннее p.

    t и c — pandas.Series; дисперсии оцениваются с ddof=1.
    Ключи результата: diff, lo, hi, p — как в ноутбуке ДЗ-3.
    """
    d = float(t.mean() - c.mean())
    se = math.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    q = NormalDist().inv_cdf(1 - alpha / 2)
    if se == 0:
        p = 1.0 if d == 0 else 0.0
    else:
        # erfc даёт двустороннее нормальное p без потери точности в хвостах.
        p = math.erfc(abs(d / se) / math.sqrt(2))
    return {"diff": d, "lo": d - q * se, "hi": d + q * se, "p": p}


def srm_pvalue(sizes) -> float:
    """Проверка SRM: равные ожидаемые доли для двух или трёх групп."""
    sizes = list(sizes)
    if len(sizes) not in (2, 3):
        raise ValueError("Нужны 2 или 3 группы; для остальных используйте scipy.stats.chisquare")
    n = sum(sizes)
    if n <= 0 or any(s < 0 for s in sizes):
        raise ValueError("Размеры групп должны быть неотрицательными с положительной суммой")
    expected = n / len(sizes)
    chi2 = sum((s - expected) ** 2 / expected for s in sizes)
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))
    return math.exp(-chi2 / 2)


def cuped(y, x):
    """Вернуть (метрика после CUPED, theta).

    y и x — pandas.Series с одинаковыми индексами; x измерена до пилота.
    """
    theta = y.cov(x) / x.var(ddof=1)
    return y - theta * (x - x.mean()), float(theta)
