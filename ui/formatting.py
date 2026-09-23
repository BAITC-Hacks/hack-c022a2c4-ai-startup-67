"""Единый визуальный язык приложения; только наблюдаемые данные."""
import math

ROLE_COLORS = {
    'consolidator': '#F6A04D', 'transit': '#55A7F7', 'distributor': '#AD8BF5',
    'terminal': '#8C9AAB', 'coordinator': '#F07178', 'peripheral': '#CDD5DF',
}
ROLE_LABELS = {
    'consolidator': 'Сбор средств', 'transit': 'Наблюдаемый транзит',
    'distributor': 'Распределение', 'terminal': 'Нет видимого оттока',
    'coordinator': 'Структурная связность', 'peripheral': 'Нет сильного паттерна',
}
LIMITATIONS = [
    'Наблюдается только июль 2026 года и переводы от 5 000 KZT.',
    'Граф собран обходом исходящих переводов seed-клиентов, до 4 переходов.',
    'По условиям набора наблюдаются только внутрибанковские переводы.',
    'Входящая активность seed и исходящие переводы узлов глубины 4 могут быть неполными.',
]


def numeric(value, digits=0):
    try:
        value = float(value)
        if not math.isfinite(value):
            return '—'
        return f'{value:,.{digits}f}'.replace(',', ' ')
    except (ValueError, TypeError):
        return '—'


def kzt(value):
    formatted = numeric(value, 2)
    return formatted + ' KZT' if formatted != '—' else '—'
