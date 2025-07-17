from django import template
from django.utils.safestring import mark_safe
import json

register = template.Library()

@register.filter
def zip(a, b):
    return zip(a, b)

@register.filter
def get_item(lst, i):
    try:
        return lst[i]
    except:
        return None

@register.filter
def subtract(value, arg):
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def percentage_change(new_value, old_value):
    try:
        new_value = float(new_value)
        old_value = float(old_value)
        if old_value == 0:
            return 0
        return ((new_value - old_value) / old_value) * 100
    except (ValueError, TypeError):
        return 0