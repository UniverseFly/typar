# typar

A fully-typed combinatorial parsing library in Python.

## Installation

```shell
pip install typar
```

## Example

Execute `python -m typar.example` to run the example below:

```python
import typar

year = typar.regex("[0-9]{4}")
month = typar.regex("[0-9]{2}")
day = typar.regex("[0-9]{2}")
dash = typar.string("-")
fulldate = year + dash + month + dash + day << typar.eof()
print(fulldate.parse("2019-01-01"))
# Result(index=10, value=(((('2019', '-'), '01'), '-'), '01'))
print(fulldate.parse("2019-01-01x"))
# Result(index=0, value=None)
```
