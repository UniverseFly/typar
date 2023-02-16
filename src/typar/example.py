import typar

if __name__ == "__main__":
    year = typar.regex("[0-9]{4}")
    month = typar.regex("[0-9]{2}")
    day = typar.regex("[0-9]{2}")
    dash = typar.string("-")
    fulldate = year + dash + month + dash + day << typar.eof()
    print(fulldate.parse("2019-01-01"))
    print(fulldate.parse("2019-01-01x"))
