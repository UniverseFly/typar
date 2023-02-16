import abc
import re
import typing
from dataclasses import dataclass, field
from typing import Callable, Generic, Sequence, TypeVar, cast, no_type_check

Item = TypeVar("Item")

T = TypeVar("T")
U = TypeVar("U")

Stream = Sequence


@dataclass(frozen=True)
class Result(Generic[T]):
    """None if the parser failed, otherwise the value and the index of the next token"""

    index: int
    value: T | None

    @staticmethod
    def ok(index: int, value: T) -> "Result[T]":
        return Result(index, value)

    @staticmethod
    def err(index: int) -> "Result[typing.Any]":
        return Result(index, None)


class Parser(Generic[Item, T]):
    @abc.abstractmethod
    def __call__(self, stream: Stream[Item], index: int) -> Result[T]:
        ...

    def parse(self, stream: Stream[Item]) -> Result[T]:
        return self(stream, 0)

    def bind(self, bind_fn: "Callable[[T], Parser[Item, U]]") -> "Parser[Item, U]":
        return Bind(self, bind_fn)

    def map(self, map_fn: Callable[[T], U]) -> "Parser[Item, U]":
        return Map(self, map_fn)

    def optional(self, default_value: U) -> "Parser[Item, T | U]":
        return self | ret(default_value)

    def many(self) -> "Parser[Item, list[T]]":
        return Times(0, None, self)

    def many1(self) -> "Parser[Item, list[T]]":
        return Times(1, None, self)

    def __add__(self, other: "Parser[Item, U]") -> "Parser[Item, tuple[T, U]]":
        return Add(self, other)

    def __mul__(self, n: int) -> "Parser[Item, list[T]]":
        return seq(*(self for _ in range(n)))

    def __or__(self, other: "Parser[Item, U]") -> "Parser[Item, T | U]":
        return Or[Item, T, U](self, other)

    def __rshift__(self, other: "Parser[Item, U]") -> "Parser[Item, U]":
        return self.bind(lambda _: other)

    def __lshift__(self, other: "Parser[Item, U]") -> "Parser[Item, T]":
        return (self + other).map(lambda tp: tp[0])

    def tag(self, name: U) -> "Parser[Item, tuple[U, T]]":
        return self.map(lambda result: (name, result))


def ret(value: T) -> Parser[Item, T]:
    return Ret(value)


def fail() -> Parser[Item, typing.Any]:
    return Fail()


def satisfy(test_fn: Callable[[Item], bool]) -> Parser[Item, Item]:
    return Satisfy(test_fn)


def any_token() -> Parser[Item, Item]:
    return satisfy(lambda _: True)


def char(item: Item) -> Parser[Item, Item]:
    return satisfy(lambda i: i == item)


def regex(exp: str) -> Parser[str, str]:
    return Regex(exp)


def string(s: str) -> Parser[str, str]:
    return String(s)


def none_of(rejected: Sequence[Item]) -> Parser[Item, Item]:
    return satisfy(lambda tok: tok not in rejected)


def one_of(accepted: Sequence[Item]) -> Parser[Item, Item]:
    return satisfy(lambda tok: tok in accepted)


# Unit for a default return value while None indicating failure
Unit = tuple[()]


def not_followed_by(parser: Parser[Item, T]) -> Parser[Item, Unit]:
    return Not(parser)


def eof() -> Parser[Item, Unit]:
    return not_followed_by(any_token())


def seq(*parsers: Parser[Item, T]) -> Parser[Item, list[T]]:
    return Seq(parsers)


def any(*parsers: Parser[Item, T]) -> Parser[Item, T]:
    return Any(parsers)


@dataclass(frozen=True)
class Times(Generic[Item, T], Parser[Item, list[T]]):
    min_times: int
    max_times: int | None
    parser: Parser[Item, T]

    def __call__(self, stream: Stream[Item], index: int) -> Result[list[T]]:
        times = 0
        values: list[T] = []
        while times < self.max_times if self.max_times is not None else True:
            result = self.parser(stream, index)
            if result.value is not None:
                values.append(result.value)
                index = result.index
                times += 1
            elif times >= self.min_times:
                break
            else:
                return Result.err(index)
        return Result.ok(index, values)


@dataclass(frozen=True)
class Seq(Generic[Item, T], Parser[Item, list[T]]):
    # NOTE (improvement): in Python 3.11, we can specify variadic generic types
    parsers: Sequence[Parser[Item, T]]

    def __call__(self, stream: Stream[Item], index: int) -> Result[list[T]]:
        values: list[T] = []
        for parser in self.parsers:
            result = parser(stream, index)
            if result.value is None:
                return Result.err(result.index)
            index = result.index
            values.append(result.value)
        return Result.ok(index, values)


@dataclass(frozen=True)
class Any(Generic[Item, T], Parser[Item, T]):
    parsers: Sequence[Parser[Item, T]]

    def __call__(self, stream: Stream[Item], index: int) -> Result[T]:
        for parser in self.parsers:
            result = parser(stream, index)
            if result.value is not None:
                return result
        return Result.err(index)


@dataclass(frozen=True)
class Not(Generic[Item, T], Parser[Item, Unit]):
    parser: Parser[Item, T]

    def __call__(self, stream: Stream[Item], index: int) -> Result[Unit]:
        result = self.parser(stream, index)
        return Result.err(index) if result.value is not None else Result.ok(index, ())


@dataclass(frozen=True)
class Satisfy(Generic[Item], Parser[Item, Item]):
    test_fn: Callable[[Item], bool]

    def __call__(self, stream: Stream[Item], index: int) -> Result[Item]:
        if index >= len(stream):
            return Result.err(index)
        item = stream[index]
        if self.test_fn(item):
            return Result.ok(index + 1, item)
        else:
            return Result.err(index)


@dataclass(frozen=True)
class Ret(Generic[Item, T], Parser[Item, T]):
    value: T

    def __call__(self, stream: Stream[Item], index: int) -> Result[T]:
        return Result.ok(index, self.value)


@dataclass(frozen=True)
class Fail(Generic[Item], Parser[Item, typing.Any]):
    def __call__(self, stream: Stream[Item], index: int) -> Result[typing.Any]:
        return Result.err(index)


@dataclass(frozen=True)
class Map(Generic[Item, T, U], Parser[Item, U]):
    parser: Parser[Item, T]
    map_fn: Callable[[T], U]

    def __call__(self, stream: Stream[Item], index: int) -> Result[U]:
        result = self.parser(stream, index)
        if result.value is None:
            return Result.err(index)
        else:
            return Result.ok(result.index, self.map_fn(result.value))


@dataclass(frozen=True)
class Bind(Generic[Item, T, U], Parser[Item, U]):
    parser: Parser[Item, T]
    bind_fn: Callable[[T], Parser[Item, U]]

    def __call__(self, stream: Stream[Item], index: int) -> Result[U]:
        result = self.parser(stream, index)
        if result.value is None:
            return Result.err(index)
        next_parser = self.bind_fn(result.value)
        return next_parser(stream, result.index)


@dataclass(frozen=True)
class Add(Generic[Item, T, U], Parser[Item, tuple[T, U]]):
    lhs: Parser[Item, T]
    rhs: Parser[Item, U]

    def __call__(self, stream: Stream[Item], index: int) -> Result[tuple[T, U]]:
        lhs_result = self.lhs(stream, index)
        if lhs_result.value is None:
            return Result.err(index)
        else:
            value = lhs_result.value
            rhs_result = self.rhs(stream, lhs_result.index)
            if rhs_result.value is None:
                return Result.err(lhs_result.index)
            else:
                return Result.ok(
                    rhs_result.index,
                    (value, rhs_result.value),
                )


@dataclass(frozen=True)
class Or(Generic[Item, T, U], Parser[Item, T | U]):
    lhs: Parser[Item, T]
    rhs: Parser[Item, U]

    def __call__(self, stream: Stream[Item], index: int) -> Result[T | U]:
        lhs = self.lhs(stream, index)
        if lhs.value is not None:
            return cast(Result[T | U], lhs)
        return cast(Result[T | U], self.rhs(stream, index))


@dataclass
class Regex(Parser[str, str]):
    re_expr: str
    pattern: re.Pattern[str] = field(init=False)

    def __post_init__(self):
        self.pattern = re.compile(self.re_expr)

    def __call__(self, stream: Sequence[str], index: int) -> Result[str]:
        if not isinstance(stream, str):
            raise ValueError("Can only be used with `str` stream")
        match = self.pattern.match(stream, index)
        return (
            Result.ok(match.end(), match.group(0))
            if match is not None
            else Result.err(index)
        )


@dataclass(frozen=True)
class String(Parser[str, str]):
    string: str

    def __call__(self, stream: Sequence[str], index: int) -> Result[str]:
        if not isinstance(stream, str):
            raise ValueError("Can only be used with `str` stream")
        if stream[index : index + len(self.string)] == self.string:
            return Result.ok(index + len(self.string), self.string)
        else:
            return Result.err(index)


@dataclass(frozen=True)
class Decl(Generic[Item, T], Parser[Item, T]):
    def __call__(self, stream: Stream[Item], index: int) -> Result[T]:
        raise NotImplementedError

    @no_type_check
    def become(self, other: Parser[Item, T]):
        self.__dict__ = other.__dict__
        self.__class__ = other.__class__


@dataclass(frozen=True)
class AnyParser(Generic[Item, T], Parser[Item, T]):
    fn: Callable[[Stream[Item], int], Result[T]]

    def __call__(self, stream: Stream[Item], index: int) -> Result[T]:
        return self.fn(stream, index)


# # Recovery
# def expect(parser: Parser[Item, T], recovered: U) -> Parser[Item, T | U]:
#     @(AnyParser[Item, T | U])
#     def expected(stream: Stream[Item], index: int) -> Result[T | U]:
#         result = cast(Result[T | U], parser(stream, index))
#         if result.value is None:
#             return Result.ok(index, recovered)
#         else:
#             return result
#     return expected


# @dataclass(frozen=True)
# class Unexpected:
#     ...


# UNEXPECTED = Unexpected()


# def expect_default(parser: Parser[Item, T]) -> Parser[Item, T | Unexpected]:
#     return cast(Parser[Item, T | Unexpected], expect(parser, UNEXPECTED))
