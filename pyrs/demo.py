# from sympy import Symbol
from pytket import Circuit
from pytket.passes import RemoveRedundancies
from pyrs import remove_redundancies


# a = Symbol("a")
c = Circuit(2).CX(0, 1).CX(0, 1).Rx(2, 1)


def main_just_rs():
    c2 = remove_redundancies(c)
    print(repr(c2))


def main():

    RemoveRedundancies().apply(c)

    c2 = remove_redundancies(c)

    assert c == c2

    print(f"{c2.phase=}")
    print(c2.get_commands())


if __name__ == "__main__":
    main_just_rs()
