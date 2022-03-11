from functools import wraps
import json
from sympy import Symbol
from pytket import Circuit
from pytket.passes import RemoveRedundancies
from pyrs import remove_redundancies

def ser_wrapper(f):
    @wraps(f)
    def wrapped(c: Circuit) -> Circuit:
        return Circuit.from_dict(json.loads(f(json.dumps(c.to_dict()))))
    
    return wrapped

rust_remove_redundancies = ser_wrapper(remove_redundancies)
a = Symbol("a")
c = Circuit(2).Rz(a,0).Rz(-a,0).CX(0,1).CX(0,1).Rx(2, 1)

cd = json.dumps(c.to_dict())
def main_just_rs():
    remove_redundancies(cd)

def main():

    RemoveRedundancies().apply(c)

    c2 = rust_remove_redundancies(c)


    assert c == c2


    print(f"{c2.phase=}")
    print(c2.get_commands())

if __name__ == "__main__":
    main()