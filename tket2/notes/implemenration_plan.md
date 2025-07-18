All the basic quantum operations are wrapped in ExtensionOp.

We need to solve fixed point problem:
    - which modified function is needed?
    - Don't want to generate a circuit until we solve this fixed point
        1. Collect constraints: how do you modify function calls?
        2. Solve constraints: to know something like "we need CCC-U but not CC-U, so we make CCC-U".
        3. I thought this could be done at the same time as DeadCodeElimination, but it seems to be defined in Hugr and called in Tket2. I don't want to change those flow.

As a data structure, we use these.
1. Input = Hugr with Call having metadata
  - qif is interpreted as call, but with some metadata about control
2. Call graph with modifier information:
   ```rust
   struct ModifierCFG<T:HugrView> {
    h : T,
    constraint : Map<Node, Vec<Node>>
   }
   ```
3. RichCircuit = Hugr with FuncDefn having metadata
    - The body of function is the same, but notes of modifier
    - Some functions will be duplicated at this point
    - The function call should be changed from Call(U) to Call(CU)
4. Circuit
    - Finally, compile it to normal circuit.
