# NPOW
A Neural Proof-of-Work [W.I.P]

## Advantages
- A much more diverse range of solutions to solving a Proof-of-Work.
- Lower power consumption as the process to finding a solution is not a constant thrashing of hashes.
- Almost completely reduces dependence on mining pools, it seems hard to imagine a scenario where a mining pool would be beneficial.
- GPU mining is not particularly advantagous over CPU mining in this type of neural network.

## Disadvantages
- Proof-of-Work is no longer a 4 byte nonce in the submitted block but now a set of weights from 8.32kb if int8 quantised or at worst 33.2kb if provided as regular float32's.
- Proof-of-Work verification is no longer an O(1) if statement but now an O(n^2 + n^2 + 2n) iteration where n = 64.

Blocks are only generated every 10 minutes which makes the disadvantages negligible.
