# ID_208 - Global-memory alignment contract

ID_208 defines the reusable alignment gate for vectorized SM89 memory accesses.

For a candidate width `V` in bytes, a request is vector-safe only when all of the following hold:

- `address % V == 0`
- `stride_bytes % V == 0`
- `V % element_size == 0`
- `element_count % (V / element_size) == 0`

The selector tests widths in descending order `{16, 8, 4}` and returns the first safe width. If no width is safe, or the device is not SM89, the result selects one scalar element. Invalid zero-sized requests remain unavailable.

The contract does not reinterpret misaligned tails as vector traffic. Callers use the returned scalar fallback, preserving exact access semantics without assuming board-specific alignment behavior.
