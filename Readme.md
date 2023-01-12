# SIMD Experiments

This repository contains SIMD implementations of some common vector and string operations. This is a by-product of my study of SIMD intrinsics. This is mainly based off of the work done in:

- https://en.algorithmica.org/hpc/simd/
- http://0x80.pl/articles/simd-strfind.html and other articles by Wojciech Mula.

## Benchmarks

These tests were run on my personal laptop which has Ryzen 5 4600h.

### Vector

- The execution time in the tables are in micro-seconds.
- Tests were run on a vector 2^16 length.
- The implementation can be found in the file `ROOT/vector/parvec.cpp`. The template argument differentiates the different implementations of the same function.

|Operation|Naive|SIMD|SIMD Optimized|
|-|-|-|-|
|Sum|34.4|15.6|13.5|
|Mathematical length|84.8|21.14||
|Normalization|169|41.8||
|Sum of positives|291|13.2||
|Search|0.06|0.02|0.009|
|Count Occurrence of a value (Random list)|288.6|13.6|6.8|
|Count Occurrence of a value (Identical elements)|34.5|13.6|6.8|
|Minimum|237.4|12.7|5.6|

### String

- The execution time in the tables are in millis-seconds.
- Tests were run on a string of length 2^24.
- The implementation can be found in the file `ROOT/string/string_ops.cpp`. The template argument differentiates the different implementations of the same function.
- The library implementation of `Equality` and `String length` is faster/similar to SIMD because internally it seems to process multiple characters at once.

|Operation|Naive|Lib|SIMD|SIMD Optimized|
|-|-|-|-|-|
|To Lower||30.98|2.7||
|Equality|5.5|0.98|0.92||
|String length|6.4|0.81|0.89|1.1|
|Search|6.5|57|2||
|Small string search|6.7|57|2||

## TODO

- Investigate cases where SIMD is not giving significant performance improvment.
- My initial experiments with prefetching only caused slowness in the code, so experiment more thoroughly on it and see if it helps.
- Benchmarks for different vector and string length, e.g. when it fits L3 cache, when it hits main memory.
- Exploring Tries designed using SIMD instructions like FAST/VAST/ART.