# Cuda ethash tester

The libcuda directory is copied from nsfminer.

This test program is only tested on one GeForce GTX 1060 6GB.

## Prerequisite

- Linux
- Gcc, Cmake, etc.
- Cuda driver

## Build

* Get source

```shell
git clone --recursive git@github.com:Ed-Yang/cudahash.git
```

* Build Ethash library

```shell
cd ethash
mkdir -p build && cd build && cmake .. && make
```

* Build test program

```shell
mkdir -p build && cd build && cmake .. && make
```

## Run

* Usage

```shell
$ ./build/cudahash -h
./build/cudahash
    -d <device-id>: <device-id> start from 0
    -s <cuda-streams>: <cuda-streams> in range of 2-4
    -m <blks> : <blks> default is 1000
    -n <start-nonce>: <start-nonce> default is 18393042511399634156
    -a: run all <cuda-streams> and <blks> combinations
```

* Run with default settings

```shell
$ ./build/cudahash 

Number of Cuda devices founds: 1
  Device ID: 00 - GeForce GTX 1060 6GB

count of m_devices: 1
>>> Run dev: 0, start-nonce: 18393042511399634156, streams: 2, multiple: 1000
set context for epoch 41
device: [0] dag generated for epoch 41
dag generated.
dev: start nonce = 18393042511399634156
dev: streams = 2
dev: m_block_multiple = 1000
search target: 0x0000000fffffffff
search found nonce: 18393042511533851884, took 8353.00 ms
dev st m_blks idx found nonce        time (ms)
=== == ====== === ================== ========
  0  2   1000   0 0xff4136b6b6a244ec  8353.00
```

* Run for all streams/block-multiple combination

```shell
....

dev st m_blks idx found nonce        time (ms)
=== == ====== === ================== ========
  0  2    128   0 0xff4136b6b6a244ec  8692.00
  0  3    128   2 0xff4136b6b6a244ec  8521.00
  0  4    128   0 0xff4136b6b6a244ec  8459.00
  0  2    512   0 0xff4136b6b6a244ec  8378.00
  0  3    512   2 0xff4136b6b6a244ec  8354.00
  0  2   1024   0 0xff4136b6b6a244ec  8353.00
  0  4    512   0 0xff4136b6b6a244ec  8353.00
  0  3   1024   1 0xff4136b6b6a244ec  8352.00
  0  4   1024   0 0xff4136b6b6a244ec  8352.00
  0  4   2048   0 0xff4136b6b6a244ec  8352.00
  0  2   2048   0 0xff4136b6b6a244ec  8351.00
  0  3   2048   2 0xff4136b6b6a244ec  8351.00
```

## Informamation

### Cuda device

set default cuda device.

```shell
export CUDA_VISIBLE_DEVICES=0
```

### Grid size

* DAG Generation

gridSize (default: 8192)

ethash_calculate_dag_item<<<gridSize, blockSize, 0, stream>>>(base)

* Search

gridSize = m_block_multiple (default: 1000)

ethash_search<<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);



