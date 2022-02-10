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

* Build test program (release)

```shell
mkdir -p build && cd build && cmake .. && make
```

* Build test program (enable source-level debug)

```shell
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make
```

## Run

* Usage

```shell
$ ./build/cudahash -h
./build/cudahash
-d <device-id>: <device-id> start from 0
    -s <cuda-streams>: <cuda-streams> in range of 1-4
    -m <mblks> : <mblks> that is grid size, default is 1000
    -b <blk-size> : <blk-size> block size, default is 128 (must be multiple of 8)
    -n <start-nonce>: <start-nonce> default is 18393042511399634156
    -a: run all <cuda-streams> and <blks> combinations
```

Note, set "-m" or "-b" only affect the block size in search operation, 
the DAG generation use fixed value (-m 1000 -b 128).

* Run with default settings

```shell
$ ./build/cudahash 
```

Number of Cuda devices founds: 1
  Device ID: 00 - GeForce GTX 1060 6GB

count of m_devices: 1
>>> Run dev: 0, start-nonce: 18393042511399634156, streams: 2, multiple: 1000, block-size=128
set context for epoch 41
device: [0] dag generated for epoch 41
dag generated.
dev: start nonce = 18393042511399634156
dev: streams = 2
dev: m_block_multiple = 1000
dev: block_size = 128
search target: 0x0000000fffffffff
search: nonce = start_nonce(18393042511534034156) - stream_blocks(256000) + r.gid[i](73728)
search: found nonce = 18393042511533851884, took 8326.00 ms
dev st m_blks idx found nonce          time (ms)
=== == ====== === ==================== ========
  0  2   1000   0 18393042511533851884  8326.00

* Run for all streams/block-multiple combination

```shell
$ ./build/cudahash -a
....

dev st m_blks idx found nonce          time (ms)
=== == ====== === ==================== ========
  0  1    128   0 18393042511533851884 10148.00
  0  1    512   0 18393042511533851884  8966.00
  0  1   1024   0 18393042511533851884  8627.00
  0  2    128   0 18393042511533851884  8530.00
  0  1   2048   0 18393042511533851884  8485.00
  0  3    128   2 18393042511533851884  8426.00
  0  4    128   0 18393042511533851884  8374.00
  0  2    512   0 18393042511533851884  8339.00
  0  3    512   2 18393042511533851884  8325.00
  0  4    512   0 18393042511533851884  8325.00
  0  3   1024   1 18393042511533851884  8324.00
  0  4   1024   0 18393042511533851884  8324.00
  0  2   1024   0 18393042511533851884  8323.00
  0  2   2048   0 18393042511533851884  8323.00
  0  3   2048   2 18393042511533851884  8323.00
  0  4   2048   0 18393042511533851884  8323.00

* Run with verbose message enabled

In file "dagger_shuffled.cuh", change DEBUG_HASH  to 1 and re-build program.

```shell
./cudahash -n 18393042511533851884 -m 1 -s 1 -b 8
```

The detail log, refer to [this](doc/flow-debug-message.txt)

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
blockSize = cuBlockSize = 128

run_ethash_search(gridSize, blockSize, stream, g_output, start_nonce)
    ethash_search<<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
        uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
        bool r = compute_hash(start_nonce + gid);
