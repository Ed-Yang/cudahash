# Cuda ethash tester

The libcuda directory is copied from nsfminer.

Only test on Nvidia 1060 card.

## Cuda device

set default cuda device.

```shell
export CUDA_VISIBLE_DEVICES=0
```

## Build

## Run


## Informamation

### Grid size

* DAG Generation

gridSize (default: 8192)

ethash_calculate_dag_item<<<gridSize, blockSize, 0, stream>>>(base)

* Search

gridSize = m_block_multiple (default: 1000)

ethash_search<<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);



