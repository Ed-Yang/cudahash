# Shuffle Test

## Build

```shell
cd expr
/usr/local/cuda/bin/nvcc -arch=sm_61 -o shuffle shuffle.cu
```

## Run

```shell
./shuffle 
```

or 

```shell
/usr/local/cuda/bin/cuda-memcheck ./shuffle
```

```shell
test_sync_1:
===============================
00 0:10 20:30 40:50 60:70
01 1:11 21:31 41:51 61:71
02 2:12 22:32 42:52 62:72
03 3:13 23:33 43:53 63:73
04 400:410 420:430 440:450 460:470
05 401:411 421:431 441:451 461:471
06 402:412 422:432 442:452 462:472
07 403:413 423:433 443:453 463:473
test_sync_2:
===============================
00 0:10 20:30 40:50 60:70
01 1:11 21:31 41:51 61:71
02 2:12 22:32 42:52 62:72
03 3:13 23:33 43:53 63:73
04 400:410 420:430 440:450 460:470
05 401:411 421:431 441:451 461:471
06 402:412 422:432 442:452 462:472
07 403:413 423:433 443:453 463:473
test_sync_3 (mask = 0x01):
===============================
00 mask = 0x1 value = 0x0 (active = 0xff)
01 mask = 0x1 value = 0x0 (active = 0xff)
02 mask = 0x1 value = 0x0 (active = 0xff)
03 mask = 0x1 value = 0x0 (active = 0xff)
04 mask = 0x1 value = 0x0 (active = 0xff)
05 mask = 0x1 value = 0x0 (active = 0xff)
06 mask = 0x1 value = 0x0 (active = 0xff)
07 mask = 0x1 value = 0x0 (active = 0xff)
test_sync_3 (mask = 0xff):
00 mask = 0xff value = 0x0 (active = 0xff)
01 mask = 0xff value = 0x0 (active = 0xff)
02 mask = 0xff value = 0x0 (active = 0xff)
03 mask = 0xff value = 0x0 (active = 0xff)
04 mask = 0xff value = 0x0 (active = 0xff)
05 mask = 0xff value = 0x0 (active = 0xff)
06 mask = 0xff value = 0x0 (active = 0xff)
07 mask = 0xff value = 0x0 (active = 0xff)
```