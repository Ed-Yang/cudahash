#pragma once

#include <iostream>

struct EpochContexts
{
    // ethash::epoch_context m_ec;
    int epochNumber;
    int lightNumItems;
    size_t lightSize;
    const void *lightCache;
    int dagNumItems;
    uint64_t dagSize;
};
