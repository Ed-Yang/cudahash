#pragma once

#include <iostream>
#include <ethash/ethash.hpp>
#include <ethash/global_context.hpp>

#include "epoch_ctx.hpp"

class EthTester
{
    const ethash::epoch_context& m_ec;
public:
    EthTester() = default; 
    EthTester(int epoch);
    struct EpochContexts& get_epoch_ctx();
    bool add_device(int devId);
    bool remove_device(int devId);
    bool gen_dag(int devId);
    std::vector<test_result_t> search(int devId, void *hdr_hash, uint64_t target, uint64_t start_nonce);
    bool is_dev_existed(int devId);
    vector<NvDev>::iterator get_nv_dev(int devId);
protected:
    struct EpochContexts m_ctx;
    uint64_t m_target = 0;
    std::vector<NvDev> m_devices;
};