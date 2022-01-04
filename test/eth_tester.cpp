#include <iostream>

#include <ethash/ethash.hpp>

#include "nv_dev.hpp"
#include "eth_tester.hpp"

using namespace  std;


EthTester::EthTester(int epoch) : m_ec(ethash::get_global_epoch_context(epoch))
{
    m_ctx.epochNumber = epoch;
    m_ctx.lightNumItems = m_ec.light_cache_num_items;
    m_ctx.lightSize = ethash::get_light_cache_size(m_ec.light_cache_num_items);
    m_ctx.dagNumItems = m_ec.full_dataset_num_items;
    m_ctx.dagSize = ethash::get_full_dataset_size(m_ec.full_dataset_num_items);
    m_ctx.lightCache = m_ec.light_cache;  
}

struct EpochContexts& EthTester::get_epoch_ctx()
{
    return m_ctx;
}

bool EthTester::is_dev_existed(int devId)
{
    try {
        NvDev nv_dev = m_devices.at(devId);
        return true;
    } 
    catch (...) {
        return false;
    }
}

vector<NvDev>::iterator EthTester::get_nv_dev(int devId)
{
    vector<NvDev>::iterator it;

    for (it = m_devices.begin(); it != m_devices.end(); it++) {
        if (it->getDeviceId() == devId)
            return it;
    }

    return m_devices.end();
}

bool EthTester::add_device(int devId)
{
    if (get_nv_dev(devId) != m_devices.end()) {
        std::cout << "already existed, devId: " << devId << std::endl;
        return false;
    }

    m_devices.insert(m_devices.begin()+devId, NvDev(devId));

    std::cout << "count of m_devices: " << m_devices.size() << std::endl;

    return true;
}

bool EthTester::remove_device(int devId)
{
    if (get_nv_dev(devId) == m_devices.end()) {
        std::cout << "device is not added, devId: " << devId << std::endl;
        return false;
    }

    try {
        m_devices.erase(m_devices.begin() + devId);
    }
    catch(...) {
        return false;
    }

    std::cout << "count of m_devices: " << m_devices.size() << std::endl;

    return true;
}

bool EthTester::gen_dag(int devId)
{
    vector<NvDev>::iterator it = get_nv_dev(devId);

    if (it == m_devices.end()) {
        std::cout << "gen_dag: no device " << devId << std::endl;
        return false;
    }

    it->set_epoch_ctx(m_ctx);
    it->gen_dag();

    return true;
}

std::vector<test_result_t> EthTester::search(int devId, void *hdr_hash, uint64_t target, uint64_t start_nonce)
{
    vector<NvDev>::iterator it = get_nv_dev(devId);

    if (it == m_devices.end()) {
        std::cout << "search: no device " << devId << std::endl;
        return vector<test_result_t>();
    }

    std::vector<test_result_t> results = it->search(hdr_hash, target, start_nonce);

    // generate mix/final hash and verify
    for (auto res=results.begin();  res != results.end(); res++) {
        res->devId = devId;
        ethash::hash256 hhash;
        memcpy(hhash.bytes, hdr_hash, sizeof(hhash));
        struct ethash_result r = ethash::hash(m_ec, hhash, res->nonce);
        memcpy(res->mix_hash.bytes, r.mix_hash.bytes, sizeof(ethash::hash256));
        memcpy(res->final_hash.bytes, r.final_hash.bytes, sizeof(ethash::hash256));
    }
    return results;
}
