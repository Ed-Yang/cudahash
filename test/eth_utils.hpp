#pragma once

#include <ethash/ethash.hpp>

inline void hex_dump(const char *hdr, uint8_t *data, size_t len)
{
    auto *ptr = data;

    printf("%s", hdr);

    for (int i = 0; i < len; i++, ptr++)
    {
        if (i > 0 && i % 64 == 0)
        {
            printf("\n");
        }
        printf("%02x", *ptr);
        if (i > 0 && (i%4) == 0)
            printf(" ");
    }
    std::cout << std::endl;
}

inline uint8_t char2int(uint8_t input)
{
    if (input >= '0' && input <= '9')
        return input - '0';
    if (input >= 'A' && input <= 'F')
        return input - 'A' + 10;
    if (input >= 'a' && input <= 'f')
        return input - 'a' + 10;
    throw std::invalid_argument("Invalid input string");
}

// This function assumes src to be a zero terminated sanitized string with
// an even number of [0-9a-f] characters, and target to be sufficiently large
inline void hex2bin(const uint8_t *src, uint8_t *target)
{
    while (*src && src[1])
    {
        *(target++) = char2int(*src) * 16 + char2int(src[1]);
        src += 2;
    }
}

inline uint64_t to_target(ethash::hash256 boundary)
{
    uint64_t v;
    uint8_t *p = (uint8_t *)&v;
    int i;

    for (i=0; i < sizeof(uint64_t); i++, p++)
        *p = boundary.bytes[sizeof(uint64_t)-i-1];

    return v;
}
