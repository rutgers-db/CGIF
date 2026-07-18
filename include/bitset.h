/*
 * Created on Wed Jul 03 2024
 *
 * Author: Chaoji Zuo
 * Email: chaoji.zuo@rutgers.edu
 *
 * Copyright (c) 2024 Rutgers University
 */

#ifndef BITSET_H
#define BITSET_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "bm25.h"

/**
empty(): 判断位集是否为空。
size(): 返回位集的大小（位数）。
byte_size(): 返回存储这些位所需的字节数。
data(): 返回底层数据指针。
test(int64_t index): 测试指定索引处的位是否为1。
count(): 计算位集中为1的位的数量。
to_string(size_t from, size_t to): 将位集从from到to的部分转换为字符串。
 *
 */

// Borrow from zilliz knowhere implementation:
// https://github.com/zilliztech/knowhere/blob/main/include/knowhere/bitsetview.h

// reference on Bit Twiddling Hacks: https://graphics.stanford.edu/~seander/bithacks.html
// another reference: https://hackmd.io/@0xff07/ORAORAORAORA

// 1 will be skipped during search

class PredicateBaseView {
 public:
    virtual inline bool
    is_member(int64_t id) const = 0;
    virtual ~PredicateBaseView() = default;
};

class BitsetView final : public PredicateBaseView {
 public:
    BitsetView() = default;
    ~BitsetView() override {
    }

    BitsetView(uint8_t* data, size_t num_bits) : bits_(data), num_bits_(num_bits) {
    }

    // BitsetView(const uint8_t* data, size_t num_bits) : bits_(data), num_bits_(num_bits) {
    // }

    BitsetView(const std::nullptr_t) : BitsetView() {
    }

    bool
    empty() const {
        return num_bits_ == 0;
    }

    size_t
    size() const {
        return num_bits_;
    }

    size_t
    byte_size() const {
        return (num_bits_ + 8 - 1) >> 3;
    }

    const uint8_t*
    data() const {
        return bits_;
    }

    bool
    test(int64_t index) const {
        // when index is larger than the max_offset, ignore it
        return (index >= static_cast<int64_t>(num_bits_)) || (bits_[index >> 3] & (0x1 << (index & 0x7)));
    }

    inline bool
    is_member(int64_t id) const override final {
        return !((id >= static_cast<int64_t>(num_bits_)) || (bits_[id >> 3] & (0x1 << (id & 0x7))));
    }

    size_t
    count() const {
        size_t ret = 0;
        auto len_uint8 = byte_size();
        auto len_uint64 = len_uint8 >> 3;

        auto popcount8 = [&](uint8_t x) -> int {
            x = (x & 0x55) + ((x >> 1) & 0x55);
            x = (x & 0x33) + ((x >> 2) & 0x33);
            x = (x & 0x0F) + ((x >> 4) & 0x0F);
            return x;
        };

        uint64_t* p_uint64 = (uint64_t*)bits_;
        for (size_t i = 0; i < len_uint64; i++) {
            ret += __builtin_popcountll(*p_uint64);
            p_uint64++;
        }

        // calculate remainder
        uint8_t* p_uint8 = (uint8_t*)bits_ + (len_uint64 << 3);
        for (size_t i = (len_uint64 << 3); i < len_uint8; i++) {
            ret += popcount8(*p_uint8);
            p_uint8++;
        }

        return std::min(ret, num_bits_);
    }

    std::string
    to_string(size_t from, size_t to) const {
        if (empty()) {
            return "";
        }
        std::stringbuf buf;
        to = std::min<size_t>(to, num_bits_);
        for (size_t i = from; i < to; i++) {
            buf.sputc(test(i) ? '1' : '0');
        }
        return buf.str();
    }

    // find next unset position / find next sataisfied point
    std::size_t
    next_unset_position(size_t current_pos) const {
        size_t current_index = (current_pos) >> 3;
        bool find_one_satisfied = false;
        size_t found_position = 0;
        for (size_t idx = current_index + 1; idx < this->byte_size(); idx++) {
            if (this->bits_[idx] != 0xff) {
                found_position = __builtin_ffs(~this->bits_[idx]) + idx * 8 - 1;
                find_one_satisfied = true;
                break;
            }
        }
        if (!find_one_satisfied) {
            for (size_t idx = 0; idx < current_index; idx++) {
                if (this->bits_[idx] != 0xff) {
                    found_position = __builtin_ffs(~this->bits_[idx]) + idx * 8 - 1;
                    find_one_satisfied = true;
                    break;
                }
            }
        }
        return find_one_satisfied ? found_position : current_pos;
    }

    size_t
    find_next_unset(size_t current_pos) const {
        if (current_pos + 1 >= num_bits_)
            return num_bits_;

        size_t target = current_pos + 1;
        constexpr size_t WORD_BITS = sizeof(unsigned long) * 8;

        // number of machine‐words covering the bitset
        size_t n_words = (byte_size() + sizeof(unsigned long) - 1) / sizeof(unsigned long);
        auto words = reinterpret_cast<const unsigned long*>(bits_);

        // which word + offset to start at
        size_t wi = target / WORD_BITS;
        size_t offset = target % WORD_BITS;

        // build an initial mask to ignore bits < offset
        unsigned long mask = (~0UL << offset);

        // invert the word to treat 0→1, 1→0, then mask off lower bits
        unsigned long w = (~words[wi] & mask);

        // if this is the last word, also mask off bits beyond num_bits_
        if (wi + 1 == n_words) {
            size_t valid_bits = num_bits_ - wi * WORD_BITS;
            // tail_mask has ones in the low `valid_bits` positions
            unsigned long tail_mask = valid_bits == WORD_BITS ? ~0UL : (~0UL >> (WORD_BITS - valid_bits));
            w &= tail_mask;
        }

        // scan current word and any following words
        for (; wi < n_words; ++wi) {
            if (w) {
                // __builtin_ctzll gives the index of the first 1-bit in w,
                // i.e. where our original bit was 0
                size_t bit = __builtin_ctzll(w);
                size_t pos = wi * WORD_BITS + bit;
                return pos < num_bits_ ? pos : current_pos;
            }
            // advance to the next inverted word
            if (wi + 1 < n_words) {
                w = ~words[wi + 1];
                // if it's the last word, apply the tail mask again
                if (wi + 2 == n_words) {
                    size_t valid_bits = num_bits_ - (wi + 1) * WORD_BITS;
                    unsigned long tail_mask = valid_bits == WORD_BITS ? ~0UL : (~0UL >> (WORD_BITS - valid_bits));
                    w &= tail_mask;
                }
            }
        }

        // no zero-bit found
        return num_bits_;
    }

    size_t
    find_next_set(size_t current_pos) const {
        // if we’ve already reached or passed the end, stop
        if (current_pos + 1 >= num_bits_)
            return num_bits_;

        size_t target = current_pos + 1;
        constexpr size_t WORD_BITS = sizeof(unsigned long) * 8;
        size_t n_words = (byte_size() + sizeof(unsigned long) - 1) / sizeof(unsigned long);
        auto words = reinterpret_cast<const unsigned long*>(bits_);

        size_t wi = target / WORD_BITS;
        size_t offset = target % WORD_BITS;

        // mask off bits < offset
        unsigned long w = words[wi] & (~0UL << offset);

        // final-word tail masking
        if (wi + 1 == n_words) {
            size_t valid = num_bits_ - wi * WORD_BITS;
            unsigned long tail = (valid == WORD_BITS) ? ~0UL : (~0UL >> (WORD_BITS - valid));
            w &= tail;
        }

        // hunt for the next set bit
        for (; wi < n_words; ++wi) {
            if (w) {
                size_t bit = __builtin_ctzll(w);
                size_t pos = wi * WORD_BITS + bit;
                return (pos < num_bits_) ? pos : num_bits_;
            }
            if (wi + 1 < n_words) {
                w = words[wi + 1];
                if (wi + 2 == n_words) {
                    size_t valid = num_bits_ - (wi + 1) * WORD_BITS;
                    unsigned long tail = (valid == WORD_BITS) ? ~0UL : (~0UL >> (WORD_BITS - valid));
                    w &= tail;
                }
            }
        }
        return num_bits_;
    }

    void
    set_one(size_t pos) {
        bits_[pos >> 3] |= (0x1 << (pos & 0x7));
    }

    void
    init_filter_id_map() {
        filter_id_map_ = new char[num_bits_];

        // Process in byte-sized chunks where possible
        size_t byte_count = byte_size();
        for (size_t byte_idx = 0; byte_idx < byte_count; byte_idx++) {
            uint8_t current_byte = bits_[byte_idx];
            size_t base_bit_idx = byte_idx * 8;

            // Process each bit in the current byte
            for (size_t bit_offset = 0; bit_offset < 8; bit_offset++) {
                size_t bit_idx = base_bit_idx + bit_offset;
                if (bit_idx >= num_bits_)
                    break;  // Don't exceed total bits

                // Check if the bit is set (1)
                filter_id_map_[bit_idx] = (current_byte & (1 << bit_offset)) ? 0 : 1;
            }
        }
    }

    char*
    get_filter_id_map() {
        if (filter_id_map_ == nullptr) {
            std::cout << "filter_id_map_ is not initialized" << std::endl;
            assert(false);
        }
        return filter_id_map_;
    }

 private:
    uint8_t* bits_ = nullptr;
    size_t num_bits_ = 0;

    char* filter_id_map_ = nullptr;
};

class TextPredicateView : public PredicateBaseView {
 public:
    TextPredicateView(std::shared_ptr<std::vector<std::vector<std::string>>> base_text_data,
                      std::vector<std::string> query_words)
        : base_text_data_(base_text_data), query_words_(query_words) {
    }

    inline bool
    is_member(int64_t id) const override final {
        for (const auto& word : query_words_) {
            if (std::find(base_text_data_->at(id).begin(), base_text_data_->at(id).end(), word) !=
                base_text_data_->at(id).end()) {
                return true;
            }
        }
        return false;
    }

 private:
    std::shared_ptr<std::vector<std::vector<std::string>>> base_text_data_;
    std::vector<std::string> query_words_;
};

class BM25View : public PredicateBaseView {
 public:
    BM25View(std::shared_ptr<BM25> bm25, std::vector<std::string> query_words, float threshold)
        : bm25_(bm25), query_words_(query_words), threshold_(threshold) {
    }

    inline bool
    is_member(int64_t id) const override final {
        if (!bm25_ || id < 0) {
            return false;
        }
        float score = bm25_->getScore(query_words_, id);
        return score >= threshold_;
    }

 private:
    std::shared_ptr<BM25> bm25_;
    std::vector<std::string> query_words_;
    float threshold_;
};

#endif /* BITSET_H */
