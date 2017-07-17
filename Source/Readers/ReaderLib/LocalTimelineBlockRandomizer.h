//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "SequenceEnumerator.h"
#include "DataDeserializer.h"
#include "ReaderUtil.h"
#include "LocalTimelineRandomizerBase.h"
#include <tuple>

namespace CNTK {

// A randomizer that firstly randomizes chunks and then sequences inside a tumbling window of chunks.
class LocalTimelineBlockRandomizer : public LocalTimelineRandomizerBase
{
    typedef LocalTimelineRandomizerBase Base;

public:
    LocalTimelineBlockRandomizer(
        DataDeserializerPtr deserializer,
        bool sampleBasedRandomizationWindow,
        size_t randomizationRange,
        size_t seedOffset = 0,
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences= 0); // per worker

    std::map<std::wstring, size_t> GetInnerState() override;
    void SetInnerState(const std::map<std::wstring, size_t>& state) override;
    void RefillSequenceWindow(SequenceWindow& window) override;
    void Prefetch() const override;

private:
    const size_t m_randomizationRange;
    const size_t m_seedOffset;
    const bool m_sampleBasedRandomizationWindow;

    // Current chunk position that the randomizer works with.
    ChunkIdType m_chunkPosition;
    // Current sweep index.
    size_t m_sweepIndex;

    // Expandable, can be recalculated any time, no need to store them
    // in checkpoint.
    mutable std::mt19937_64 m_rng;
    mutable std::vector<ChunkInfo> m_prefetchedChunkDescriptions;
    mutable std::vector<SequenceInfo> m_prefetchedSequences;
    mutable std::vector<std::tuple<ChunkInfo, ChunkPtr>> m_prefetchedChunks;
};

}
