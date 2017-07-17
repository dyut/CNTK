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

namespace CNTK {

// A randomizer that does not randomize input (identity function over the original timeline).
class LocalTimelineNoRandomizer : public LocalTimelineRandomizerBase
{
    typedef LocalTimelineRandomizerBase Base;

public:
    LocalTimelineNoRandomizer(
        DataDeserializerPtr deserializer,
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences = 0); // per worker

    std::map<std::wstring, size_t> GetInnerState() override;
    void SetInnerState(const std::map<std::wstring, size_t>& state) override;

    void RefillSequenceWindow(SequenceWindow& window) override;
    void Prefetch() const override;

private:
    // Current chunk position.
    ChunkIdType m_currentChunkPosition;

    // Current sequence position.
    size_t m_currentSequencePosition;

    // Prefetched chunks, expandable
    // can be recomputed after restore from the checkpoint.
    mutable std::tuple<ChunkInfo, ChunkPtr, std::vector<SequenceInfo>> m_prefetchedChunk;
};

}
