//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include "LocalTimelineNoRandomizer.h"

namespace CNTK {

LocalTimelineNoRandomizer::LocalTimelineNoRandomizer(DataDeserializerPtr deserializer, bool multithreadedGetNextSequences, size_t maxNumberOfInvalidSequences)
: Base(deserializer, multithreadedGetNextSequences, maxNumberOfInvalidSequences),
  m_currentChunkPosition(0),
  m_currentSequencePosition(0)
{
}

void LocalTimelineNoRandomizer::Prefetch() const
{
    size_t capturedPosition = m_currentChunkPosition;
    auto chunkId = m_originalChunkDescriptions[capturedPosition].m_id;
    std::get<0>(m_prefetchedChunk) = m_originalChunkDescriptions[capturedPosition];
    std::get<1>(m_prefetchedChunk) = m_deserializer->GetChunk(chunkId);
    std::get<2>(m_prefetchedChunk).clear();
    std::get<1>(m_prefetchedChunk)->SequenceInfos(std::get<2>(m_prefetchedChunk));
}

void LocalTimelineNoRandomizer::RefillSequenceWindow(SequenceWindow& window)
{
    window.m_sequences.assign(std::get<2>(m_prefetchedChunk).begin(), std::get<2>(m_prefetchedChunk).end());
    window.m_dataChunks.clear();
    window.m_dataChunks[std::get<0>(m_prefetchedChunk).m_id] = std::get<1>(m_prefetchedChunk);

    if (Config().m_numberOfWorkers > 1)
    {
        // Decimate according to the position.
        size_t currentSequencePosition = m_currentSequencePosition;
        size_t currentInputIndex = 0;
        for (size_t i = 0; i < window.m_sequences.size(); ++i, ++currentSequencePosition)
        {
            if (currentSequencePosition % Config().m_numberOfWorkers == Config().m_workerRank)
                std::swap(window.m_sequences[currentInputIndex++], window.m_sequences[i]);
        }

        m_currentSequencePosition += window.m_sequences.size();
        window.m_sequences.erase(window.m_sequences.begin() + currentInputIndex);
    }

    // If last chunk, add the sweep marker.
    if (m_currentChunkPosition == m_originalChunkDescriptions.size() - 1)
    {
        window.m_sequences.push_back(s_endOfSweep);
        m_currentSequencePosition = 0;
    }

    // Moving to the next chunk.
    m_currentChunkPosition = (m_currentChunkPosition + 1) % m_originalChunkDescriptions.size();
}

// Properties used in the checkpoint.
const static std::wstring s_currentChunkPositionProperty = L"currentChunkPosition";
const static std::wstring s_currentSequencePositionProperty = L"currentSequencePosition";

std::map<std::wstring, size_t> LocalTimelineNoRandomizer::GetInnerState()
{
    std::map<std::wstring, size_t> state;
    state[s_currentChunkPositionProperty] = m_currentChunkPosition;
    state[s_currentSequencePositionProperty] = m_currentSequencePosition;
    return state;
}

void LocalTimelineNoRandomizer::SetInnerState(const std::map<std::wstring, size_t>& state)
{
    m_currentChunkPosition = (ChunkIdType)ValueFrom(state, s_currentChunkPositionProperty);
    m_currentSequencePosition = ValueFrom(state, s_currentSequencePositionProperty);
}

}
