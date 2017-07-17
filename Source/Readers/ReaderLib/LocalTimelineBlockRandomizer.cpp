//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <random>

#include "LocalTimelineBlockRandomizer.h"
#include "RandomOrdering.h"
#include <tuple>

namespace CNTK {

LocalTimelineBlockRandomizer::LocalTimelineBlockRandomizer(
    DataDeserializerPtr deserializer,
    bool sampleBasedRandomizationWindow,
    size_t randomizationRange,
    size_t seedOffset,
    bool multithreadedGetNextSequences,
    size_t maxNumberOfInvalidSequences)
: Base(deserializer, multithreadedGetNextSequences, maxNumberOfInvalidSequences),
  m_randomizationRange(randomizationRange),
  m_seedOffset(seedOffset),
  m_chunkPosition(0),
  m_sampleBasedRandomizationWindow(sampleBasedRandomizationWindow),
  m_sweepIndex(0)
{
    m_prefetchedChunkDescriptions = m_originalChunkDescriptions;
    m_rng.seed((unsigned long)m_sweepIndex + m_seedOffset);
    Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
}

void LocalTimelineBlockRandomizer::Prefetch() const
{
    size_t originalChunkPosition = m_chunkPosition;
    size_t originalSweepIndex = m_sweepIndex;

    size_t position = m_chunkPosition;
    size_t sweepIndex = m_sweepIndex;

    // Prefetch does not change any state that cannot be recalculated,
    // only prefetches data.
    int64_t range = m_randomizationRange;
    m_prefetchedChunks.clear();
    m_prefetchedSequences.clear();
    while (range > 0)
    {
        auto desc = m_prefetchedChunkDescriptions[position];
        if (position % Config().m_numberOfWorkers == Config().m_workerRank) // Need to add to the window
        {
            size_t oldSize = m_prefetchedSequences.size();

            // Query deserializer.
            ChunkPtr data = m_deserializer->GetChunk(desc.m_id);
            data->SequenceInfos(m_prefetchedSequences);
            m_prefetchedChunks.push_back(std::make_tuple(desc, data));

            if (!m_sampleBasedRandomizationWindow)
                --range;
            else
                for (size_t i = oldSize; i < m_prefetchedSequences.size(); ++i)
                    range -= m_prefetchedSequences[i].m_numberOfSamples;
        }
        else
        {
            // Empty, we do not need data , only for tracking the current chunk.
            m_prefetchedChunks.push_back(std::make_tuple(ChunkInfo{}, nullptr));
        }

        if (position == m_originalChunkDescriptions.size() - 1)
        {
            sweepIndex++;
            m_prefetchedChunkDescriptions = m_originalChunkDescriptions;
            m_rng.seed((unsigned long)sweepIndex + m_seedOffset);
            Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
            m_prefetchedSequences.push_back(s_endOfSweep);
        }

        position = (position + 1) % m_originalChunkDescriptions.size();
    }

    // Find all end of sweep markers and randomize among them.
    if (sweepIndex == originalSweepIndex) // Same sweep, simply randomize.
    {
        m_rng.seed((unsigned long)(originalChunkPosition + sweepIndex + m_seedOffset));
        Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedSequences, m_rng);
    }
    else // When several sweeps - make sure randomize only inside the sweep.
    {
        std::vector<std::pair<size_t, size_t>> sweepIndices;
        size_t curPos = 0;
        for (size_t i = 0; i < m_prefetchedSequences.size(); ++i)
            if (IsEndOfSweep(m_prefetchedSequences[i]))
            {
                sweepIndices.push_back(std::make_pair(curPos, i));
                curPos = i + 1;
            }

        sweepIndices.push_back(std::make_pair(curPos, m_prefetchedSequences.size()));
        size_t randomizationPositionInSweep = originalChunkPosition;
        for (size_t i = 0; i < sweepIndices.size(); ++i)
        {
            m_rng.seed((unsigned long)(randomizationPositionInSweep + originalSweepIndex + i + m_seedOffset));
            randomizationPositionInSweep = 0; // Make sure same as we start from the beginning of the sweep.
            Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedSequences, sweepIndices[i].first, sweepIndices[i].second, m_rng);
        }
    }
}

void LocalTimelineBlockRandomizer::RefillSequenceWindow(SequenceWindow& window)
{
    window.m_sequences.clear();
    window.m_dataChunks.clear();

    window.m_sequences.insert(window.m_sequences.end(), m_prefetchedSequences.begin(), m_prefetchedSequences.end());
    for (const auto& s : window.m_sequences)
        if (IsEndOfSweep(s))
            m_sweepIndex++;

    for (const auto& c : m_prefetchedChunks)
    {
        window.m_dataChunks.insert(std::make_pair(std::get<0>(c).m_id, std::get<1>(c)));
        m_chunkPosition = (m_chunkPosition + 1) % m_originalChunkDescriptions.size();
    }
}

// Properties used in the checkpoint.
const static std::wstring s_chunkPositionProperty = L"chunkPosition";
const static std::wstring s_sweepIndexProperty = L"sweepIndex";

std::map<std::wstring, size_t> LocalTimelineBlockRandomizer::GetInnerState()
{
    std::map<std::wstring, size_t> state;
    state[s_chunkPositionProperty] = m_chunkPosition;
    state[s_sweepIndexProperty] = m_sweepIndex;
    return state;
}

void LocalTimelineBlockRandomizer::SetInnerState(const std::map<std::wstring, size_t>& state)
{
    m_sweepIndex = ValueFrom(state, s_sweepIndexProperty);
    m_rng.seed((unsigned long)m_sweepIndex + m_seedOffset);
    m_prefetchedChunkDescriptions = m_originalChunkDescriptions;
    Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
    m_chunkPosition = (ChunkIdType)ValueFrom(state, s_chunkPositionProperty);
}

}
