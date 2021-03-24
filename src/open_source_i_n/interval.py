"""Interval/partition-related functionality."""
from math import ceil


class Interval(tuple):
    """Interval type for a range of values."""
    def __new__(cls, start, end, step=1):  # pylint: disable=unused-argument
        new_obj = super().__new__(cls, (start, end))
        return new_obj

    def __init__(self, start, end, step=1):  # pylint: disable=super-init-not-called,unused-argument
        self._step = step

    def __getnewargs__(self):
        return tuple(self)

    def __eq__(self, other):
        return tuple(self) == tuple(other) and self.step == other.step

    start = property(lambda self: self[0])
    end = property(lambda self: self[1])
    length = property(lambda self: self.end - self.start)
    step = property(lambda self: self._step)
    steps = property(lambda self: int(ceil(self.length / self.step)))


def partition_interval(start, end, partition_size, skip=1):
    """Return an iterator over a range into blocks of no more than parition_size; yields
    tuples of (block_start, block_end); same range semantics as Python for
    input and output.

    Partition sizes will be rounded down to the nearest multiple of skip.

    Args:
        start (int): Start of first sub-interval.
        end (int): Last sub-interval's last value.
        partition_size (int): Maximum length of a sub-interval.
        skip (int): Each sub-interval, except possibly the last one, is a multiple of skip.

    Returns:
        iterator: An iterator over the intervals."""
    assert partition_size >= skip
    partition_size -= partition_size % skip

    while start < end:
        yield Interval(start, min(start + partition_size, end), skip)
        start += partition_size + ((skip - partition_size % skip) % skip)


def partition_interval_chunks(start, end, chunks, align=1):
    """Return an iterator over partitions of a range into a total of `chunks` groups.
    tuples of (block_start, block_end); same range semantics as Python for
    input and output.

    Block offsets from start will align on multiples of skip, except for
    possibly the last partition.

    Args:
        start (int): Start of first sub-interval.
        end (int): Last sub-interval's last value.
        chunks (int): Number of chunks to create; this is the length of the iterator.
        align (int): Each interval that starts, aside from possibly the first, is aligned to this.

    Returns:
        iterator: An iterator over the intervals.
    """
    total_elements = int(ceil((end - start) / align))
    # Number of chunks with more elements
    larger_chunks = total_elements % chunks

    # Number of elements in smaller chunk size.
    smaller_chunk_size = total_elements // chunks

    for _ in range(chunks - larger_chunks):
        new_next = min(smaller_chunk_size * align + start, end)
        yield Interval(start, new_next, align)
        start = new_next

    while start < end:
        new_next = min((smaller_chunk_size + 1) * align + start, end)
        yield Interval(start, new_next, align)
        start = new_next
