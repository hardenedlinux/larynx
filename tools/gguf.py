"""Minimal, self-contained GGUF v3 reader/writer (no torch dependency).

Implements exactly the binary layout documented in ggml's include/gguf.h and
produced by src/gguf.cpp::gguf_write_to_file, so files written here are loadable
by GGML's ``gguf_init_from_file``:

    "GGUF" | version(u32) | n_tensors(i64) | n_kv(i64)
    for each KV:      key(str) | type(i32) | value
    for each tensor:  name(str) | n_dims(u32) | dims[i64 x n_dims] | type(i32) | offset(u64)
    [pad to alignment]
    tensor data blob (each tensor padded to alignment; offsets relative to blob)

Strings = u64 length + UTF-8 bytes (no terminator). Enums stored as i32.
Bools stored as i8. All integers little-endian. Default alignment 32.
"""

import struct

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# gguf_type (KV value types), from gguf.h
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# ggml_type (tensor data types), from ggml.h — the subset relevant here.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_BF16 = 30

_GGML_TYPE_SIZE = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_I8: 1,
    GGML_TYPE_I16: 2,
    GGML_TYPE_I32: 4,
    GGML_TYPE_I64: 8,
    GGML_TYPE_F64: 8,
    GGML_TYPE_BF16: 2,
}

_SCALAR_FMT = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<b",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}

_SCALAR_SIZE = {t: struct.calcsize(f) for t, f in _SCALAR_FMT.items()}


def ggml_type_size(t):
    return _GGML_TYPE_SIZE[t]


def align_up(x, alignment):
    return (x + alignment - 1) // alignment * alignment


# --- low-level encoders ------------------------------------------------------

def _write_str(buf, s):
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b)) + b


def _write_scalar(buf, gguf_type, value):
    if gguf_type == GGUF_TYPE_STRING:
        _write_str(buf, value)
        return
    fmt = _SCALAR_FMT[gguf_type]
    buf += struct.pack(fmt, value)


# --- writer ------------------------------------------------------------------

def write_gguf(path, tensors, metadata=(), alignment=DEFAULT_ALIGNMENT):
    """Write a GGUF file.

    tensors: list of (name, ggml_type, ne, data) where ne is the GGML dimension
             list (reversed PyTorch shape) and data is the raw contiguous bytes.
    metadata: list of (key, gguf_type, value), or for arrays
              (key, GGUF_TYPE_ARRAY, (elem_gguf_type, [elements])).
    """
    n_tensors = len(tensors)
    n_kv = len(metadata)

    # Tensor byte offsets are relative to the start of the data blob and each
    # tensor is padded to `alignment` (first tensor at offset 0).
    offsets = []
    sizes = []
    off = 0
    for name, gtype, ne, data in tensors:
        nbytes = len(data)
        expected = 1
        for d in ne:
            expected *= d
        expected *= ggml_type_size(gtype)
        if nbytes != expected:
            raise ValueError(
                f"tensor '{name}': data {nbytes} bytes != ne={ne} * {ggml_type_size(gtype)}")
        offsets.append(off)
        sizes.append(nbytes)
        off += align_up(nbytes, alignment)

    buf = bytearray()
    buf += GGUF_MAGIC
    buf += struct.pack("<I", GGUF_VERSION)
    buf += struct.pack("<q", n_tensors)
    buf += struct.pack("<q", n_kv)

    for key, gtype, value in metadata:
        _write_str(buf, key)
        if gtype == GGUF_TYPE_ARRAY:
            elem_type, elems = value
            buf += struct.pack("<i", GGUF_TYPE_ARRAY)
            buf += struct.pack("<i", elem_type)
            buf += struct.pack("<Q", len(elems))
            for e in elems:
                _write_scalar(buf, elem_type, e)
        else:
            buf += struct.pack("<i", gtype)
            _write_scalar(buf, gtype, value)

    for (name, gtype, ne, data), offset in zip(tensors, offsets):
        _write_str(buf, name)
        buf += struct.pack("<I", len(ne))
        for d in ne:
            buf += struct.pack("<q", d)
        buf += struct.pack("<i", gtype)
        buf += struct.pack("<Q", offset)

    # pad the metadata section to alignment; data blob starts here
    while len(buf) % alignment != 0:
        buf.append(0)

    for (name, gtype, ne, data), size in zip(tensors, sizes):
        buf += data
        pad = align_up(size, alignment) - size
        buf += b"\x00" * pad

    with open(path, "wb") as f:
        f.write(buf)
    return path


# --- reader (for round-trip verification) ------------------------------------

def _read_str(data, pos):
    (n,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    s = data[pos:pos + n].decode("utf-8")
    return s, pos + n


def _read_scalar(data, pos, gtype):
    if gtype == GGUF_TYPE_STRING:
        return _read_str(data, pos)
    fmt = _SCALAR_FMT[gtype]
    (val,) = struct.unpack_from(fmt, data, pos)
    return val, pos + struct.calcsize(fmt)


def read_gguf(path):
    """Return (metadata, tensors) where tensors is list of
    (name, ggml_type, ne, data_bytes). Used only for round-trip tests."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    assert data[pos:pos + 4] == GGUF_MAGIC, "bad magic"
    pos += 4
    (version,) = struct.unpack_from("<I", data, pos)
    pos += 4
    (n_tensors,) = struct.unpack_from("<q", data, pos)
    pos += 8
    (n_kv,) = struct.unpack_from("<q", data, pos)
    pos += 8

    alignment = DEFAULT_ALIGNMENT
    metadata = []
    for _ in range(n_kv):
        key, pos = _read_str(data, pos)
        (gtype,) = struct.unpack_from("<i", data, pos)
        pos += 4
        if gtype == GGUF_TYPE_ARRAY:
            (elem_type,) = struct.unpack_from("<i", data, pos)
            pos += 4
            (count,) = struct.unpack_from("<Q", data, pos)
            pos += 8
            elems = []
            for _ in range(count):
                e, pos = _read_scalar(data, pos, elem_type)
                elems.append(e)
            metadata.append((key, GGUF_TYPE_ARRAY, (elem_type, elems)))
            if key == "general.alignment" and elem_type == GGUF_TYPE_UINT32:
                alignment = int(elems[0])
        else:
            val, pos = _read_scalar(data, pos, gtype)
            metadata.append((key, gtype, val))
            if key == "general.alignment" and gtype == GGUF_TYPE_UINT32:
                alignment = int(val)

    infos = []
    for _ in range(n_tensors):
        name, pos = _read_str(data, pos)
        (n_dims,) = struct.unpack_from("<I", data, pos)
        pos += 4
        dims = struct.unpack_from("<" + "q" * n_dims, data, pos)
        pos += 8 * n_dims
        (gtype,) = struct.unpack_from("<i", data, pos)
        pos += 4
        (offset,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        infos.append((name, gtype, list(dims), offset))

    # data blob begins at the next alignment boundary
    data_start = align_up(pos, alignment)
    tensors = []
    for name, gtype, ne, offset in infos:
        nbytes = 1
        for d in ne:
            nbytes *= d
        nbytes *= ggml_type_size(gtype)
        start = data_start + offset
        tensors.append((name, gtype, ne, data[start:start + nbytes]))
    return metadata, tensors
