"""Pure-Python Illumina IDAT reader. No methylprep/minfi. Stdlib + numpy only.
IDAT v3 binary format (little-endian). Returns per-address mean intensity, bead
count, and the header tags (barcode, chip type) needed for platform inference."""
import struct
import numpy as np

# IDAT field codes
NSNPSREAD, ILLUMINA_ID, SD, MEAN, NBEADS = 1000, 102, 103, 104, 107
MIDBLOCK, REDGREEN, BARCODE, CHIPTYPE = 200, 400, 402, 403

def _read_string(f):
    # Illumina/.NET 7-bit length-prefixed string
    n = 0; shift = 0
    while True:
        b = f.read(1)[0]
        n |= (b & 0x7F) << shift
        if not (b & 0x80): break
        shift += 7
    return f.read(n).decode("latin-1")

def read_idat(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"IDAT", "not an IDAT file"
        version = struct.unpack("<q", f.read(8))[0]
        nfields = struct.unpack("<i", f.read(4))[0]
        fields = {}
        for _ in range(nfields):
            code = struct.unpack("<H", f.read(2))[0]
            off  = struct.unpack("<q", f.read(8))[0]
            fields[code] = off
        f.seek(fields[NSNPSREAD]); n = struct.unpack("<i", f.read(4))[0]
        f.seek(fields[ILLUMINA_ID]); ids   = np.frombuffer(f.read(4*n), "<i4")
        f.seek(fields[MEAN]);        mean  = np.frombuffer(f.read(2*n), "<u2")
        nbeads = None
        if NBEADS in fields:
            f.seek(fields[NBEADS]); nbeads = np.frombuffer(f.read(n), "u1")
        barcode = chiptype = None
        if BARCODE in fields:  f.seek(fields[BARCODE]);  barcode  = _read_string(f)
        if CHIPTYPE in fields: f.seek(fields[CHIPTYPE]); chiptype = _read_string(f)
        redgreen = None
        if REDGREEN in fields:
            f.seek(fields[REDGREEN]); redgreen = struct.unpack("<i", f.read(4))[0]
    return {"n": n, "ids": ids, "mean": mean.astype(np.float64), "nbeads": nbeads,
            "barcode": barcode, "chiptype": chiptype, "redgreen": redgreen}

if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        d = read_idat(p)
        print(f"{p.split('/')[-1]}: n_addresses={d['n']:,}  barcode={d['barcode']}  "
              f"chip={d['chiptype']}  mean[min/med/max]={d['mean'].min():.0f}/"
              f"{np.median(d['mean']):.0f}/{d['mean'].max():.0f}  redgreen={d['redgreen']}")
