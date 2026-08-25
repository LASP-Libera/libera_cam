# WFOV FSW Header Reference

Reference for the 36-byte FSW metadata block at the start of each WFOV NAND image blob, as delivered in ICIE WFOV science packets (APID 1040 / `WFOV_SCI`).

**Audience:** L1A/L1B developers and operators decoding WFOV camera data.

**Canonical decoder:** [`libera_utils.l1a.wfov_image_metadata`](https://github.com/LASP-Libera/libera_utils)
(`enhance_wfov_l1a_dataset`). `libera_cam` reads the decoded L1A variables and does not parse
NAND headers itself. Each decoded field carries the prefix of its metadata block —
`WFOV_FSW_HEADER_`, `WFOV_IMAGE_HEADER_`, `WFOV_IMAGE_FOOTER_`, or `WFOV_FPGA_STATUS_` — and the
JPEG-LS payload itself is `WFOV_COMPRESSED_IMAGE`.

**Related specs:**

- FSW struct: `wfovImgMetaData_t` in libfsw `modules/wfov/src/WFOVPkt.h`
- CTDB frame: `wfov_img_meta_data` in libctdb
- L1A per-SOP metadata: libera_utils `libera_utils/l1a/wfov_image_metadata.py`
- Exposure conversions (FPGA actual / DELTA → ms): [Exposure timing](#exposure-timing)

---

## Scope

WFOV SCI (APID 1040) is a **mem-dump packet stream**. Each packet carries raw `wfov_data` bytes read from NAND flash. The FSW header is **not** a separate CCSDS field; it is embedded at **byte 0** of each reassembled image blob (the SOP packet where `mem_dump_offset == 0`).

Layout:

```
[NAND image blob]
├── FSW header     (36 bytes)  ← this document
├── FPGA header    (140 bytes)
├── compressed image payload
└── FPGA footer / EOF markers
```

---

## Data path: where the header is written

FSW does not assemble the header into the telemetry packet directly. The flow is:

```mermaid
flowchart LR
  camStart["WFOV_CAM_START command"]
  acquire["WFOVManager::acquire"]
  imgMeta["FPGA IMG_META registers"]
  fpgaWrite["FPGA writes image plus header to NAND"]
  tlmTask["WFOVTlmTask reads NAND"]
  sciPkt["APID 1040 mem-dump packets"]
  l1a["L1A SOP decode"]

  camStart --> acquire
  acquire --> imgMeta
  imgMeta --> fpgaWrite
  fpgaWrite --> tlmTask
  tlmTask --> sciPkt
  sciPkt --> l1a
```

1. **`WFOV_CAM_START`** sets imaging parameters (`mode`, `pixelMaskID`, `cadence`, `videoNum`, `videoSkip`, etc.).
2. **`WFOVManager::acquire()`** (libfsw `modules/wfov/src/WFOVManager.cpp`) packs metadata into FPGA `IMG_META_*` registers via `HAL_WFOV::setIMG_META_*`, then triggers the camera.
3. **FPGA hardware** prepends those registers (36 bytes) to each image written to NAND.
4. **`WFOVTlmTask`** reads NAND and publishes mem-dump SCI packets.
5. **Ground L1A** decodes the FSW block from each qualifying SOP slice.

Key write site (byte 2 = `pixel_mask_id`, byte 1 flags = `img_mode`, etc.):

```cpp
meta0 |= sharedData.managerOwned.pixelMaskID << 8;
meta0 |= HAL_WFOV::getPXL_PROC_EN_SELECT() << 17;
// ... jpeg_bypass, bitmask_disable, testpattern, bitmask_id ...
HAL_WFOV::setIMG_META_0(meta0);
```

Timestamps are set after the camera command is sent (`setIMG_META_3/4` from `protocol.lastTxTime`).

---

## FSW header byte layout (36 bytes)

All multi-byte integers are **big-endian** unless noted. Total size is 9 × 32-bit meta words.

| Offset | Field                  | Size | Type    | Notes                                                          |
| -----: | ---------------------- | ---: | ------- | -------------------------------------------------------------- |
|      0 | `fsw_length`           |    1 | uint8   | **Number of 32-bit meta words (9)**, not total byte count      |
|      1 | flags (packed)         |    1 | uint8   | See [Byte 1 flags](#byte-1-packed-flags)                       |
|      2 | `pixel_mask_id`        |    1 | uint8   | From `WFOV_CAM_START` `pixelMaskID`                            |
|      3 | `simulator`            |    1 | uint8   | Reserved/pad in flight; ops sim may set to 1                   |
|    4–5 | `cadence`              |    2 | uint16  | Imaging cadence (ms)                                           |
|      6 | `image_total`          |    1 | uint8   | Total images in sequence                                       |
|      7 | `image_count`          |    1 | uint8   | Current image index in sequence                                |
|   8–11 | `flash_write_pointer`  |    4 | uint32  | NAND write page pointer at trigger                             |
|  12–15 | `timestamp_seconds`    |    4 | uint32  | Acquisition time (seconds)                                     |
|  16–19 | `timestamp_subseconds` |    4 | uint32  | Acquisition time (microseconds)                                |
|  20–21 | `rad_obs_id`           |    2 | uint16  | Radiometer observation ID                                      |
|  22–23 | `cam_obs_id`           |    2 | uint16  | WFOV observation ID                                            |
|  24–27 | `commanded_exp_time_1` |    4 | uint32  | Commanded exposure 1 (see [Exposure timing](#exposure-timing)) |
|  28–31 | `commanded_exp_time_2` |    4 | uint32  | Commanded exposure 2 (see [Exposure timing](#exposure-timing)) |
|  32–35 | `azimuth_angle`        |    4 | float32 | Azimuth angle (radians)                                        |

Decode implementation:

```python
metadata["fsw_length"] = struct.unpack("B", file.read(1))[0]
second_byte = struct.unpack("B", file.read(1))[0]
metadata["jpeg_bypass"] = (second_byte >> 7) & 1
metadata["bitmask_disable"] = (second_byte >> 6) & 1
metadata["testpattern"] = (second_byte >> 5) & 1
metadata["bitmask_id"] = (second_byte >> 3) & 0x03
metadata["img_mode"] = (second_byte >> 1) & 0x03
metadata["pixel_mask_id"] = struct.unpack("B", file.read(1))[0]
# ...
```

In `libera_utils` L1A products, these fields appear as `WFOV_FSW_HEADER_*` variables on the `CAMERA_TIME` dimension.

---

## Byte 1 packed flags

| Bit(s) | Field             | Meaning                                                             |
| ------ | ----------------- | ------------------------------------------------------------------- |
| 7      | `jpeg_bypass`     | JPEG compression disabled when 1                                    |
| 6      | `bitmask_disable` | Pixel bitmask processing disabled when 1 (`WFOV_SET_CFG mask DIS`)  |
| 5      | `testpattern`     | Test pattern enabled when 1                                         |
| 4–3    | `bitmask_id`      | 2-bit custom bitmask **slot** (0–3) for `CUSTOM_0`…`CUSTOM_3` masks |
| 2–1    | `img_mode`        | 2-bit FPGA processing select; see [img_mode](#img_mode)             |
| 0      | (unused)          | —                                                                   |

---

## pixel_mask_id (byte 2)

Single-byte mask type from the **`pixelMaskID`** argument on `WFOV_CAM_START`. Stored in `sharedData.managerOwned.pixelMaskID` and packed into `IMG_META_0` before each exposure.

### Command values (`PixelMaskIds` / `WFOV_PIX_MASK_ID_CMD_*`)

| Value | Name        | Description        |
| ----: | ----------- | ------------------ |
|     0 | FF          | Full frame         |
|     1 | NADIR       | Nadir stripe       |
|     2 | FORWARD     | Forward stripe     |
|     4 | CROSS       | Cross stripe       |
|     7 | STRIPE_ALL  | All stripes        |
|     8 | ADM         | ADM mask           |
|    15 | CROSS_TRACK | Cross-track stripe |
|    16 | LIMB        | Limb mask          |
|    32 | CUSTOM_0    | Custom mask slot 0 |
|    33 | CUSTOM_1    | Custom mask slot 1 |
|    34 | CUSTOM_2    | Custom mask slot 2 |
|    35 | CUSTOM_3    | Custom mask slot 3 |
|    64 | LOCK_ON     | Lock-on mask       |
|   128 | RAPS        | RAPS stripe        |
|   137 | RAPS_ADM    | RAPS + ADM         |

### Do not confuse with

| Concept                             | Where        | Meaning                                                                            |
| ----------------------------------- | ------------ | ---------------------------------------------------------------------------------- |
| **`pixel_mask_id`** (byte 2)        | FSW header   | Full mask type from `WFOV_CAM_START`                                               |
| **`bitmask_id`** (byte 1, bits 4–3) | FSW header   | Which custom bitmask **slot** (0–3) when using `CUSTOM_*`                          |
| **`PixelMasks` table slots**        | FSW internal | Table row IDs (0=NADIR, 1=FORWARD, …) used by the pixel task — different numbering |

For `CUSTOM_0`…`CUSTOM_3` commands, FSW sets both `pixel_mask_id` (32–35) and `bitmask_id` (0–3).

**Note:** After the first image in a multi-image sequence, FSW may clear `pixelMaskID` to 0 for subsequent frames (non-recalc masks). The header on later frames may show `pixel_mask_id = 0` even though the mask was applied on the first frame.

---

## img_mode

`img_mode` records the **FPGA `PXL_PROC_EN.SELECT`** register at trigger time. It describes the **processing path** for that exposure, not the `WFOV_CAM_START` `mode` argument directly.

### Header values (`ImageMode` enum)

| Value | Name  | FPGA select                                                   |
| ----: | ----- | ------------------------------------------------------------- |
|     0 | DUAL  | Dual-image SRAM path                                          |
|     1 | VIDEO | Video / quick-look path                                       |
|     2 | IMGA  | Image A SRAM (standard science still)                         |
|     3 | IMGB  | Image B SRAM (defined in HW; rarely set by FSW in normal ops) |

Decode: `img_mode = (second_byte >> 1) & 0x03`

### Command `mode` vs header `img_mode`

`WFOV_CAM_START` `mode` uses a **different** enum (`WFOV_PROCESS_MODE_*`):

| Command `mode` | Constant | Maps to header `img_mode` |
| -------------: | -------- | ------------------------- |
|              0 | SINGLE   | **2** (IMGA)              |
|              1 | DUAL     | **0** (DUAL)              |
|              2 | VIDEO    | **1** (VIDEO)             |

Mapping is done in `WFOVManager::setImgMode()`.

### videoNum / videoSkip interleave

Even when command `mode` is SINGLE or DUAL, `videoNum` and `videoSkip` can force **VIDEO** select on individual frames within a sequence. Expect `img_mode == 1` on those frames only.

### Not the same as FPGA `readout`

The FPGA header field `readout` (sensor readout SINGLE/DUAL) is independent of FSW `img_mode`. Do not use `readout` to infer command `mode` or VIDEO pairing.

---

## Images per timestamp

| Situation                                         | Header `img_mode`              | NAND blobs per camera trigger          | Same FSW timestamp?             |
| ------------------------------------------------- | ------------------------------ | -------------------------------------- | ------------------------------- |
| `mode SINGLE`, `videoNum=0`                       | 2 (IMGA)                       | **1**                                  | Unique                          |
| `mode DUAL`                                       | 0 (DUAL)                       | **1** (combined ~2× payload)           | Unique                          |
| `mode VIDEO`, or VIDEO forced by `videoNum`       | 1 (VIDEO)                      | **2**                                  | **Duplicate**                   |
| `mode SINGLE` + `videoNum > 0` on selected frames | 1 on video frames, 2 on others | **2** on video frames, **1** otherwise | Duplicates only on video frames |

### VIDEO double-write behavior

When `PXL_PROC_EN.SELECT == VIDEO`, the FPGA (and ops-sim model) writes **two separate NAND images** per camera trigger:

1. **First write:** bitmask processing **disabled** → full-frame image
2. **Second write:** normal bitmask processing → masked/stripe image

Both writes snapshot the **same** `IMG_META_*` registers from the single `acquire()` call, so FSW header fields including `timestamp_seconds`, `timestamp_subseconds`, and `img_mode` are **identical** on both blobs.

Reference: libfsw `WFOVModel_Processor.cpp` `processNewImage()`.

---

## Separating duplicate-timestamp pairs

`CAMERA_TIME` (derived from FSW `timestamp_seconds` + `timestamp_subseconds`) is **not a unique image key** when VIDEO processing ran. Use a composite key instead.

### Primary discriminator: packet order

For duplicate `CAMERA_TIME` rows with `img_mode == 1`:

| Order in packet stream                      | Product                           |
| ------------------------------------------- | --------------------------------- |
| **Lower** `CAMERA_PACKET_INDEX` (first SOP) | Full-frame (bitmask bypass write) |
| **Higher** `CAMERA_PACKET_INDEX` (next SOP) | Masked / stripe image             |

Recommended unique key: **`(CAMERA_TIME, img_mode, CAMERA_PACKET_INDEX)`** or ordering by packet index alone within duplicate-time groups.

In libera_cam, use L1B `Camera_Packet_Index` (from L1A `CAMERA_PACKET_INDEX`) the same way.

### Secondary validation: image content

After decompressing the JPEG-LS payload:

- **Full frame:** large fraction of pixels > 0
- **Masked frame:** most pixels zero outside the active stripe/mask

libera_cam exposes `valid_pixel_mask` (`image_data > 0`) and per-pixel `integration_mask` (13th bit of JPEG-LS payload). Use these to confirm packet-order assignment.

### Fields that do NOT distinguish the VIDEO pair

These are typically **identical** on both blobs from the same trigger:

- `timestamp_seconds` / `timestamp_subseconds` (defines `CAMERA_TIME`)
- `img_mode` (both are 1 = VIDEO)
- `image_count`, `image_total`, `cadence`
- `pixel_mask_id` (may already be 0 on later sequence frames)
- `bitmask_disable` in the FSW header (reflects register at trigger, not per-write path)

`flash_write_pointer` may differ between the two blobs on flight hardware; verify on your dataset (ops sim may show the same value).

### Recommended logical streams

| Stream           | Selection rule                                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------------------------------- |
| **ScienceStill** | `img_mode ∈ {0, 2, 3}` **plus** the **masked** member of each VIDEO pair (`img_mode == 1`, higher packet index) |
| **VideoFull**    | **Full-frame** member of each VIDEO pair only (`img_mode == 1`, lower packet index)                             |

For NetCDF or xarray coordinates that require unique datetime labels, either:

- Use **`CAMERA_PACKET_INDEX`** (or a composite index) as the coordinate instead of raw `CAMERA_TIME`, or
- Split into separate datasets with distinct coordinate names (`SCIENCE_TIME`, `VIDEO_TIME`), each monotonic in packet order, or
- Apply synthetic sub-microsecond offsets within duplicate groups (document as logical, not physical time)

---

## L1A time coordinates

WFOV SCI carries two independent times:

| Coordinate         | Source                                          | Use                            |
| ------------------ | ----------------------------------------------- | ------------------------------ |
| `PACKET_ICIE_TIME` | CCSDS secondary header on every mem-dump packet | Packet ordering, deduplication |
| `CAMERA_TIME`      | FSW `timestamp_*` from each qualifying SOP      | Per-image acquisition metadata |

One `CAMERA_TIME` row is added per qualifying SOP (`flags == SOP` and `offset == 0`), in **packet stream order** (not sorted by acquisition time).

See libera_utils `doc/source/user-docs/l1a_processing.md` (WFOV camera science section) and product definition `icie_wfov_sci_l1a.yml`.

---

## Exposure timing

WFOV stores several related exposure quantities: FSW **commanded** times, FPGA **actual**
integration registers, FPGA **DELTA_EXP** between dual-frame requests, and a per-pixel
**13th-bit mask** in the JPEG-LS payload. L1A keeps commanded and FPGA fields as metadata on
`CAMERA_TIME`. libera_cam converts the FPGA actual and DELTA registers to milliseconds at
L1B and writes them on the CAM product.

### Clock constant

FPGA timing registers use a fixed clock period (µs per raw count):

```text
WFOV_DEFAULT_CLK_PER_VALUE = 0.15625   # microseconds per register count
```

Defined in `libera_cam.constants.WFOV_DEFAULT_CLK_PER_VALUE` and used by
`libera_cam.image_parsing.exposure`. The same `0.15625` factor appears in the L1A product
definition comments for `WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_*`.

### FSW commanded exposure (`commanded_exp_time_1` / `_2`)

| Layer     | Names                                                                          | Units        |
| --------- | ------------------------------------------------------------------------------ | ------------ |
| FSW / L1A | `WFOV_FSW_HEADER_COMMANDED_EXP_TIME_1`, `WFOV_FSW_HEADER_COMMANDED_EXP_TIME_2` | milliseconds |
| L1B       | _not written_ (commanded times are not passed through to CAM L1B)              | —            |

These are the commanded exposures packed into the FSW header. L1B science uses the FPGA
**actual** registers (and the per-pixel mask) instead.

### FPGA actual exposure (`actual_exp_time_1` / `_2`)

L1A variables `WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_1` and `WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_2` are **raw
counts**. Convert to milliseconds with:

```text
ms = (value + 0.43 * 20) * 129.0 * 0.15625 / 1000
```

Equivalently:

```text
ms = (value + 8.6) * 129.0 * WFOV_DEFAULT_CLK_PER_VALUE / 1000
```

| Symbol / factor | Role                                    |
| --------------- | --------------------------------------- |
| `value`         | FPGA integration-time register (raw)    |
| `0.43 * 20`     | Fixed register offset (8.6)             |
| `129.0`         | Line / readout timing factor            |
| `0.15625`       | `WFOV_DEFAULT_CLK_PER_VALUE` (µs/count) |
| `/ 1000`        | Convert microseconds to milliseconds    |

Implementation: `actual_exposure_counts_to_ms()` in
`libera_cam/image_parsing/exposure.py`.

L1B outputs:

| L1B variable             | Source                                           | Units        |
| ------------------------ | ------------------------------------------------ | ------------ |
| `Actual_Exposure_Time_1` | `WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_1` + equation | milliseconds |
| `Actual_Exposure_Time_2` | `WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_2` + equation | milliseconds |

`Actual_Exposure_Time_2` is the second integration in DUAL / interleaved readout (odd-row
or second-frame exposure). In DUAL `img_mode`, `CAMERA_TIME` remains the **first**
integration time; the second exposure typically lags by about **111–350 ms** and is **not**
a separate L1A time coordinate.

### FPGA DELTA_EXP (`WFOV_IMAGE_HEADER_DELTA`)

`WFOV_IMAGE_HEADER_DELTA` is the FPGA DELTA_EXP between two frame requests in dual exposure
(exposure step in multi-frame readout). It is stored as **raw counts** at L1A. Convert
with the clock period only (do **not** apply the actual-exposure equation):

```text
delta_us = value * WFOV_DEFAULT_CLK_PER_VALUE
delta_ms = delta_us / 1000
```

or:

```text
ms = value * 0.15625 / 1000
```

Implementation: `delta_exposure_counts_to_ms()` in
`libera_cam/image_parsing/exposure.py`.

| L1B variable     | Source                    | Units        |
| ---------------- | ------------------------- | ------------ |
| `Exposure_Delta` | `WFOV_IMAGE_HEADER_DELTA` | milliseconds |

### Per-pixel exposure mask (13th bit)

Each decompressed pixel is 13 bits:

| Bits | Meaning                               |
| ---- | ------------------------------------- |
| 0–11 | Science DN (12-bit)                   |
| 12   | Integration-time flag (short vs long) |

After JPEG-LS decompress, libera_cam splits:

```text
image_12bit       = raw & 0x0FFF
integration_mask  = (raw >> 12) & 0x0001
```

| Layer    | Name                    | Meaning                    |
| -------- | ----------------------- | -------------------------- |
| Internal | `integration_mask`      | uint8, 0 = short, 1 = long |
| L1B      | `Integration_Time_Flag` | Same per-pixel flag        |

Radiometric calibration uses this mask (`convert_dn_to_radiance`). Frame-level
`Actual_Exposure_Time_1` / `_2` give the instrument register times for the two integrations.
Mapping each mask class to the correct SPICE time for geolocation is unresolved.

### Summary: L1A → L1B exposure fields

| L1A                                    | Conversion                               | L1B                      |
| -------------------------------------- | ---------------------------------------- | ------------------------ |
| `WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_1`  | `(v + 0.43*20) * 129.0 * 0.15625 / 1000` | `Actual_Exposure_Time_1` |
| `WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_2`  | same                                     | `Actual_Exposure_Time_2` |
| `WFOV_IMAGE_HEADER_DELTA`              | `v * 0.15625 / 1000`                     | `Exposure_Delta`         |
| JPEG-LS bit 12                         | decompress + bit split                   | `Integration_Time_Flag`  |
| `WFOV_FSW_HEADER_COMMANDED_EXP_TIME_*` | already ms at L1A                        | (not written to L1B)     |

---

## Cross-references

| Component                           | Location                                                                   |
| ----------------------------------- | -------------------------------------------------------------------------- |
| Canonical FSW/FPGA decoder          | `libera_utils/libera_utils/l1a/wfov_image_metadata.py`                     |
| L1B JPEG-LS decompress + mask split | `libera_cam/image_parsing/l1a_parser.py` / `read_l1a_cam_data.py`          |
| L1B exposure count → ms conversions | `libera_cam/image_parsing/exposure.py`                                     |
| Clock period constant               | `libera_cam.constants.WFOV_DEFAULT_CLK_PER_VALUE`                          |
| L1A stitch + image plane            | `libera_utils/libera_utils/l1a/wfov_image_metadata.py`                     |
| Hydra / FITS decoder                | `libctdb/libfsw/projects/icie/Python/process_wfov_sci.py`                  |
| FSW header writer                   | `libfsw/modules/wfov/src/WFOVManager.cpp` (`acquire`)                      |
| FSW struct definition               | `libfsw/modules/wfov/src/WFOVPkt.h` (`wfovImgMetaData_t`)                  |
| FPGA select constants               | `libfsw/projects/icie/src/include/hal/wfov.h`                              |
| Pixel mask command values           | `libctdb/libfsw/projects/icie/Python/constants.py`                         |
| L1A product variables               | `libera_utils/libera_utils/data/product_definitions/icie_wfov_sci_l1a.yml` |

---

## Out of scope

This document describes metadata layout, exposure-timing conversions, and operational
guidance. It does **not** cover:

- Automatic VIDEO stream splitting in the L1A or L1B pipeline
- New L1A product variables (e.g. `WFOV_IMAGE_STREAM`)
- Per-pixel / dual-mask geolocation time assignment
- Full multi-packet image stitching — performed in libera_utils
  (`enhance_wfov_l1a_dataset`); libera_cam reads `WFOV_COMPRESSED_IMAGE` directly
