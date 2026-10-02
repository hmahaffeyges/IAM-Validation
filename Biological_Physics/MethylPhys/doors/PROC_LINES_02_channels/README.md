# IMR90 per-channel readings on the current method (development, 2026-10-02)

GSE48580 WGBS (Cruickshanks et al. 2013): proliferating x3, replicatively senescent x3, SV40-immortalised x3. Script `imr90_channels.py` (box job e0f561c0).
Each culture is read against the proliferating cultures. Sites: beta 0.75-0.95 (methylated channel) or 0.05-0.25 (unmethylated channel) on one
proliferating culture used only to choose the sites; the reference mean depth-corrected H from a different proliferating culture; every ordered pair of
proliferating cultures other than the specimen, averaged. Depth >= 10 in all nine.

| IMR90 | methylated channel | unmethylated channel | both |
|---|---|---|---|
| proliferating (held out) | 0.991-1.005 | 0.991-1.016 | 0.997-1.002 |
| senescent | 0.941-1.016 | 0.685-0.695 | 0.874-0.925 |
| SV40 | 1.078-1.120 | 0.587-0.664 | 0.965-0.975 |

1. Choosing the sites and computing the reference on the same cultures biases the reading: held-out proliferating cultures read 0.92-0.94 that way
   (first run, ebf9a2b4/7ab374ed). Selection and reference are therefore on different cultures.
2. Senescence: methylated channel at the healthy rate, unmethylated channel cleaner; combined below Normal.
3. Immortalisation: methylated channel more error, unmethylated cleaner; they cancel and the combined reading sits inside Normal. Channels are reported apart.
4. Not checked: a conversion difference between cultures on the unmethylated channel (conversion failure reads as methylation).
