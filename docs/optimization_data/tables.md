# 自动重算的数据表

由 `tools/build_optimization_data.py` 从原始 CSV/JSON 生成。

## 当前版本：五次完整运行的中位数

同一形状才可比较；跨 Step 是完整配置观察，不是单因素实验。单位 ms。

| Step | 1024³ | 2048³ | 4096³ | 8192³ |
|---|---:|---:|---:|---:|
| 3 | 0.016562 | 0.053451 | — | — |
| 4 | 0.012457 | 0.020810 | — | — |
| 5 | 0.010426 | 0.019911 | 0.150484 | — |
| 6 | 0.008419 | 0.024813 | 0.155573 | 1.339871 |
| 7 | 0.008445 | 0.022275 | 0.142769 | 1.277727 |
| 8 | 0.008373 | 0.018337 | 0.118931 | 1.013805 |
| 9 | 0.012417 | 0.016772 | 0.098673 | 0.702735 |
| 10 | 0.010830 | 0.016688 | 0.094295 | 0.644471 |

## 已归档独立实验

耗时下降为逐对 `(1 - 候选/对照)` 的中位数，负数表示变慢。

| id | 对照→候选中位 ms | 配对加速 | 耗时下降 | 更快轮数 | 候选先测 | 位置完全平衡 | 原始数据 |
|---|---|---:|---:|---:|---:|---|---|
| s4_permit | 0.076641 → 0.043150 | 1.775853× | +43.689% | 5/5 | 1/5 | 否 | [JSON](../../results_b300/step45_probe.PS9CFi/probe/samples.json) |
| s4_k128 | 0.022709 → 0.018613 | 1.220287× | +18.052% | 5/5 | 1/5 | 否 | [JSON](../../results_b300/mma64_step4.dh9ZC8/probe/samples.json) |
| s5_permit | 0.045228 → 0.033455 | 1.351900× | +26.030% | 5/5 | 2/5 | 否 | [JSON](../../results_b300/step45_probe.PS9CFi/probe/samples.json) |
| s5_mma_wait | 0.016515 → 0.014513 | 1.137219× | +12.066% | 5/5 | 2/5 | 否 | [JSON](../../results_b300/wait1024.UP24Tv/probe/samples.json) |
| s5_poll | 0.016515 → 0.017083 | 0.967414× | -3.368% | 0/5 | 2/5 | 否 | [JSON](../../results_b300/wait1024.UP24Tv/probe/samples.json) |
| s6_k128 | 0.316145 → 0.249217 | 1.267565× | +21.109% | 5/5 | 1/5 | 否 | [JSON](../../results_b300/persistent_probe.VB42kg/probe/samples.json) |
| s6_fence | 0.316145 → 0.314058 | 1.006647× | +0.660% | 5/5 | 1/5 | 否 | [JSON](../../results_b300/persistent_probe.VB42kg/probe/samples.json) |
| s7_k128 | 0.309405 → 0.219035 | 1.412584× | +29.208% | 5/5 | 2/5 | 否 | [JSON](../../results_b300/persistent_probe.VB42kg/probe/samples.json) |
| s7_epi128 | 0.309405 → 0.310886 | 0.994876× | -0.515% | 0/5 | 2/5 | 否 | [JSON](../../results_b300/persistent_probe.VB42kg/probe/samples.json) |
| s8_tma_wait | 0.029859 → 0.029245 | 1.018372× | +1.804% | 5/5 | 2/5 | 否 | [JSON](../../results_b300/step810_probe.Wl6HTg/step08/samples.json) |
| s8_cache | 0.029491 → 0.028849 | 1.024379× | +2.380% | 4/5 | 2/5 | 否 | [JSON](../../results_b300/profile_guided.6KUfDZ/step08/samples.json) |
| s9_cache4096 | 0.144269 → 0.142462 | 1.012452× | +1.230% | 7/7 | 3/7 | 否 | [JSON](../../results_b300/step9_cache.EWrE9G/step9_4096/samples.json) |
| s9_cache8192 | 0.912429 → 0.895060 | 1.019953× | +1.956% | 7/7 | 3/7 | 否 | [JSON](../../results_b300/step9_cache.BBJEQB/step9_8192/samples.json) |
| s10_cache | 0.142305 → 0.140656 | 1.010485× | +1.038% | 5/5 | 1/5 | 否 | [JSON](../../results_b300/profile_guided.6KUfDZ/step10/samples.json) |
| s10_balance | 0.139632 → 0.137918 | 1.012212× | +1.206% | 5/5 | 3/5 | 否 | [JSON](../../results_b300/step10_fused_a.1VXYz2/step10/samples.json) |
| s10_fused_a | 0.137918 → 0.138321 | 0.995966× | -0.405% | 0/5 | 1/5 | 否 | [JSON](../../results_b300/step10_fused_a.1VXYz2/step10/samples.json) |
| s10_b_first | 0.138356 → 0.137891 | 1.004687× | +0.466% | 5/5 | 3/5 | 否 | [JSON](../../results_b300/step10_roles.rx5lNL/step10/samples.json) |
| s10_n128_1024 | 0.027389 → 0.018566 | 1.475045× | +32.205% | 7/7 | 3/7 | 否 | [JSON](../../results_b300/step10_tmem_sizes.I9nGIJ/step10_1024/samples.json) |
| s10_n128_2048 | 0.043305 → 0.027029 | 1.606506× | +37.753% | 7/7 | 3/7 | 否 | [JSON](../../results_b300/step10_tmem_sizes.I9nGIJ/step10_2048/samples.json) |
| s10_n128_only4096 | 0.138129 → 0.140209 | 0.983632× | -1.664% | 0/7 | 1/7 | 否 | [JSON](../../results_b300/step10_tmem_sizes.I9nGIJ/step10_4096/samples.json) |
| s10_epi32_4096 | 0.140209 → 0.138861 | 1.010913× | +1.080% | 7/7 | 5/7 | 否 | [JSON](../../results_b300/step10_tmem_sizes.I9nGIJ/step10_4096/samples.json) |
| s10_double4096 | 0.138861 → 0.137422 | 1.009652× | +0.956% | 7/7 | 2/7 | 否 | [JSON](../../results_b300/step10_tmem_sizes.I9nGIJ/step10_4096/samples.json) |
| s10_combined4096 | 0.138129 → 0.137422 | 1.004499× | +0.448% | 7/7 | 2/7 | 否 | [JSON](../../results_b300/step10_tmem_sizes.I9nGIJ/step10_4096/samples.json) |
| s10_combined8192 | 0.868543 → 0.902522 | 0.962622× | -3.883% | 0/7 | 2/7 | 否 | [JSON](../../results_b300/step10_tmem_sizes.I9nGIJ/step10_8192/samples.json) |
| s10_k64_depth2 | 0.139238 → 0.224105 | 0.620542× | -61.149% | 0/7 | 1/7 | 否 | [JSON](../../results_b300/step10_input_ring.9SsKZM/step10/samples.json) |
| s10_k128_depth2 | 0.139238 → 0.166755 | 0.834347× | -19.854% | 0/7 | 3/7 | 否 | [JSON](../../results_b300/step10_input_ring.9SsKZM/step10/samples.json) |
| s10_depth5 | 0.139042 → 0.138689 | 1.003139× | +0.313% | 8/8 | 4/8 | 是 | [JSON](../../results_b300/step10_depth5_balanced.Fm3CcJ/step10_4096/samples.json) |
| s10_l2_4 | 0.139125 → 0.140091 | 0.993604× | -0.644% | 0/8 | 4/8 | 是 | [JSON](../../results_b300/step10_l2.nqNoLW/step10_4096/samples.json) |
| s10_l2_2 | 0.139125 → 0.139975 | 0.994571× | -0.546% | 0/8 | 4/8 | 是 | [JSON](../../results_b300/step10_l2.nqNoLW/step10_4096/samples.json) |
| s10_l2_1 | 0.139125 → 0.139717 | 0.995912× | -0.410% | 2/8 | 4/8 | 是 | [JSON](../../results_b300/step10_l2.nqNoLW/step10_4096/samples.json) |
| s10_maxclusters | 0.109001 → 0.113731 | 0.954677× | -4.748% | 0/8 | 4/8 | 是 | [JSON](../../results_b300/step10_geometry.PywrmG/step10_4096/samples.json) |
| s10_n64 | 0.113731 → 0.147771 | 0.775871× | -28.888% | 0/8 | 4/8 | 是 | [JSON](../../results_b300/step10_geometry.PywrmG/step10_4096/samples.json) |
| s10_n64depth5 | 0.147771 → 0.135395 | 1.098989× | +9.007% | 8/8 | 4/8 | 是 | [JSON](../../results_b300/step10_geometry.PywrmG/step10_4096/samples.json) |
| s10_n64depth5_total | 0.109001 → 0.135395 | 0.805065× | -24.217% | 0/8 | 4/8 | 是 | [JSON](../../results_b300/step10_geometry.PywrmG/step10_4096/samples.json) |
| s10_batch_fixed | 0.138352 → 0.138343 | 0.998207× | -0.180% | 2/5 | 3/5 | 否 | [JSON](../../results_b300/step10_mma_unroll4.iR07fS/step10/samples.json) |
| s10_shared_a | 0.112822 → 0.112003 | 1.004371× | +0.435% | 179/201 | 150/201 | 否 | [JSON](../../results_b300/step10_share_a_state.fGOjDy/step10/samples.json) |
| s10_epi64 | 0.139205 → 0.141452 | 0.984119× | -1.614% | 0/8 | 4/8 | 是 | [JSON](../../results_b300/step10_epilogue.xVWC2d/step10_4096/samples.json) |
| s10_epi128 | 0.139205 → 0.143390 | 0.971084× | -2.978% | 0/8 | 4/8 | 是 | [JSON](../../results_b300/step10_epilogue.xVWC2d/step10_4096/samples.json) |
| s10_epi32_double | 0.139205 → 0.140787 | 0.988920× | -1.120% | 0/8 | 4/8 | 是 | [JSON](../../results_b300/step10_epilogue.xVWC2d/step10_4096/samples.json) |
| s10_split_ready | 0.139018 → 0.138973 | 1.000492× | +0.049% | 5/8 | 4/8 | 是 | [JSON](../../results_b300/step10_ready.FAnkEQ/step10_4096/samples.json) |
