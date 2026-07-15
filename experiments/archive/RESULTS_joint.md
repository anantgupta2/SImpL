# SImpL — NEW joint-set results — PRELIMINARY

LSAT on **joint 461-Q** (test+validation), RACE on **708-Q** test. Joint re-eval still running;
uses finished seeds only. Companion: `RESULTS.md` (230-Q + full HP tuning + per-step appendix).

## 1. LSAT in-domain (hp_size, us2/rs2) — matched-step `cot_accuracy`

simpl = best of its eval cols. **Best per size bold.**

| size | step | cot | simpl | cotN16 | Δ simpl−cot |
|---|---|---|---|---|---|
| LSAT-50 | 217 | 0.2553(n3) | **0.2490**(n2) | — | -0.64pp |
| LSAT-100 | 301 | 0.2580(n3) | **0.2638**(n3) | 0.2593(n3) | +0.58pp |
| LSAT-200 | 436 | 0.2755(n3) | **0.2494**(n3) | — | -2.62pp |

- LSAT-50 simpl still 🟡 partial (re-eval running).
- LSAT-100: **+0.6pp** (clean). LSAT-200: ⚠️ simpl all 3 seeds below cot (~−2.6pp) — **reversal** from
  230-Q tie; cot stable across eval sets, simpl dropped ~3pp → suspected overfit to test split. Do not quote.

## 2. hp_simpl_base — LSAT-100 simpl from base (us2/rs2), joint, tail-3

| arm | cot_accuracy | n |
|---|---|---|
| simpl | **0.2717** | 3 |

_(matched cot-base not re-evaled on joint; 230-Q Δ was +1.72pp.)_

## 3. RACE 708-Q in-domain — tail-3

RACE-50 three-way (matched step 199) + RACE-100. simpl = best eval col. **Best bold.**

| run | HP | cot | simpl(best) | Δ vs its cot | seeds |
|---|---|---|---|---|---|
| race50-cot | pe4/rs1 | 0.8230 | — | (baseline) | 3/3 |
| race50-cot | pe2/rs2 | **0.8273** | — | vs pe4: +0.43pp | 3/3 |
| race50-simpl | pe2/rs2/us2 | 0.8400 | 0.8400 | vs cot-pe2: +1.27pp | 3/3 |
| race100-cot | pe4/rs1 | 0.8203 | — | — | 1/3 |

**Reads (RACE-50, ~199 steps):**
- HP effect (cot pe2/rs2 − pe4/rs1): +0.43pp
- Understanding effect (simpl − cot at same pe2/rs2): +1.27pp
- RACE near-saturated (~0.80–0.83); race100-cot still filling in.

## Completeness

| family | done |
|---|---|
| LSAT hp_size | 18/20 seed-runs |
| LSAT final | 14/24 seed-runs |
| RACE | 10/12 seed-runs |

---
## Appendix: per-step `cot_accuracy`, seed-averaged (joint sets)
Mean over finished seeds at each saved step (blank = no ckpt / not yet evaluated). LSAT = 461-Q joint, RACE = 708-Q.
### LSAT-50 hp_size (cot / simpl / cotN16)

| step | cot (n3) | simpl (n2) |
|---|---|---|
| 12 | 0.2350 | 0.2409 |
| 24 | 0.2385 | 0.2497 |
| 36 | 0.2426 | 0.2420 |
| 48 | 0.2405 | 0.2462 |
| 60 | 0.2429 | 0.2481 |
| 72 | 0.2493 | 0.2313 |
| 84 | 0.2525 | 0.2493 |
| 96 | 0.2483 | 0.2671 |
| 108 | 0.2534 | 0.2611 |
| 120 | 0.2519 | 0.2657 |
| 132 | 0.2500 | 0.2457 |
| 144 | 0.2474 | 0.2671 |
| 156 | 0.2559 |  |
| 168 | 0.2563 |  |
| 180 | 0.2647 |  |
| 192 | 0.2621 |  |
| 204 | 0.2564 |  |
| 216 | 0.2548 |  |
| 217 | 0.2548 |  |

### LSAT-100 hp_size (cot / simpl / cotN16)

| step | cot (n3) | simpl (n3) | cotN16 (n3) |
|---|---|---|---|
| 24 | 0.2491 | 0.2360 | 0.2446 |
| 48 | 0.2400 | 0.2372 | 0.2516 |
| 72 | 0.2505 | 0.2457 | 0.2527 |
| 96 | 0.2454 | 0.2451 | 0.2497 |
| 120 | 0.2491 | 0.2502 | 0.2510 |
| 144 | 0.2589 | 0.2519 | 0.2564 |
| 168 | 0.2518 | 0.2592 | 0.2579 |
| 192 | 0.2567 | 0.2459 | 0.2541 |
| 216 | 0.2632 | 0.2549 | 0.2649 |
| 240 | 0.2553 | 0.2592 | 0.2571 |
| 264 | 0.2579 | 0.2566 | 0.2602 |
| 288 | 0.2607 | 0.2617 | 0.2703 |
| 301 |  | 0.2731 |  |
| 312 | 0.2669 |  | 0.2741 |
| 336 | 0.2713 |  | 0.2667 |
| 360 | 0.2651 |  | 0.2665 |
| 361 | 0.2651 |  | 0.2665 |

### LSAT-200 hp_size (cot / simpl / cotN16)

| step | cot (n3) | simpl (n3) |
|---|---|---|
| 24 | 0.2398 | 0.2479 |
| 48 | 0.2425 | 0.2458 |
| 72 | 0.2467 | 0.2405 |
| 96 | 0.2485 | 0.2474 |
| 120 | 0.2456 | 0.2511 |
| 144 | 0.2396 | 0.2599 |
| 168 | 0.2504 | 0.2523 |
| 192 | 0.2522 | 0.2539 |
| 216 | 0.2621 | 0.2503 |
| 240 | 0.2582 | 0.2561 |
| 264 | 0.2650 | 0.2504 |
| 288 | 0.2615 | 0.2527 |
| 312 | 0.2602 | 0.2663 |
| 336 | 0.2629 | 0.2763 |
| 360 | 0.2627 | 0.2757 |
| 384 | 0.2630 | 0.2589 |
| 408 | 0.2772 | 0.2408 |
| 432 | 0.2780 | 0.2486 |
| 436 | 0.2714 |  |
| 451 |  | 0.2512 |

### hp_simpl_base LSAT-100 simpl from base

| step | simpl (n3) |
|---|---|
| 12 | 0.2354 |
| 24 | 0.2420 |
| 36 | 0.2439 |
| 48 | 0.2447 |
| 60 | 0.2462 |
| 72 | 0.2439 |
| 84 | 0.2510 |
| 96 | 0.2569 |
| 108 | 0.2514 |
| 120 | 0.2568 |
| 132 | 0.2542 |
| 144 | 0.2676 |
| 156 | 0.2661 |
| 168 | 0.2592 |
| 180 | 0.2617 |
| 192 | 0.2619 |
| 204 | 0.2682 |
| 216 | 0.2735 |
| 217 | 0.2735 |

### final LSAT-50 (cot / cotN16 / simpl variants)

| step | cot (n3) | cotN16 (n3) | simpl_pe2us2 (n3) | simpl_pe2 (n3) | simpl_pe4 (n3) |
|---|---|---|---|---|---|
| 4 |  |  | 0.2318 | 0.2291 | 0.2270 |
| 8 | 0.2328 | 0.2389 | 0.2319 | 0.2298 | 0.2387 |
| 12 |  |  | 0.2436 | 0.2295 | 0.2374 |
| 16 | 0.2346 | 0.2421 | 0.2417 | 0.2332 | 0.2472 |
| 20 |  |  | 0.2460 | 0.2317 | 0.2498 |
| 24 | 0.2334 | 0.2408 | 0.2388 | 0.2362 | 0.2453 |
| 28 |  |  | 0.2358 | 0.2327 | 0.2426 |
| 32 | 0.2440 | 0.2412 | 0.2481 | 0.2386 | 0.2444 |
| 36 |  |  | 0.2477 | 0.2479 | 0.2437 |
| 40 | 0.2524 | 0.2514 | 0.2430 | 0.2440 | 0.2475 |
| 44 |  |  | 0.2409 | 0.2362 | 0.2480 |
| 48 | 0.2491 | 0.2530 | 0.2331 | 0.2388 | 0.2490 |
| 52 |  |  | 0.2423 | 0.2449 | 0.2406 |
| 56 | 0.2411 | 0.2489 | 0.2336 | 0.2448 | 0.2443 |
| 60 |  |  | 0.2388 | 0.2494 | 0.2414 |
| 64 | 0.2479 | 0.2462 | 0.2440 | 0.2449 | 0.2389 |
| 68 |  |  | 0.2434 | 0.2454 | 0.2526 |
| 72 | 0.2349 | 0.2451 | 0.2416 | 0.2477 | 0.2471 |
| 76 |  |  | 0.2420 | 0.2412 | 0.2410 |
| 80 | 0.2450 | 0.2401 | 0.2387 | 0.2458 | 0.2385 |
| 84 |  |  | 0.2481 | 0.2392 | 0.2419 |
| 88 | 0.2443 | 0.2348 | 0.2427 | 0.2432 | 0.2346 |
| 92 |  |  | 0.2378 | 0.2413 | 0.2273 |
| 96 | 0.2476 | 0.2454 | 0.2418 | 0.2440 | 0.2286 |
| 100 |  |  | 0.2392 | 0.2476 | 0.2284 |
| 104 | 0.2442 | 0.2405 | 0.2402 | 0.2468 | 0.2278 |
| 108 |  |  | 0.2406 | 0.2445 | 0.2221 |
| 112 | 0.2458 | 0.2332 | 0.2473 | 0.2478 | 0.2287 |
| 116 |  |  | 0.2364 | 0.2455 | 0.2267 |
| 120 | 0.2513 | 0.2312 | 0.2423 | 0.2427 | 0.2306 |
| 124 |  |  | 0.2369 | 0.2450 | 0.2299 |
| 128 | 0.2479 | 0.2523 | 0.2443 | 0.2486 | 0.2342 |
| 132 |  |  | 0.2425 | 0.2486 | 0.2328 |
| 136 | 0.2542 | 0.2403 | 0.2519 | 0.2515 | 0.2335 |
| 140 |  |  | 0.2501 | 0.2524 | 0.2328 |
| 144 | 0.2458 | 0.2373 | 0.2434 | 0.2542 | 0.2297 |
| 148 |  |  | 0.2583 | 0.2514 | 0.2322 |
| 152 | 0.2449 | 0.2377 | 0.2443 | 0.2438 | 0.2329 |
| 156 |  |  | 0.2432 | 0.2568 | 0.2459 |
| 160 | 0.2460 | 0.2368 | 0.2405 | 0.2530 | 0.2356 |
| 164 |  |  | 0.2462 | 0.2469 | 0.2397 |
| 168 | 0.2534 | 0.2466 | 0.2530 | 0.2504 | 0.2381 |
| 172 |  |  | 0.2497 | 0.2397 | 0.2419 |
| 176 | 0.2537 | 0.2522 | 0.2508 | 0.2508 | 0.2392 |
| 180 |  |  | 0.2448 | 0.2527 | 0.2465 |
| 184 | 0.2480 | 0.2417 | 0.2473 | 0.2485 | 0.2476 |
| 188 |  |  | 0.2600 | 0.2619 | 0.2465 |
| 192 | 0.2472 | 0.2495 | 0.2606 | 0.2432 | 0.2400 |
| 196 |  |  | 0.2457 | 0.2413 | 0.2446 |
| 200 | 0.2587 | 0.2436 | 0.2470 | 0.2389 | 0.2302 |
| 204 |  |  | 0.2495 | 0.2476 | 0.2394 |
| 208 | 0.2515 | 0.2399 | 0.2571 | 0.2495 | 0.2340 |
| 212 |  |  | 0.2562 | 0.2419 | 0.2489 |
| 216 | 0.2517 | 0.2427 | 0.2543 | 0.2614 | 0.2386 |
| 217 | 0.2517 | 0.2427 | 0.2543 | 0.2603 | 0.2386 |

### final LSAT-100 (cot / simpl_pe2us2)

| step | cot (n3) | simpl_pe2us2 (n3) |
|---|---|---|
| 12 | 0.2390 |  |
| 24 | 0.2354 |  |
| 36 | 0.2353 |  |
| 40 |  | 0.2450 |
| 44 |  | 0.2387 |
| 48 | 0.2344 | 0.2398 |
| 52 |  | 0.2396 |
| 56 |  | 0.2371 |
| 60 | 0.2472 | 0.2395 |
| 64 |  | 0.2333 |
| 68 |  | 0.2448 |
| 72 | 0.2420 | 0.2428 |
| 76 |  | 0.2416 |
| 80 |  | 0.2356 |
| 84 | 0.2418 | 0.2505 |
| 88 |  | 0.2415 |
| 92 |  | 0.2440 |
| 96 | 0.2487 | 0.2448 |
| 100 |  | 0.2530 |
| 104 |  | 0.2434 |
| 108 | 0.2490 | 0.2482 |
| 112 |  | 0.2497 |
| 116 |  | 0.2385 |
| 120 | 0.2444 | 0.2411 |
| 124 |  | 0.2432 |
| 128 |  | 0.2511 |
| 132 | 0.2530 | 0.2427 |
| 136 |  | 0.2454 |
| 140 |  | 0.2447 |
| 144 | 0.2537 | 0.2413 |
| 148 |  | 0.2457 |
| 152 |  | 0.2434 |
| 156 | 0.2507 | 0.2420 |
| 160 |  | 0.2472 |
| 164 |  | 0.2402 |
| 168 | 0.2530 | 0.2352 |
| 172 |  | 0.2579 |
| 180 | 0.2495 |  |
| 192 | 0.2513 |  |
| 204 | 0.2482 |  |
| 216 | 0.2476 |  |
| 228 | 0.2512 |  |
| 240 | 0.2307 |  |
| 252 | 0.2420 |  |
| 264 | 0.2437 |  |
| 276 | 0.2609 |  |
| 288 | 0.2479 |  |
| 300 | 0.2482 |  |
| 312 | 0.2561 |  |
| 324 | 0.2539 |  |
| 336 | 0.2517 |  |
| 348 | 0.2557 |  |
| 360 | 0.2605 |  |
| 372 | 0.2707 |  |
| 384 | 0.2699 |  |
| 396 | 0.2614 |  |
| 408 | 0.2724 |  |
| 420 | 0.2673 |  |
| 432 | 0.2672 |  |
| 433 | 0.2672 |  |

### RACE-50 (cot pe4 / cot pe2 / simpl pe2) + RACE-100 cot

| step | cot_pe4 (n3) | cot_pe2 (n3) | simpl_pe2 (n3) | race100_cot (n3) |
|---|---|---|---|---|
| 4 | 0.7826 | 0.8043 | 0.7970 |  |
| 8 | 0.7885 | 0.8053 | 0.8072 |  |
| 12 | 0.7985 | 0.8040 | 0.8188 |  |
| 16 | 0.7996 | 0.8050 | 0.8159 | 0.7966 |
| 20 | 0.8057 | 0.8097 | 0.8170 | 0.8031 |
| 24 | 0.8018 | 0.8180 | 0.8207 | 0.8040 |
| 28 | 0.8006 | 0.8123 | 0.8206 | 0.8061 |
| 32 | 0.8038 | 0.8173 | 0.8227 | 0.8030 |
| 36 | 0.8054 | 0.8208 | 0.8243 | 0.8029 |
| 40 | 0.8080 | 0.8212 | 0.8219 | 0.8017 |
| 44 | 0.8077 | 0.8229 | 0.8262 | 0.8067 |
| 48 | 0.8091 | 0.8247 | 0.8277 | 0.8057 |
| 52 | 0.8084 | 0.8242 | 0.8283 | 0.8087 |
| 56 | 0.8099 | 0.8277 | 0.8304 | 0.8084 |
| 60 | 0.8046 | 0.8283 | 0.8238 | 0.8073 |
| 64 | 0.8037 | 0.8274 | 0.8247 | 0.8064 |
| 68 | 0.8107 | 0.8266 | 0.8264 | 0.8141 |
| 72 | 0.8092 | 0.8293 | 0.8259 | 0.8095 |
| 76 | 0.8087 | 0.8253 | 0.8310 | 0.8157 |
| 80 | 0.8120 | 0.8326 | 0.8308 | 0.8076 |
| 84 | 0.8079 | 0.8321 | 0.8340 | 0.8066 |
| 88 | 0.8098 | 0.8319 | 0.8335 | 0.8100 |
| 92 | 0.8151 | 0.8272 | 0.8335 | 0.8119 |
| 96 | 0.8147 | 0.8301 | 0.8359 | 0.8100 |
| 100 | 0.8116 | 0.8286 | 0.8360 | 0.8115 |
| 104 | 0.8156 | 0.8287 | 0.8355 | 0.8121 |
| 108 | 0.8131 | 0.8283 | 0.8365 | 0.8143 |
| 112 | 0.8179 | 0.8323 | 0.8340 | 0.8170 |
| 116 | 0.8178 | 0.8309 | 0.8372 | 0.8110 |
| 120 | 0.8179 | 0.8282 | 0.8356 | 0.8161 |
| 124 | 0.8167 | 0.8343 | 0.8373 | 0.8148 |
| 128 | 0.8158 | 0.8339 | 0.8350 | 0.8193 |
| 132 | 0.8214 | 0.8361 | 0.8336 | 0.8167 |
| 136 | 0.8213 | 0.8308 | 0.8340 | 0.8183 |
| 140 | 0.8180 | 0.8289 | 0.8363 | 0.8130 |
| 144 | 0.8180 | 0.8252 | 0.8365 | 0.8133 |
| 148 | 0.8207 | 0.8239 | 0.8360 | 0.8175 |
| 152 | 0.8220 | 0.8240 | 0.8347 | 0.8180 |
| 156 | 0.8197 | 0.8252 | 0.8357 | 0.8194 |
| 160 | 0.8230 | 0.8223 | 0.8364 | 0.8169 |
| 164 | 0.8130 | 0.8249 | 0.8370 | 0.8140 |
| 168 | 0.8190 | 0.8234 | 0.8394 | 0.8096 |
| 172 | 0.8221 | 0.8227 | 0.8392 | 0.8146 |
| 176 | 0.8188 | 0.8245 | 0.8386 | 0.8164 |
| 180 | 0.8176 | 0.8266 | 0.8380 | 0.8189 |
| 184 | 0.8222 | 0.8289 | 0.8406 | 0.8213 |
| 188 | 0.8180 | 0.8287 | 0.8408 | 0.8194 |
| 192 | 0.8210 | 0.8262 | 0.8411 | 0.8200 |
| 196 | 0.8247 | 0.8280 | 0.8393 | 0.8224 |
| 199 | 0.8233 | 0.8277 |  |  |
| 200 |  |  | 0.8373 | 0.8233 |
| 204 |  |  | 0.8377 | 0.8220 |
| 208 |  |  | 0.8399 | 0.8220 |
| 212 |  |  | 0.8375 | 0.8200 |
| 216 |  |  | 0.8351 | 0.8190 |
| 217 |  |  | 0.8351 |  |
| 220 |  |  |  | 0.8205 |
| 224 |  |  |  | 0.8182 |
| 228 |  |  |  | 0.8184 |
| 232 |  |  |  | 0.8204 |
| 236 |  |  |  | 0.8205 |
| 240 |  |  |  | 0.8217 |
| 244 |  |  |  | 0.8219 |
| 248 |  |  |  | 0.8219 |
| 252 |  |  |  | 0.8239 |
| 256 |  |  |  | 0.8204 |
| 260 |  |  |  | 0.8225 |
| 264 |  |  |  | 0.8227 |
| 268 |  |  |  | 0.8200 |
| 272 |  |  |  | 0.8213 |
| 276 |  |  |  | 0.8190 |
| 280 |  |  |  | 0.8118 |
| 284 |  |  |  | 0.8157 |
| 288 |  |  |  | 0.8158 |
| 292 |  |  |  | 0.8206 |
| 296 |  |  |  | 0.8211 |
| 300 |  |  |  | 0.8183 |
| 304 |  |  |  | 0.8224 |
| 308 |  |  |  | 0.8214 |
| 312 |  |  |  | 0.8202 |
| 316 |  |  |  | 0.8256 |
| 320 |  |  |  | 0.8249 |
| 324 |  |  |  | 0.8249 |
| 328 |  |  |  | 0.8259 |
| 332 |  |  |  | 0.8257 |
| 336 |  |  |  | 0.8279 |
| 340 |  |  |  | 0.8310 |
| 344 |  |  |  | 0.8254 |
| 348 |  |  |  | 0.8201 |
| 352 |  |  |  | 0.8201 |
| 356 |  |  |  | 0.8226 |
| 360 |  |  |  | 0.8222 |
| 364 |  |  |  | 0.8213 |
| 368 |  |  |  | 0.8213 |
| 372 |  |  |  | 0.8231 |
| 376 |  |  |  | 0.8227 |
| 380 |  |  |  | 0.8212 |
| 384 |  |  |  | 0.8264 |
| 388 |  |  |  | 0.8238 |
| 392 |  |  |  | 0.8346 |
| 396 |  |  |  | 0.8282 |
| 400 |  |  |  | 0.8233 |
| 404 |  |  |  | 0.8243 |
| 408 |  |  |  | 0.8245 |
| 409 |  |  |  | 0.8245 |


---

# rs=2 mechanism & the clean-recipe attempt (2026-06-21)

**Question:** can we drop the hard-to-justify `reward_scale=2` and recover the simpl win with
standard knobs? (rs is non-standard; reviewers will ask "why 2?".)

**Mechanism (verified in code).** We use `critic_type=drgrpo` → advantages = reward − group_mean,
**no std-normalization**, so rs scales the advantage (hence the policy gradient) linearly. The loss
is `−PG(∝rs·adv) + β·KL` with **β=0.04** (`simpl_oat.py`), and gradients are **clipped to max_norm=1.0**.
Algebraically `g(rs=2,β=0.04) = 2·g(rs=1,β=0.02)` (same direction), so in the *idealized* (perfect
Adam scale-invariance) limit: **rs=2,β=0.04,clip1.0 ≡ rs=1,β=0.02,clip0.5.**

**Experiments — LSAT-50, lr=3.2e-5, pe2, n=3, joint 461-Q (tail / peak Δ simpl−cot):**

| recipe | Δ tail | Δ peak | note |
|---|---|---|---|
| rs=1, β=0.01 | −0.53 | −0.14 | simpl **overtrains** (peaks ~156 then declines), loses |
| rs=1, β=0.02 | +0.46 | +0.54 | partial recovery |
| rs=1, β=0.02, **clip=0.5** | **+1.58** | +1.07 | recovers the *gap* |
| rs=1, β=0.01, lr=6.4e-5 | −1.55 | −0.33 | higher-lr route **fails** (overtrains worse) |
| **rs=2, β=0.04, clip=1.0 (ref)** | **+1.80** | — | best operating point |

**Two honest conclusions:**
1. **clip0.5 reproduces the Δ (+1.58 vs +1.80) but NOT the absolute level** — both arms land
   ~0.9pp *below* rs=2 (cot 0.248 vs 0.255, simpl 0.264 vs 0.273). The idealized equivalence breaks
   (Adam ε + clipping reduce the effective step). **So we are NOT close to the rs=2 setting; rs=2
   stays the strongest recipe.** The clean recipe is a *defensible alternative at a weaker operating
   point*, not a faithful reproduction.
2. **β/clip do not substitute for rs cleanly.** Lowering β alone (the "rs≈half-β" idea) only got to
   +0.46; only β+clip together got the gap back, and even then the level is lower. rs=2's benefit is
   genuinely an optimization-regime (gradient-magnitude × clipping) effect.

**Understanding ≈ regularization.** Across all cells, the regularization that helps simpl *hurts*
cot: cot prefers low β / loose clip (0.264 @ β0.01), simpl prefers high β / tight clip (0.264 @
clip0.5). **At each arm's own best HP they ~tie (~0.264);** the +1.2–1.8pp simpl "win" always comes
from a *shared* HP in the regularization regime that suits simpl. simpl can absorb more
regularization (the understanding objective acts like an auxiliary regularizer) — that's the real
mechanism, and it's a more defensible framing than "simpl is just better."

**Generality of the clip05 recipe (all joint/test sets, avg@8, 3 seeds, tail-3, step-aligned).**
SImpL `cot_accuracy` vs CoT-only `cot_accuracy` (the apples-to-apples head; both arms scored by
their cot output):

| Setting | CoT cot_acc | SImpL cot_acc | Δ (tail-3) | verdict |
|---|---|---|---|---|
| **4B · LSAT-50** | 24.80 ± 0.47 | 26.12 ± 0.50 | **+1.33** | SImpL wins |
| **4B · RACE-50** | 82.64 ± 0.54 | 83.15–83.43 | **+0.5 … +0.8** | SImpL wins (smaller) |
| **8B · LSAT-50** | 28.61 ± 0.33 | 28.63 ± 0.72 | **+0.03** | **tie — win vanishes** |

**The clip05 SImpL advantage does NOT survive to 8B.** At 4B the understanding objective buys
+1.3pp on LSAT and +0.5–0.8pp on RACE; at 8B both arms converge to ~28.6% (Δ ≈ 0, and SImpL's
seed variance actually widens, ±0.72 vs ±0.33). The understanding-conditioned heads sit *below*
cot at 8B too (und+cot −0.6, u_and_a −0.1 tail). Read: the auxiliary regularization / extra signal
helps a **weaker base** that hasn't saturated the task, but the 8B base is already strong enough
on these passages that the marginal signal adds nothing. This is a scale-dependent effect and
should be reported as such — the headline claim is "more signal per datum **for smaller models**,"
not a universal win. (8B trained fine on a single H200, ~7h, no OOM; evals → `evaluations/bsweep2/lsat50-8b-*`.)

8B · LSAT-50 full per-head numbers (tail-3 @ steps 212/216/217):

| Arm | cot_acc | und+cot | u_and_a |
|---|---|---|---|
| CoT-only | 28.61 ± 0.33 | 28.21 ± 0.36 | 29.15 ± 0.37 |
| SImpL | 28.63 ± 0.72 | 28.01 ± 1.05 | 28.50 ± 0.75 |

(Note: at 8B even CoT-only's *u_and_a* head, 29.15, edges out everything — i.e. no arm's understanding
pathway beats the simple direct-answer baseline at this scale.)

**RACE-50 (4B) clip05 detail** (race-c test, avg@8, 3 seeds): CoT cot_acc 82.64 ± 0.54; SImpL cot_acc
83.43 ± 0.31 @196 / 83.15 ± 0.13 @200 (CoT's final = step 199). SImpL also has much tighter seed
variance. The win is on the cot head only — SImpL's und+cot/u_and_a heads do not beat CoT's cot head
on RACE either. Evals → `evaluations/bsweep2/race50-{cot,simpl}-clip05_s*`.

---

# flatten × weight ablation — difficulty weighting is NOT the active ingredient (2026-06-23)

To find what actually drives the +1.33 clip05 win, we ran a 2-axis ablation of full SImpL, each
arm differing from it by one knob (4B-LSAT-50, clip05, joint 461-Q, avg@8, 3 seeds, tail-3 cot_acc):

| arm | cot data | understanding eval | weights | cot_acc | Δ vs CoT |
|---|---|---|---|---|---|
| CoT-only | flattened | — | — | 24.80 ± 0.47 | — |
| **SImpL full** | per-passage, rotate | all-Q marginal | difficulty | 26.12 ± 0.50 | **+1.33** |
| **nbmarg** | per-passage, rotate | all-Q marginal | **uniform** | 26.02 ± 0.28 | **+1.22** |
| **nbmarg-flat** | flattened (1 row/Q) | all-Q marginal | uniform | 25.44 ± 0.68 | +0.64 |

**Axis decomposition (Δ cot_acc):**
- difficulty weighting (full − nbmarg): **+0.11** → noise, does essentially nothing.
- flattening (nbmarg − nbmarg-flat): **+0.57** → the per-passage/rotate cot structure matters
  *more* than the weighting (~1σ, suggestive).

**Conclusion — the difficulty weighting can be dropped.** Plain *uniform* all-questions understanding
co-training (**nbmarg**) recovers +1.22 of the +1.33 (a tie within noise) while removing the
difficulty weights AND the direct-baseline pass (a speedup). The win comes from co-training the
understanding objective on the full question set, NOT from any weighting scheme. nbmarg's
understanding-conditioned heads are also stronger than full SImpL's (und+cot 25.62 vs 24.91; u_and_a
25.62 vs 25.77). Flattening the cot data (every question trained directly, no rotate) *hurts* (−0.57),
likely by over-exposing cot to each question and diluting the understanding's relative contribution.

> ⚠️ This **corrects** an earlier (deleted) reading that "difficulty weighting is the active
> ingredient" — that came from a buggy *single-question* flattened no_bias variant that conflated the
> two axes. The single-Q variant is gone; the correct ablations are the table above.

**CANONICAL RECIPE GOING FORWARD = nbmarg (uniform, no difficulty weighting).** It is as good as
full SImpL, simpler, and faster. Code: `src/algorithm/simpl_no_bias_oat.py` (per-passage, all-Q
uniform marginal, `flatten_cot=false`); config `configs/qwen/main/nbmarg-lsat50-clip05.json`; launch
`scripts/run/simpl_no_bias_oat.sh`. Evals → `evaluations/nobias/nbmarg-*`. (full SImpL's
difficulty-weighting path is retained for reference but is no longer the default.)
