# Experiment-to-result map

## Ideal forecasting study

All five encodings use a 14-qubit direct data-reuploading QNN, six reuploading depths, seeds 42-46, 100 epochs, Adam, and analytic `default.qubit` simulation. Encoding-specific circuit definitions remain in separate training scripts so their provenance is explicit.

| Encoding | Training script | Production launcher |
| --- | --- | --- |
| Standard Ry | `data_reupload/scripts/final_sweep/train_final_single_qnn.py` | `final_single_tanh_a1_multiseed_d1_d3.sbatch` and `final_single_tanh_a1_multiseed_d4_d6.sbatch` |
| Same-scalar RxRyRz | `data_reupload/scripts/final_sweep/train_final_single_qnn_same_scalar_rxryrz.py` | `run_same_scalar_rxryrz_a1_full.sbatch` |
| Seasonal meridian | `data_reupload/scripts/final_sweep/train_final_single_qnn_seasonal_meridian.py` | `run_seasonal_meridian_a1_full.sbatch` |
| Learnable seasonal CDF | `data_reupload/scripts/final_sweep/train_final_single_qnn_learnable_seasonal_cdf.py` | `run_learnable_seasonal_cdf_a1_full.sbatch` |
| Learnable seasonal CDF with Rz | `data_reupload/scripts/final_sweep/train_final_single_qnn_learnable_seasonal_cdf_rz.py` | `run_learnable_seasonal_cdf_rz_a1_full.sbatch` |

All launchers are under `data_reupload/slurm/`.

## Ideal aggregation and validation

| Scientific output | Analysis workflow | Curated result location |
| --- | --- | --- |
| Five-encoding run manifest and depth summaries | `analysis/four_encoding_master/scripts/build_master_manifest.py` | `results/tables/ideal/forecasting/` |
| Cross-encoding, metric, spectral, and inventory audits | `analysis/four_encoding_master/scripts/build_master_manifest.py` | `results/tables/ideal/validation/` |
| Fifth-encoding learned parameters and paired validation | `analysis/four_encoding_master/scripts/validate_aggregate_model5_rz.py` | `results/tables/ideal/encoding_diagnostics/learnable_seasonal_cdf_rz/` |
| Exact temporal fidelity | `analysis/four_encoding_master/scripts/compute_exact_temporal_fidelity.py` and `aggregate_exact_temporal_fidelity.py` | `results/tables/ideal/geometry/` |
| Seasonal prominence | `analysis/four_encoding_master/scripts/analyze_fidelity_seasonal_prominence.py` | `results/tables/ideal/geometry/` |
| Directional QGT epsilon audit | `analysis/four_encoding_master/scripts/audit_directional_qgt_epsilon.py` | `results/tables/ideal/geometry/` |
| Exact reconstruction audit | `analysis/four_encoding_master/scripts/audit_exact_four_encoding_reconstruction.py` | `results/tables/ideal/geometry/` |

Paths in the analysis column are relative to `data_reupload/`.

## Finite-shot noisy workflows

| Workflow | Worker | Launcher or status |
| --- | --- | --- |
| Core seasonal smoke test | `noisy_direct14/scripts/smoke_test_noisy_seasonal_core.py` | `noisy_direct14/slurm/smoke_noisy_seasonal_core.slurm` |
| FakeMelbourneV2 depths 1-3 | `noisy_direct14/scripts/direct14_noisy_seasonal_two_pass_worker_pool_hpc.py` | `noisy_direct14/slurm/production_noisy_seasonal_array.slurm` |
| FakeMelbourneV2 depths 4-6 | `noisy_direct14/scripts/direct14_noisy_seasonal_two_pass_worker_pool_hpc.py` | `noisy_direct14/slurm/production_noisy_seasonal_depth456_array.slurm` |
| Noisy aggregation | `noisy_direct14/scripts/aggregate_noisy_seasonal_results.py` | `noisy_direct14/slurm/aggregate_noisy_seasonal_results.slurm` |
| Ideal-noisy comparison | `noisy_direct14/scripts/compare_ideal_and_noisy.py` | `noisy_direct14/slurm/compare_ideal_and_noisy.slurm` |
| Aachen topology-aware simulation | `noisy_aachen/scripts/direct14_noisy_aachen_topology_worker_pool_hpc.py` | Backend snapshot and worker committed; final launcher/results deferred until completion |
| Aachen candidate-30 simulation | `noisy_aachen/scripts/direct14_noisy_aachen_candidate30_worker.py` | Backend snapshot and worker committed; final launcher/results deferred until completion |

Paths in this table are relative to `data_reupload/`.

## Result policy

The repository tracks compact summaries, validation records, figure-source tables, and selected final figures. It intentionally excludes checkpoints, prediction arrays, raw Slurm logs, caches, smoke outputs, and duplicated report trees. Noisy result summaries are added only after the corresponding completion block has been validated.
