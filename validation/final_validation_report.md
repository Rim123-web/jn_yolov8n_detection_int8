# Final Validation

**PASS**

| Gate | Result | Evidence |
|---|---|---|
| source_hash_matches_fp32_fp16 | PASS |  |
| onnx_hash_matches_fp32_fp16 | PASS |  |
| static_input_shape | PASS |  |
| batch_one | PASS |  |
| calibration_dataset_valid | PASS |  |
| class_mapping | PASS |  |
| exact_test_overlap_zero | PASS |  |
| pixel_overlap_zero | PASS |  |
| unresolved_near_duplicates_zero | PASS |  |
| calibration_selection_deterministic | PASS |  |
| calibration_preprocessing_parity | PASS |  |
| int8_flag | PASS |  |
| fp16_flag_documented | PASS |  |
| tf32_off | PASS |  |
| calibrator_attached | PASS |  |
| fresh_cache | PASS |  |
| binding_dtypes_documented | PASS |  |
| fallback_documented | PASS |  |
| platform_capability_documented | PASS |  |
| dla_disabled | PASS |  |
| engine_deserializes | PASS |  |
| engine_hash | PASS |  |
| cache_hash | PASS |  |
| script_parity | PASS |  |
| preprocessing_code_parity | PASS |  |
| postprocessing_code_parity | PASS |  |
| nms_code_parity | PASS |  |
| timing_field_parity | PASS |  |
| threshold_parity | PASS |  |
| test_order_parity | PASS |  |
| warmup_order_parity | PASS |  |
| correctness | PASS |  |
| official_acceptance | PASS |  |
| warmup_rows_10 | PASS |  |
| warmup_executions_10 | PASS |  |
| measured_rows_192 | PASS |  |
| prediction_rows_192 | PASS |  |
| hidden_failures_discarded_zero | PASS |  |
| accepted_executions_202 | PASS |  |
| timings_finite_nonnegative | PASS |  |
| independent_nms_timing | PASS |  |
| independent_engine_timing | PASS |  |
| e2e_timing | PASS |  |
| jtop_health | PASS |  |
| idle_baseline | PASS |  |
| energy | PASS |  |
| metrics | PASS |  |
| runtime_oom_zero | PASS | accepted official validator no_oom PASS |
| temporary_swap_absent | PASS |  |
| reference_immutability | PASS |  |
| figures_all_formats | PASS |  |
| figure_data | PASS |  |
| reports | PASS |  |
| complete_manifest | PASS | [] |
