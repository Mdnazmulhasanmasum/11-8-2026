# TinyBrainNet PART-21 report

Research prototype. Not clinically validated. Model confidence is not clinical certainty.

## What ran
- Plan **B** (v6 initialisation and v6 classification teacher).
- Part-15 checkpoint: UNAVAILABLE; Part-16 checkpoint: UNAVAILABLE; segmentation teacher: tinyunet_pre (rule metric: history_best_val_dice).
- Stem fold (3->1) verified on identical tensors: max |diff| 4.112720489501953e-06; student accuracy at initialisation under the NEW preprocessing (BRISC clean-val): 0.9649717514124294.
- GPU time used: 127.1 of 300 min. Skipped for budget: none. Reduced: none.

## Selection (validation only)
- E2 if S_BRISC_clean_val(E2) >= S_BRISC_clean_val(E1) - 0.01, else E1 (pre-declared)
- S on BRISC clean-val: E1 0.9451, E2 0.9433 -> deployed **E2** (best epoch 27, checkpoint SHA-256 c069c14098294f84..., reload check passed).
- Extra seeds of E2: ['E2_seed43', 'E2_seed44']; validation S across seeds: 0.9425, 0.9420, 0.9430

## Results by tier (zero-shot, deployed model; reported, never used for selection)

| Tier | n | Main metric | Segmentation |
|---|---|---|---|
| T1_brisc_historical | 1000 | accuracy 0.9790 [0.9681, 0.9862]; macro-F1 0.9798 | Dice 0.8128 [0.7969, 0.8268]; healthy FP 0.0000 [0.0000, 0.0267] |
| T2_brisc_clean | 767 | accuracy 0.9831 [0.9712, 0.9901]; macro-F1 0.9832 | Dice 0.7961 [0.7771, 0.8143]; healthy FP 0.0000 [0.0000, 0.0267] |
| T3_nick_strict | 642 | accuracy 0.8816 [0.8543, 0.9044]; macro-F1 0.8240 | healthy FP 0.0930 [0.0682, 0.1255] |
| T4_figshare26 | 221 | accuracy 0.9774 [0.9481, 0.9903]; macro-F1 0.6943 | Dice 0.4909 [0.3887, 0.5864] |
| T5_br35h | 2930 | sensitivity 0.6895 [0.6650, 0.7130]; specificity 0.9467 [0.9341, 0.9569] | healthy FP 0.0813 [0.0685, 0.0963] |
| T6_tcga_lgg | 3929 | glioma recall 0.5084 [0.4819, 0.5348]; tumour detected 0.8194 [0.7981, 0.8388]; patients 0.8091 [0.7257, 0.8716] | Dice 0.3085 [0.2632, 0.3509]; empty-slice FP 0.4370 [0.4179, 0.4563] |

Overlap labels: T1 contains leaked images (historical only); T2 in-domain; T3 parent of BRISC (same-source for E2); T4 in-domain source, new patients;
T5 Br35H = a whole UNSEEN source (never trained or validated on; BRISC duplicates removed); T6 independent, glioma-only, FLAIR masks (approximate Dice).
empty_mask_fp = share of slices with an empty ground-truth mask that get > 20 predicted pixels (healthy images; for T6, tumour-free slices of glioma patients).

## Part-21 vs historical v6 (same images)
- T2_brisc_clean: Part-21 0.9831 vs v6 0.9817, exact McNemar p = 1 (n=767)
- T1_brisc_historical: Part-21 0.9790 vs v6 0.9830, exact McNemar p = 0.4807 (n=1000)
- T3_nick_strict: Part-21 0.8816 vs v6 0.7617, exact McNemar p = 1.1e-19 (n=642)

## Part-21 vs every historical classifier (same images, exact McNemar)
- vs v6 on T2_brisc_clean: Part-21 0.9831 vs 0.9817, p = 1 (n=767) -> Part-21 better, not significant
- vs v6 on T3_nick_strict: Part-21 0.8816 vs 0.7617, p = 1.1e-19 (n=642) -> Part-21 better, significant
- vs v6 on T4_figshare26: Part-21 0.9774 vs 0.9412, p = 0.0386 (n=221) -> Part-21 better, significant
- Teachers that saw Nickparvar TRAINING images (robust Part-15 v2) pass that knowledge to both arms; Nickparvar Testing (T3), Br35H (T5) and Figshare patients absent from BRISC (T4) were never used by any teacher or student.

## Comparison table

| model                   | tier                |      acc |   macro_f1 |   glioma_recall |     dice |   empty_mask_fp |     sens |     spec |   detected |
|:------------------------|:--------------------|---------:|-----------:|----------------:|---------:|----------------:|---------:|---------:|-----------:|
| Part-21 E2 (deployed)   | T1_brisc_historical |   0.979  |     0.9798 |          0.9567 |   0.8128 |          0      | nan      | nan      |   nan      |
| Part-21 E2 (deployed)   | T2_brisc_clean      |   0.9831 |     0.9832 |          0.966  |   0.7961 |          0      | nan      | nan      |   nan      |
| Part-21 E2 (deployed)   | T3_nick_strict      |   0.8816 |     0.824  |          0.5857 | nan      |          0.093  | nan      | nan      |   nan      |
| Part-21 E2 (deployed)   | T4_figshare26       |   0.9774 |     0.6943 |          0.9773 |   0.4909 |        nan      | nan      | nan      |   nan      |
| Part-21 E2 (deployed)   | T5_br35h            | nan      |   nan      |        nan      | nan      |          0.0813 |   0.6895 |   0.9467 |   nan      |
| Part-21 E2 (deployed)   | T6_tcga_lgg         | nan      |   nan      |          0.5084 |   0.3085 |          0.437  | nan      | nan      |     0.8194 |
| Part-21 E1              | T1_brisc_historical |   0.981  |     0.9822 |          0.9724 |   0.8095 |          0      | nan      | nan      |   nan      |
| Part-21 E1              | T2_brisc_clean      |   0.9857 |     0.9858 |          0.9806 |   0.7889 |          0      | nan      | nan      |   nan      |
| Part-21 E1              | T3_nick_strict      |   0.8458 |     0.7938 |          0.5929 | nan      |          0.1482 | nan      | nan      |   nan      |
| Part-21 E1              | T4_figshare26       |   0.9819 |     0.8287 |          0.9818 |   0.4789 |        nan      | nan      | nan      |   nan      |
| Part-21 E1              | T5_br35h            | nan      |   nan      |        nan      | nan      |          0.1473 |   0.7566 |   0.8973 |   nan      |
| Part-21 E1              | T6_tcga_lgg         | nan      |   nan      |          0.5368 |   0.2934 |          0.3842 | nan      | nan      |     0.8252 |
| historical v6           | T1_brisc_historical |   0.983  |     0.9834 |          0.9764 | nan      |        nan      | nan      | nan      |   nan      |
| historical v6           | T2_brisc_clean      |   0.9817 |     0.9815 |          0.9757 | nan      |        nan      | nan      | nan      |   nan      |
| historical v6           | T3_nick_strict      |   0.7617 |     0.702  |          0.4714 | nan      |        nan      | nan      | nan      |   nan      |
| historical v6           | T4_figshare26       |   0.9412 |     0.5757 |          0.9409 | nan      |        nan      | nan      | nan      |   nan      |
| historical v6           | T5_br35h            | nan      |   nan      |        nan      | nan      |        nan      |   0.7671 |   0.7667 |   nan      |
| historical v6           | T6_tcga_lgg         | nan      |   nan      |          0.5812 | nan      |        nan      | nan      | nan      |     0.7473 |
| historical tinyunet_pre | T1_brisc_historical | nan      |   nan      |        nan      |   0.8464 |          0.8071 | nan      | nan      |   nan      |
| historical tinyunet_pre | T2_brisc_clean      | nan      |   nan      |        nan      |   0.8324 |          0.8071 | nan      | nan      |   nan      |
| historical tinyunet_pre | T3_nick_strict      | nan      |   nan      |        nan      | nan      |          0.8417 | nan      | nan      |   nan      |
| historical tinyunet_pre | T4_figshare26       | nan      |   nan      |        nan      |   0.5872 |        nan      | nan      | nan      |   nan      |
| historical tinyunet_pre | T5_br35h            | nan      |   nan      |        nan      | nan      |          0.7987 | nan      | nan      |   nan      |
| historical tinyunet_pre | T6_tcga_lgg         | nan      |   nan      |        nan      |   0.3604 |          0.7527 | nan      | nan      |   nan      |

## Ablations (BRISC-only arm, 12 epochs, one seed)

| variant                     | flags                                                                                                                                   | ran   |   best_epoch |   val_S |   val_macro_f1 |   val_glioma_recall |   val_dice_tumour |   val_fp_healthy |   val_ece |   delta_S_vs_A0 |
|:----------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|:------|-------------:|--------:|---------------:|--------------------:|------------------:|-----------------:|----------:|----------------:|
| A0_reference                | {'use_kd': True, 'use_segkd': True, 'use_cons': True, 'use_att_sup': True, 'use_adapter': False, 'focal': False, 'photometric': True}   | True  |           11 |  0.9379 |         0.9921 |              0.9951 |            0.7952 |           0.0636 |    0.0082 |          0      |
| A1_no_KD                    | {'use_kd': False, 'use_segkd': False, 'use_cons': True, 'use_att_sup': True, 'use_adapter': False, 'focal': False, 'photometric': True} | True  |           12 |  0.9377 |         0.9875 |              0.9951 |            0.798  |           0.0578 |    0.0067 |         -0.0002 |
| A2_no_consistency           | {'use_kd': True, 'use_segkd': True, 'use_cons': False, 'use_att_sup': True, 'use_adapter': False, 'focal': False, 'photometric': True}  | True  |           11 |  0.9367 |         0.9898 |              0.9951 |            0.7888 |           0.052  |    0.0071 |         -0.0011 |
| A3_no_attention_supervision | {'use_kd': True, 'use_segkd': True, 'use_cons': True, 'use_att_sup': False, 'use_adapter': False, 'focal': False, 'photometric': True}  | True  |           12 |  0.9351 |         0.9877 |              0.9951 |            0.7883 |           0.0636 |    0.003  |         -0.0027 |
| A4_input_adapter            | {'use_kd': True, 'use_segkd': True, 'use_cons': True, 'use_att_sup': True, 'use_adapter': True, 'focal': False, 'photometric': True}    | True  |           12 |  0.9385 |         0.9887 |              0.9951 |            0.7952 |           0.0462 |    0.0074 |          0.0006 |
| A5_focal_loss               | {'use_kd': True, 'use_segkd': True, 'use_cons': True, 'use_att_sup': True, 'use_adapter': False, 'focal': True, 'photometric': True}    | True  |           12 |  0.9365 |         0.9887 |              0.9951 |            0.7907 |           0.052  |    0.0101 |         -0.0014 |

## Explainability (BRISC clean test tumour slices)
- learned attention (cam): pointing 0.6583 [0.5697, 0.7371], energy in mask 0.13883168415050023
- Grad-CAM++: pointing 0.0250 [0.0085, 0.0709], energy 0.01784606700530276; random-map expectation 0.017635091145833334
- sanity (randomised head) Spearman median: 0.045876859888126266

## Efficiency (measured)
- 496,503 trainable parameters (limit 500,000); 2.14 MB fp32; 0.482 GFLOPs
- CPU batch-1, 1 thread: 38.7 ms; all 1 threads: 38.8 ms; GPU Tesla T4: 7.20 ms, peak 4067 MB

## Deployment verification
- ONNX parity passed: True; TorchScript parity passed: True
- preprocess.py identical to the training cache: {'n': 50, 'max_abs_diff_u8': 0}
- fresh Python process (bundle only, no teacher): {'n_images': 4, 'max_abs_logit_diff': 1.780688762664795e-06, 'statuses': ['ok', 'ok', 'review', 'review']}
- ZIP: /kaggle/working/TinyBrainNet_PART21/TinyBrainNet_PART21_Ultimate_Deployment.zip (26 files)

## Targets, measured (never assumed)
- Classification >= 95 % on BRISC clean test: MET (accuracy 0.9831 [0.9712, 0.9901]; 'met' requires the lower 95 % bound >= 0.95).
- Dice on BRISC clean-test tumour images: 0.7961 [0.7771, 0.8143]. A 0.95 Dice target was NOT set: 2D brain-tumour masks have inter-rater agreement well below 0.95, so such a value would point to leakage rather than skill.
- Reference Dice from the original notebook (seg_eval_summary.json; official BRISC test incl. leaked images, tumour slices only, different protocol): {"attention_unet": {"params_M": 7.854717, "dice224": {"mean": 0.852435290813446, "median": 0.9194183945655823, "sd": 0.17844702303409576, "ci95": [0.8402953147888184, 0.8640976548194885], "frac_gt_0.8": 0.7953488372093023}, "dice_native": {"mean": 0.8519626176431444, "median": 0.9178014777227483, "sd": 0.17919682572169968, "ci95": [0.8398222680190519, 0.8635977784606034], "frac_gt_0.8": 0.791860465116279}}, "tinyunet_v6_pre": {"params_M": 0.482209, "dice224": {"mean": 0.8429777026176453, "median": 0.9114090204238892, "sd": 0.18471240997314453, "ci95": [0.8302276730537415, 0.8552631139755249], "frac_gt_0.8": 0.7883720930232558}, "dice_native": {"mean": 0.8408335135113122, "median": 0.9067193563903446, "sd": 0.18498308109574255, "ci95": [0.8279810774880598, 0.8531080002436682], "frac_gt_0.8": 0.7872093023255814}, "wilcoxon_p_vs_attunet_dice224": 5.1662102643177276e-15, "wilcoxon_p_vs_attun

## How the app answers (selective prediction + OOD gate; thresholds from validation only)
- The app answers only if confidence >= 0.25 (proves >= 95% accuracy on validation with a one-sided 95 % bound), the image is not out-of-distribution, classifier and segmenter agree, and the head was found; otherwise it refers the image to a clinician.

| Tier | n | answered | accuracy, all | accuracy, answered | target 95 % |
|---|---|---|---|---|---|
| T1_brisc_historical | 1000 | 0.9590 [0.9449, 0.9696] | 0.9790 [0.9681, 0.9862] | 0.9823 [0.9718, 0.9889] | met (lower 95% bound >= 0.95) |
| T2_brisc_clean | 767 | 0.9752 [0.9616, 0.9841] | 0.9831 [0.9712, 0.9901] | 0.9853 [0.9739, 0.9918] | met (lower 95% bound >= 0.95) |
| T3_nick_strict | 642 | 0.9050 [0.8798, 0.9253] | 0.8816 [0.8543, 0.9044] | 0.9105 [0.8845, 0.9311] | NOT met |
| T4_figshare26 | 221 | 0.9005 [0.8539, 0.9333] | 0.9774 [0.9481, 0.9903] | 0.9749 [0.9425, 0.9892] | point >= 0.95, not proven |
| T5_br35h | 2930 | 0.7744 [0.7589, 0.7892] | n/a | n/a | - |
|  | | | detection, answered: sensitivity 0.7821 [0.7535, 0.8082] | specificity 0.9642 [0.9531, 0.9727] | |
| T6_tcga_lgg | 3929 | 0.6905 [0.6759, 0.7048] | 0.5084 [0.4819, 0.5348] | 0.5735 [0.5433, 0.6031] | NOT met |

## Limitations
- BRISC, Nickparvar and Br35H have no patient IDs: near-duplicate groups (dHash) are a proxy; patient independence is verified only via Figshare PIDs.
- One sequence (T1 contrast-enhanced). TCGA-LGG uses its post-contrast channel and FLAIR masks: its Dice is approximate and low-grade gliomas often do not enhance.
- Ablations use one seed; differences below the seed spread are not evidence.
- External results are retrospective; no prospective or clinical validation.

## Keeping the outputs on Kaggle
- Run as a committed version (Save Version -> Save & Run All): /kaggle/working/TinyBrainNet_PART21/ becomes the notebook output.
- To resume later: add that output as an Input; Part-21 skips every finished stage (done markers, resume checkpoints).