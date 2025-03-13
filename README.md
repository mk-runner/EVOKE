# EVOKE
[EVOKE: Elevating Chest X-ray Report Generation via Multi-View Contrastive Learning and Patient-Specific Knowledge](https://arxiv.org/abs/2411.10224)

Radiology reports are crucial for planning treatment strategies and facilitating effective doctor-patient communication. However, the manual creation of these reports places a significant burden on radiologists. While automatic radiology report generation presents a promising solution, existing methods often rely on single-view radiographs, which constrain diagnostic accuracy. To address this challenge, we propose \textbf{EVOKE}, a novel chest X-ray report generation framework that incorporates multi-view contrastive learning and patient-specific knowledge. Specifically, we introduce a multi-view contrastive learning method that enhances visual representation by aligning multi-view radiographs with their corresponding report. After that, we present a knowledge-guided report generation module that integrates available patient-specific indications (e.g., symptom descriptions) to trigger the production of accurate and coherent radiology reports. To support research in multi-view report generation, we construct Multi-view CXR and Two-view CXR datasets using publicly available sources. Our proposed EVOKE surpasses recent state-of-the-art methods across multiple datasets, achieving a 2.9% $F_{1}$ RadGraph improvement on MIMIC-CXR, a 7.3% BLEU-1 improvement on MIMIC-ABN, a 3.1% BLEU-4 improvement on Multi-view CXR, and an 8.2% $F_{\text{1,mic-14}}$ CheXbert improvement on Two-view CXR.
<div align=center><img src="results/figure2.png"></div>

## Update
- The code, checkpoints, and generated radiology reports are coming soon.

## Multi-view CXR
Multi-view CXR aggregates studies with multiple views from MIMIC-CXR [1] and IU X-ray [2]. 

- Regarding radiographs, they can be obtained from [physionet](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) and [NIH](https://openi.nlm.nih.gov/faq#collection). The file structure for storing these images can be represented as:

```
files/
├── p10
├── p11
├── p12
├── p13
├── p14
├── p15
├── p16
├── p17
├── p18
├── p19
└── NLMCXR_png
```
- As for radiology reports, they can be downloaded in [huggingface 🤗](https://huggingface.co/datasets/MK-runner/Multi-view-CXR).

## Two-view CXR
Two-view CXR is a variant of Multi-view CXR that includes only two views per study. The dataset can be downloaded in [huggingface 🤗](https://huggingface.co/datasets/MK-runner/Multi-view-CXR).

## Usage
    ```python
    # obtain all studies of Multi-view CXR
    import json
    path = 'multiview_cxr_annotation.json'
    multi_view_cxr_data = json.load(open(path))

    # obtain all studies of Two-view CXR
    ann_data = json.load(open(path))
    two_view_cxr_data = {}
    for key, value in ann_data.items():
       two_view_cxr_data[key] = []
       for item in ann_data:
            ## current image_num
            image_num = len(item['anchor_scan']['image_path']) + len(item['auxiliary_references']['image_path'])
            if image_num != 2:
                two_view_cxr_data[key].append(item)
      
    ```

## Statistics for the training, validation, and test sets across MIMIC-CXR, MIMIC-ABN, Multi-view CXR, and Two-view CXR.
<div align=center><img src="results/data-statistics.png"></div>


## Citations

If you use or extend our work, please cite our paper at arXiv.

```
@misc{miao2025evokeelevatingchestxray,
      title={EVOKE: Elevating Chest X-ray Report Generation via Multi-View Contrastive Learning and Patient-Specific Knowledge}, 
      author={Qiguang Miao and Kang Liu and Zhuoqi Ma and Yunan Li and Xiaolu Kang and Ruixuan Liu and Tianyi Liu and Kun Xie and Zhicheng Jiao},
      year={2025},
      eprint={2411.10224},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.10224}, 
}
```

## Acknowledgement

- [R2Gen](https://github.com/zhjohnchan/R2Gen) Some codes are adapted based on R2Gen.
- [R2GenCMN](https://github.com/zhjohnchan/R2GenCMN) Some codes are adapted based on R2GenCMN.
- [MGCA](https://github.com/HKU-MedAI/MGCA) Some codes are adapted based on MGCA.

## References
[1] Johnson, Alistair EW, et al. "MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs." arXiv preprint arXiv:1901.07042 (2019).

[2] Demner-Fushman, Dina, et al. "Preparing a collection of radiology examinations for distribution and retrieval." Journal of the American Medical Informatics Association 23.2 (2016): 304-310.

[3] Ni, Jianmo, et al. "Learning Visual-Semantic Embeddings for Reporting Abnormal Findings on Chest X-rays." Findings of the Association for Computational Linguistics: EMNLP 2020. 2020.

[4] Chen, Zhihong, et al. "Generating Radiology Reports via Memory-driven Transformer." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020.

[5] Chen, Zhihong, et al. "Cross-modal Memory Networks for Radiology Report Generation." Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2021. 

[6] Wang, Fuying, et al. "Multi-granularity cross-modal alignment for generalized medical visual representation learning." Advances in Neural Information Processing Systems 35 (2022): 33536-33549.
