# Nemotron 3.5 ASR Streaming 파인튜닝 사례 (2026-11-13 조사)

후보 모델 `nvidia/nemotron-3.5-asr-streaming-0.6b`(cache-aware FastConformer-RNNT, 600M)를 파인튜닝한 공개 사례를 모았다. **한국어나 구음장애로 이 모델을 파인튜닝한 사례는 찾지 못했다.** 수치는 각 출처가 스스로 밝힌 값이고, 우리가 재현한 것은 아니다.

## 1. 모델 기본 정보

- 파라미터 600M, 가중치 F32. 런타임은 NeMo 26.06(main 브랜치 설치), Python 3.11 이상
- 한국어(ko-KR)는 "transcription-ready" 19개 로케일에 들어 있다. 따라서 우리 작업은 언어를 새로 넣는 것이 아니라 **도메인(구음장애) 적응**이다
- 한국어 FLEURS CER (LangID 지정): 80ms 7.59 / 160ms 7.70 / 320ms 7.27 / 560ms 7.18 / 1.12s 7.12. 언어 자동 감지일 때는 7.30~8.31
  - GitHub 미러는 같은 숫자를 WER로 적었다. 22명 제로샷으로 직접 확인한다
- 지연 설정은 `att_context_size=[56, R]`, R ∈ {0, 1, 3, 6, 13}이다. 80ms 단위라 각각 80ms / 160ms / 320ms / 560ms / 1.12s
- LoRA는 지원하지 않는다 (HF 블로그 댓글). 전체 파인튜닝으로 간다
- 라이선스 OpenMDW-1.1, 상업 사용 가능

## 2. 사례 요약

| 사례 | 데이터 | 결과 (전 → 후) | 장비·시간 |
|---|---|---|---|
| 사우디 아랍어 방언 (NVIDIA 블로그) | 133.7h + FLEURS 10% 섞음 | CER 31.63 → **12.18**, WER 55.05 → 29.96 | RTX PRO 6000(96GB) 2장, 12,000 step, 약 4.5h |
| 그리스어·불가리아어 (NVIDIA HF 블로그) | 약 2,000h | WER 35 → 24, 22 → 15 | GPU 1장 (기종 안 밝힘) |
| 마라티어 (커뮤니티 레포) | 밝히지 않음 | CER 1.04 → **0.24**, WER 1.33 → 0.53 | Modal H100/H200 |
| 케냐 3개 언어 (arXiv 2607.18912) | 언어별 151~207h | 예: Kikuyu WER 42.97, CER 7.79 (기준선 비교 없음) | Modal 클라우드 GPU |
| 헝가리어 중증 구음장애 1명 (arXiv 2508.06391) | 본인 음성 + TTS 합성 음성 | 제로샷 CER 36~51 → **7.3** | FastConformer(스트리밍 아님). 같은 조건에서 Whisper-turbo보다 나았다 |
| 독일어 성능 문의 (HF 토론 #11) | 정보 없음 | 수치 없음 | NVIDIA 직원 답변: "GPU 1장으로 몇 시간" |

데이터가 100~200h 수준인 사례에서도 오류율이 절반 이하로 떨어졌다. 구음장애 사례는 스트리밍 모델이 아니고 화자도 1명이라 참고만 한다.

## 3. 우리 설정에 반영할 것

### 메모리 (L4 24GB는 빠듯하다)

- 사우디 사례는 96GB GPU에서 `batch_duration` 400초가 메모리 부족으로 터져 300초, 200초로 줄였다
- StreamHear(arXiv 2608.13717)는 A100 40GB에서 Nemotron 3.5만 배치 1 + 기울기 누적 16으로 돌렸다
- NVIDIA 공식 노트북 예시는 `batch_duration=200`, `precision=bf16`이다
- → L4에서는 `batch_duration` 50초 안팎 + 기울기 누적 4~8로 시작한다 (추정, 실측 필요). **본 학습은 Colab A100 80GB/H100이 현실적**이고, L4는 제로샷 평가와 서빙에 쓴다

### 전사 정리

- 모델이 낼 수 없는 표기는 매번 오답이 된다. 우리 라벨의 `* + / ( )` 기호를 지우거나 풀어서 정리한다
- 모든 조각의 manifest에 `target_lang: "ko-KR"`(공식 노트북은 `lang`도 같이 넣음)을 넣는다. 평가 때 `strip_lang_tags=True`를 켠다. 안 켜면 언어 태그가 매 문장 삽입 오류로 잡힌다
- 기본 모델이 문장부호가 있는 문장을 내므로, 학습 전사도 `.` `?`를 남기는 쪽이 맞다 (HF 블로그). 평가할 때는 문장부호를 빼고 CER을 잰다
- 토크나이저는 그대로 쓴다. 공식 노트북은 데이터가 50h 미만이면 기존 토크나이저를 쓰라고 한다. 한국어는 이미 지원 로케일이라 바꿀 이유가 없다

### 학습 방식

- 시작점은 `+init_from_nemo_model=<base .nemo>`, 설정은 `cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml`, 스크립트는 `speech_to_text_finetune.py`
- 인코더 전체를 학습한다. 사우디 사례에서 전체 24층 29.96%, 상위 8층 32.32%, 상위 6층 33.42%였다
- `bf16-mixed`를 쓴다. 마라티어 사례에서 `bf16-true`로 바꾸고 학습률 스케줄도 함께 바꿨더니 학습 WER이 약 0.94에서 멈췄다. `bf16-mixed`로 되돌리니 0.55까지 떨어졌다
- 학습률은 작게 잡는다. 사우디 사례는 1e-4에서 2e-5로 내렸고 warmup도 50~100 step으로 줄였다. 스케줄러는 Noam이라 `lr` 값의 뜻이 일반 AdamW와 다르다. 설정 파일에서 확인한다
- 일반 음성을 10~30% 섞는다 (사우디 10%, NVIDIA 직원 권고 20~30%). 쉬운 화자 8명 점수를 지키는 효과도 기대한다. 후보는 FLEURS 한국어
  - 섞는 비율은 설정의 가중치로 선언한다. 사우디 사례에 따르면 그냥 이어 붙인 replay 세트는 학습 대부분 동안 사실상 쓰이지 않는다
- step 수는 고정 예산으로 정한다. 검증 점수가 멈추면 더 돌려도 소용없었다 (사우디, 케냐)
- `num_buckets`만 넣고 `use_bucketing`을 안 켜면 버킷팅이 동작하지 않는다

### 필터링

- 사우디 사례 기준: 조각 길이 0.5~30초, 초당 글자 수 1.5~35, 품질 점수(UTMOS 1.25 이상 등)
- **구음장애 화자는 말이 느리다.** 초당 글자 수 하한을 그대로 쓰면 중증 화자 조각이 먼저 잘린다. 하한은 우리 분포를 보고 정한다
- 품질 점수 기준도 분포를 먼저 본다. 사우디 사례에서 기본 UTMOS 3.0을 쓰면 거의 다 걸렸다
- 불확실한 전사는 고치기보다 버린다 (케냐)

### 평가

- 시연에서 쓸 지연값으로 평가한다. 사우디 사례: [56,3] 320ms 29.96% / [56,13] 1.12s 28.65% / 1.12s + beam 8 27.25%
- beam 4는 greedy 대비 실행 시간 0.59배로 보고됐다 (사우디). 스트리밍 서빙에서도 그런지는 확인 필요
- 오프라인 WER만으로는 부족하다. 실제 스트리밍 방식으로 잰다 (케냐)

### 데이터 누출

- 마라티어 사례는 tar 묶음을 만들다 `valid-*` 묶음이 학습에 섞였다. 화자 단위 분할이 tar 단계까지 유지되는지 검사한다
- 여러 GPU 학습에서 tar가 아닌 검증 세트가 NCCL 타임아웃을 냈다 (마라티어). 단일 GPU면 해당 없음

## 4. 아직 모르는 것

- L4·A100에서 실제로 어느 `batch_duration`까지 들어가는지, step당 시간
- 우리 데이터(약 50~90h)에 맞는 step 수와 학습률
- 구음장애에서 일반 음성을 섞는 비율이 쉬운 화자와 중증 화자에게 각각 어떤 영향을 주는지

## 출처

- [nvidia/nemotron-3.5-asr-streaming-0.6b 모델 카드](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [How to Fine-Tune Nemotron 3.5 ASR (HF 블로그)](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr)
- [nvidia-riva/tutorials 파인튜닝 노트북](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- [Fine-Tuning NVIDIA Nemotron for Saudi Arabic Dialects (NVIDIA 블로그)](https://developer.nvidia.com/blog/fine-tuning-nvidia-nemotron-for-saudi-arabic-dialects-with-a-path-to-other-languages/)
- [nemotron-indic-asr (마라티어)](https://github.com/Sudhish-Neelakandan-ATX/nemotron-indic-asr)
- [Data-Centric Adaptation of Nemotron 3.5 for Kenyan languages (arXiv 2607.18912)](https://arxiv.org/html/2607.18912)
- [Improved Dysarthric Speech to Text via TTS Personalization (arXiv 2508.06391)](https://arxiv.org/abs/2508.06391)
- [StreamHear (arXiv 2608.13717)](https://arxiv.org/pdf/2608.13717)
- [HF 토론 #11: 독일어 성능과 NVIDIA 답변](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/discussions/11)
- [Piyazon 파인튜닝 키트](https://github.com/Piyazon/nemotron-3.5-asr-streaming-0.6b-custom-finetune-kit)
