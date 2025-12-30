<!-- Header -->
<div align="center">

![header](https://capsule-render.vercel.app/api?type=rounded&height=170&text=IR%20Contest%20RAG%20Retrieval&desc=%EA%B3%BC%ED%95%99%20%EC%83%81%EC%8B%9D%20RAG%20%EA%B2%80%EC%83%89%20%EC%8B%9C%EC%8A%A4%ED%85%9C%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C%20%EC%8B%A4%ED%97%98%20%EB%A6%AC%ED%8F%AC%EC%A7%80%ED%86%A0%EB%A6%AC&fontSize=36&descSize=15&descAlignY=65&color=gradient&fontColor=ffffff&animation=fadeIn)

<h3>🔎 Sparse(BM25) + Dense(Embedding) + Rerank · 과학 상식 RAG Retrieval Competition 🔎</h3>

</div>

---

## 💻 프로젝트 소개

### 📌 프로젝트 개요

이 리포지토리는 **과학 상식 문서(약 4.2k)** 에서 사용자 질문에 맞는 문서를 **Top-3로 검색(retrieval)** 하는 RAG 시스템을 구현/실험한 코드입니다.

- **Task**: Retrieval (Top-3 docid 제출)
- **Key Components**
  - Sparse: **Okt 기반 BM25Okapi**, (실험) **Elasticsearch BM25**
  - Dense: **Upstage Embeddings (solar-embedding-1-large-*)** + 문서 벡터 캐시
  - (옵션) Cross-Encoder re-rank, overlap bonus, rank-fusion
  - (옵션) LLM 기반 intent 분류 + query rewrite (Upstage Solar)
- **Goal**
  - 과학/지식 질문이면 검색 수행 → topk docid 3개 제출
  - 비과학/잡담이면 검색 스킵 → `topk: []`

---

## 🧪 평가 방식(대회 요약)

대회는 end-to-end 답변 품질이 아니라, **문서 추출 성능**을 평가합니다.

- **Metric**: MAP 변형 (상위 3개 topk만 사용)
  - 과학 질문(ground truth 존재): top3 안에 정답 docid가 얼마나 포함되는지로 평균정밀도(AP) 계산
  - 비과학 질문(ground truth 없음): `topk`가 비어 있으면 1점, 하나라도 뽑으면 0점

---

## 📂 프로젝트 구조

```bash
IR_contest/
├── data/
│   ├── documents.jsonl              # 원본 문서(약 4.2k)
│   └── eval.jsonl                   # 평가 질문(220, 멀티턴 포함)
├── cache/
│   ├── *.jsonl                      # 문서 가공 결과(문서/청크 단위)
│   ├── *.npy                        # 문서 임베딩 캐시(벡터)
│   ├── *.npy.docids.json            # 벡터 row 인덱스 ↔ docid 매핑
│   └── eval_need_search_llm.*.jsonl # intent/rewrite 캐시(프롬프트 시그니처 분리)
├── tools/
│   ├── build_contextual_retrieval_docs.py  # 문서 가공(제목/요약/… 생성)
│   └── build_doc_vectors_cache.py          # 문서 임베딩 캐시 생성(.npy)
├── notebook/
│   ├── 11a_*.ipynb                  # 주력 파이프라인(BM25 + Dense + rerank)
│   ├── 11h_*.ipynb                  # 16-IR_Project 스타일 실험 파이프라인
│   └── ...                          # 실험/스냅샷/백업들
├── experiemnt/                      # 참고/이전 프로젝트 코드(16-IR_Project 포함)
├── outputs/
│   └── submission/
│       ├── *_custom_faiss.csv       # 제출 파일(JSONL lines지만 확장자는 csv로 유지)
│       └── *_custom_faiss.inspect.txt  # 사람이 확인 가능한 디버그 출력
└── env.solar.sh                     # Upstage API 키 로드용(선택)
```

---

## 🧠 핵심 아이디어(실험 축)

### 1) 문서 가공(Contextual Retrieval)
원본 문서 `content`를 그대로 쓰는 대신, LLM으로 **제목/요약(및 옵션 필드)** 를 생성해 `content`에 붙여 검색 표현을 풍부하게 만듭니다.

- 장점: 문서에 없는 질의 표현(동의어/상위개념/표현 차이)을 “요약/제목”이 흡수해 sparse/dense 모두에서 recall 개선 가능
- 주의: 문서에 없는 내용을 과도하게 생성하면 retrieval이 틀어질 수 있어 프롬프트 제약이 중요

### 2) 의도 분류 + Query Rewrite
멀티턴 대화에서 **사용자 발화만 결합**한 뒤 LLM으로:
- `need_search`: 검색 필요 여부
- `standalone_query`: 검색용 질의(재작성)

을 출력하고 캐시(`cache/eval_need_search_llm.*.jsonl`)로 재사용합니다.

### 3) 3-Way 하이브리드 검색(예: ES BM25 + 로컬 BM25 + Dense)
Sparse 2개 + Dense 1개를 **rank 기반 결합(rank-fusion)** 으로 섞어 “한 엔진이 놓친 정답을 다른 엔진이 건져올리도록” 설계합니다.

### 4) Overlap bonus / Cross-Encoder
최종 Top-3 품질을 위해:
- query 키워드가 문서에 실제 포함되면 가점(overlap bonus)
- 최상위 후보 일부를 Cross-Encoder로 재랭킹(옵션)

---

## 🔑 환경 설정(필수)

### Upstage API Key

가장 간단한 방법:

```bash
export UPSTAGE_API_KEY="YOUR_KEY"
```

또는 `env.solar.sh`에 다음처럼 저장해두고 노트북/스크립트에서 파싱해 사용할 수 있습니다:

```bash
export UPSTAGE_API_KEY="YOUR_KEY"
```

---

## 🚀 실행 방법(대표 플로우)

### 1) 문서 가공 생성(예: doc-level 제목+요약)

```bash
python tools/build_contextual_retrieval_docs.py \
  --in data/documents.jsonl \
  --out-dir cache/doclevel_1400_ov0_titlesum_only \
  --model solar-mini --temperature 0.0 \
  --prompt-mode loose --prompt-template doc_titlesum \
  --chunk-size 1400 --chunk-overlap 0 \
  --output-format documents_jsonl --write-txt \
  --workers 4 --max-inflight 16 --timeout 180 --resume
```

### 2) 문서 임베딩 캐시 생성(.npy)

```bash
python tools/build_doc_vectors_cache.py \
  --docs cache/doclevel_1400_ov0_titlesum_only/contextual_retrieval_docs_loose_temp0p0_*.jsonl \
  --out cache/doc_vectors_solarmini_doclevel_titlesum_t0p0_$(date +%Y%m%d_%H%M%S).npy \
  --model solar-embedding-1-large-passage \
  --batch 64
```

생성물:
- `*.npy`: 문서 임베딩 행렬
- `*.npy.docids.json`: `npy[row]`가 어떤 `docid`인지 매핑(검색 결과를 docid로 되돌릴 때 사용)

### 3) 노트북 실행(재현/제출 파일 생성)

노트북을 직접 열어서 실행하거나, CLI로 실행:

```bash
jupyter nbconvert --to notebook --execute notebook/<YOUR_NOTEBOOK>.ipynb \
  --output outputs/submission/<RUN_TAG>.executed.ipynb
```

노트북 실행 결과는 보통 `outputs/submission/`에 제출 파일과 inspect 파일이 생성됩니다.

---

## 🧾 제출 파일 포맷

대회 제출 파일은 **JSONL 형태의 라인**이며, 확장자만 `.csv`를 사용합니다.

```json
{"eval_id": 0, "standalone_query": "...", "topk": ["docid1","docid2","docid3"], "answer": "", "references": [{"score": 1.23, "content": "..."}]}
```

- `topk`가 비어 있으면 비과학/스킵으로 처리(평가 방식 참고)
- `references`는 제출에 필수는 아니지만 실험/앙상블/분석에 유용

---

## 🧯 자주 발생한 이슈(메모)

- **문서 임베딩 재생성 비용**: 문서 4.2k 전체를 Upstage 임베딩으로 재호출하면 비용/시간이 커서 `.npy` 캐시를 우선 사용
- **resume/중복 docid**: 문서 가공을 재실행할 때 출력 파일명이 같으면 덮어쓰기/중복이 생길 수 있어 타임스탬프/아웃디렉토리 분리가 중요
- **tokenizers/transformers 버전 충돌**: Cross-Encoder 사용 시 환경에 따라 ImportError가 날 수 있어 optional 처리 필요

---

## 📌 참고

- `experiemnt/16-IR_Project/`: 참고용(성능이 좋았던 다른 구현/아이디어들)
- `outputs/submission/*.inspect.txt`: 어떤 query가 만들어졌고 topk가 왜 뽑혔는지 확인하는 디버그 산출물

