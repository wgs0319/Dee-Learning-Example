# Vehicle_Classification

## 프로젝트 소개
딥러닝을 이용한 차종 분류 모델

## 개발기간
- 2026년 01월 02일 ~ 01월 06일

## 개발 인원
- 1인 개발

## 주요 기능
1. 데이터 전처리 · 모델 학습 · 예측 · 시각화 영역 분리 설계
 - ImageDataGenerator를 활용한 이미지 정규화 및 데이터 증강
 - CNN Basic / Deep / Dropout 모델을 단계적으로 설계하여 성능 비교 실험
 - ResNet50, MobileNetV2 전이학습 모델 적용을 통한 성능 고도화
 - 학습, 검증, 시각화 로직을 명확히 분리하여 유지보수 및 확장성 고려
2. 딥러닝 모델 비교 실험 및 파이프라인 구축
 - EarlyStopping, ModelCheckpoint를 활용한 과적합 방지 및 최적 모델 자동 저장
 - Validation Loss / Accuracy 기반 모델별 성능 정량 비교
 - CNN 단일 모델 → 전이학습 모델까지 단계적 실험을 통해 모델 선택 근거를 명확히 제시
 - Jupyter Notebook 기반 실험 코드와 Streamlit 서비스 코드 분리로 프로젝트 구조 체계화
3. Streamlit 대시보드 기반 시각화 및 서비스화
 - 학습된 최적 모델을 로드하여 실시간 이미지 예측 서비스 구현
 - 모델별 Loss / Accuracy Curve 시각화
 - Validation 성능 비교 차트 제공을 통해 비전문가도 이해 가능한 결과 표현
 - 딥러닝 모델을 실제 서비스 형태로 구현하여 실무 활용 가능성 강화
