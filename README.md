# 🚗 Vehicle Classification: 딥러닝 기반 차종 분류 시스템

## 📌 프로젝트 소개
CNN 아키텍처 설계와 전이 학습 모델 비교를 통해 차량 이미지 8종을 정밀하게 분류하는 딥러닝 프로젝트입니다. 

## 📅 개발 기간
- 2026년 01월 02일 ~ 01월 06일 (1인 개발)

## ✨ 주요 기능
### 1. 딥러닝 모델 아키텍처 실험
- **단계별 설계:** CNN Basic부터 Deep, Dropout 구조까지 직접 설계 및 비교
- **전이 학습:** ResNet50, MobileNetV2 모델을 적용하여 성능 고도화
- **정량적 비교:** Validation Loss/Accuracy 기반의 객관적인 모델 성능 지표 도출

### 2. 안정적인 학습 파이프라인
- **데이터 증강:** `ImageDataGenerator`를 활용한 정규화 및 데이터 증강
- **자동 모델 관리:** `EarlyStopping`, `ModelCheckpoint`를 통해 과적합 방지 및 최적 모델 저장
- **구조적 분리:** 학습 및 시각화 로직을 분리하여 확장성 있는 코드베이스 구축

### 3. 실시간 예측 서비스
- 학습된 최적 모델을 로드하여 실시간 이미지 예측 서비스 구현 (Streamlit 기반)

## 🛠 기술 스택
- **Framework:** TensorFlow, Keras
- **Architecture:** CNN, ResNet50, MobileNetV2
- **UI:** Streamlit
