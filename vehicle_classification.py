import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# 텐서플로우 로그 레벨 설정 (필수 유지)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- 1. 페이지 초기 설정 ---
st.set_page_config(
    page_title="고급 차량 분류기",
    page_icon="🚗",
    layout="wide"
)

# 제목 및 소개 (더 깔끔하게)
st.title("🚗 AI 기반 차량 종류 분류 시스템")
st.markdown("차량 이미지를 업로드하고, 훈련된 딥러닝 모델을 사용하여 차량 종류를 예측합니다.")
st.markdown("---")

# 클래스 이름 정의
CLASS_NAMES = ['SUV', '버스', '세단', '승합', '이륜차', '트럭', '해치백', '화물']
IMAGE_SIZE = (224, 224)

# --- 2. 사이드바 설정 (Side Bar) ---
st.sidebar.header("⚙️ 모델 및 설정")
model_choice = st.sidebar.selectbox(
    "사용할 모델 선택",
    ["CNN_With_Dropout", "ResNet50", "MobileNetV2"],
    help="성능을 비교하고 싶은 딥러닝 모델을 선택하세요."
)
st.sidebar.info(f"선택된 모델: **{model_choice}**")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 결과 정보")
st.sidebar.write(f"**분류 클래스 수:** {len(CLASS_NAMES)}개")
st.sidebar.write(f"**모델 입력 크기:** {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x3")


# --- 3. 핵심 함수: 모델 로드 및 이미지 전처리 (기존 기능 유지) ---

# @st.cache_resource를 사용하여 모델 로드 속도 향상
@st.cache_resource
def load_trained_model(model_name):
    """지정된 이름으로 Keras 또는 H5 모델을 로드합니다."""
    model_path_keras = f'./model/final_{model_name}.keras'
    model_path_h5 = f'./model/final_{model_name}.h5'
    
    try:
        # Keras 형식 우선 로드
        if os.path.exists(model_path_keras):
            model = load_model(model_path_keras)
            return model
        # H5 형식으로 대체 로드
        elif os.path.exists(model_path_h5):
            model = load_model(model_path_h5)
            return model
        else:
            # 파일이 없을 경우
            st.error(f"⚠️ **오류:** 모델 파일 ('final_{model_name}.keras' 또는 '.h5')을 찾을 수 없습니다!")
            return None
    except Exception as e:
        # 로드 중 기타 오류 발생
        st.exception(f"❌ **모델 로드 실패:** {e}")
        return None

def preprocess_image(image, target_size=IMAGE_SIZE):
    """PIL Image를 모델 입력 형식으로 전처리합니다."""
    # RGB 변환 (흑백 이미지 대비)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # 크기 조정 및 배열 변환
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0) # 배치 차원 추가
    return img_array


# --- 4. 메인 콘텐츠 레이아웃 ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("🖼️ 이미지 업로드")
    
    # 파일 업로더
    uploaded_file = st.file_uploader(
        "분류하고 싶은 차량 이미지를 선택하세요 (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    # 이미지 표시
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # 이미지 크기 및 캡션 명확화
        st.image(
            image, 
            caption=f"업로드된 이미지 (원본 크기: {image.size[0]}x{image.size[1]})", 
            use_column_width=True
        )
    else:
        # 업로드 대기 메시지
        st.info("이미지를 업로드하면, 오른쪽에 분류 결과가 표시됩니다.")

# --- 5. 분류 결과 표시 ---
with col2:
    st.header("💡 분류 및 예측 결과")
    
    if uploaded_file is not None:
        model = load_trained_model(model_choice)
        
        if model is not None:
            try:
                # 예측 과정 (스피너 사용으로 UX 개선)
                with st.spinner(f'**{model_choice}** 모델로 이미지를 분석 중입니다...'):
                    # 1. 전처리
                    img_array = preprocess_image(image)
                    
                    # 2. 예측
                    # TensorFlow에서 float32 numpy.float32를 사용할 때 Streamlit의 st.progress() 문제 방지 위해 float()으로 명시적 변환
                    predictions = model.predict(img_array, verbose=0)
                    predicted_index = np.argmax(predictions[0])
                    predicted_class = CLASS_NAMES[predicted_index]
                    confidence = float(predictions[0][predicted_index] * 100)
                    
                    # 3. 결과 표시
                    
                    # 최상위 결과 하이라이트
                    st.balloons() # 시각적 효과 추가
                    st.success(f"**🎉 최종 예측:** {predicted_class}")
                    st.metric(label="신뢰도", value=f"{confidence:.2f}%")
                    st.progress(confidence / 100)
                    
                    st.markdown("---")
                    
                    # 전체 클래스 확률 막대 그래프
                    st.subheader("📊 모든 클래스별 예측 확률")
                    
                    # 데이터프레임 대신 Streamlit의 요소들을 활용하여 깔끔한 표시
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob = float(predictions[0][i] * 100)
                        
                        # 가장 높은 확률 클래스는 별도로 강조 표시
                        if i == predicted_index:
                            st.markdown(f"**{class_name}: {prob:.2f}%**")
                            st.progress(prob / 100)
                        else:
                            st.write(f"{class_name}: {prob:.2f}%")
                            st.progress(prob / 100)
                            
            except Exception as e:
                st.error(f"❌ **예측 중 오류 발생:** {e}")
                # 오류 정보를 자세히 표시
                st.exception(e)
    else:
        st.warning("👈 이미지를 업로드해 주세요.")