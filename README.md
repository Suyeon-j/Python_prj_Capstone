# ExpressionDino
졸업 작품 웹사이트 중 일부 기능인 표정 인식 공룡 게임을 서술한 문서

> 개요
![image](https://github.com/Suyeon-j/ExpressionDino/assets/66247203/b4e45e64-b3bf-4ac3-bba5-10947c21486c)
크롬 게임으로도 유명한 T-Rex 게임을 키보드 대신 표정을 통해 점프하거나 수그리도록 함(웃으면 점프, 놀란 표정이면 수그리도록)

> 참고 코드
>> link 1: [maxontech/chrome-dinosaur](https://github.com/maxontech/chrome-dinosaur)
- T-Rex 게임을 pygame을 사용하여 구현한 코드
- 게임의 전반적인 구조(전반적인 게임 구조와 공룡의 움직임 및 장애물 처리 클래스와 함수) 참고


>> link 2 : [insung-arc/openCV-Dino](https://github.com/insung-arc/openCV-Dino)
- 웹캠을 사용하여 사용자의 입이 열렸는지 감지하여 T-Rex 게임 구현
- 실시간 감지 루프와 비디오 스트림 처리, 웹캠을 사용하여 사용자의 입이 열렸는지 감지 등 주요 기능 참고


> 사용 ViT 모델
>> [dima806/facial_emotions_image_detection](https://huggingface.co/dima806/facial_emotions_image_detection)
- 입력 이미지 크기: 224x224
- ViTEmbeddings:
 1. 입력 이미지를 16x16 패치로 나누고 각 패치를 768차원의 임베딩으로 변환
 2. 위치 정보를 포함한 임베딩이 생성
 3. Dropout 레이어를 거쳐 최종 임베딩이 출력
- ViTEncoder:
 - 12개의 ViTLayer로 구성
 - 각 ViTLayer에서는 [Self-Attention 메커니즘](https://zayunsna.github.io/blog/2023-09-05-self_attention/)과 [Feed-Forward Neural Network](https://blog.naver.com/apr407/221238611771)가 적용
 - Self-Attention 메커니즘은 Query, Key, Value 행렬을 사용하여 임베딩 간의 관계를 학습
 - Feed-Forward Neural Network는 각 임베딩을 독립적으로 처리하여 특징을 추출
- 출력 레이어:
 - 인코더의 마지막 출력 임베딩(768차원)을 사용
 - 768차원의 출력을 가지는 Linear 레이어와 Tanh 활성화 함수로 구성(출력 레이어의 크기는 감정 클래스 수에 따라 달라질 수 있음)
