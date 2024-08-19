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


> ViT 모델 정보
>> [dima806/facial_emotions_image_detection](https://huggingface.co/dima806/facial_emotions_image_detection)

> 수정 사항
>> (240707)
>> 1. 카메라가 행복과 놀람을 감지했을 때만 출력되는 게 아니라 계속 출력되도록 수정
>> 2. 게임 속도 개선


>> (240709)
>> 1. 게임 화면 중앙 상단에 감지한 표정의 텍스트 출력하도록
>> 2. 슬라이딩 하는 감정에 관한 논의

>> (2408)
>> 1. BGM 기능 추가(기본 음량 크기 50%)
>> 2. 키보드로 ESC 버튼을 눌렀을 때 계속하기, 그만두기, 소리 조절 버튼 추가


>> 이후 수정 계획
>> 1. 음성인식 기능
