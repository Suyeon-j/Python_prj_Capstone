# FindingCelebrity
나와 닮은 연예인 찾는 웹사이트

> 초안
> ![image](https://github.com/user-attachments/assets/1253c74d-c88a-4648-91b6-44cd9029d18f)
> 1. 좌측 사이드바에서 성별 선택
> 2. 저장된 사진을 불러올 것인지, 웹캠으로 사진 촬영할 것인지 선택
> 3. 결과 출력

> 초안 수정 사항
> 1. 기존: 저장된 사진을 가져오든, 사진을 새로 촬영하든 연예인별 평균 임베딩과 인풋 이미지 유사도 분석
>> 1-1. 무표정과 웃는 표정 두 가지를 인풋으로 변경(기존 함수: main.py에서 ```st.camera_input("촬영", label_visibility = "hidden")``` / 변경: capture_and_classify() 함수 생성 
>> 1-2. 각각 두 이미지를 데이터셋과 개별 임베딩으로 유사도 분석 (사용자의 웃는 표정과 무표정이 데이터셋의 연예인과 얼마나 유사한지 보기 위함)


수정
> 1. 기존 함수 cv2.VideoCapture(0) 교체
>> 이유: 느림

```bash
PJ/
 	 m_img/
 	     a.jpg
 	     b.jpg
      ...
 	 w_img/
 	     x.jpg
 	     y.jpg
      ...
 	 facenet/
       m_emb_mtcnn.csv
       w_emb_mtcnn.csv
   ...
   m_similarity_facial.py
   w_similarity_facial.py
   ...
```
