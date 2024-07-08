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
- 입력 이미지 크기: 224x224
- [self-attention](https://velog.io/@jhbale11/%EC%96%B4%ED%85%90%EC%85%98-%EB%A7%A4%EC%BB%A4%EB%8B%88%EC%A6%98Attention-Mechanism%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80)
```
ViTConfig {
  "_name_or_path": "dima806/facial_emotions_image_detection",
  "architectures": [
    "ViTForImageClassification"
  ],
  "attention_probs_dropout_prob": 0.0,
  "encoder_stride": 16,
  "hidden_act": "gelu", # gelu: https://arxiv.org/pdf/1606.08415
  "hidden_dropout_prob": 0.0,
  "hidden_size": 768,
  "id2label": {
    "0": "sad",
    "1": "disgust",
    "2": "angry",
    "3": "neutral",
    "4": "fear",
    "5": "surprise",
    "6": "happy"
  },
  "image_size": 224,
  "initializer_range": 0.02,
  "intermediate_size": 3072, # Encoder의 intermediate(=feed-forward) 차원 수
  "label2id": {
    "angry": 2,
    "disgust": 1,
    "fear": 4,
    "happy": 6,
    "neutral": 3,
    "sad": 0,
    "surprise": 5
  },
  "layer_norm_eps": 1e-12,
  "model_type": "vit",
  "num_attention_heads": 12,
  "num_channels": 3,
  "num_hidden_layers": 12,
  "patch_size": 16,
  "problem_type": "single_label_classification",
  "qkv_bias": true,
  "torch_dtype": "float32",
  "transformers_version": "4.41.2"
}
```
```
ViTForImageClassification(
  (vit): ViTModel(
    (embeddings): ViTEmbeddings(
      (patch_embeddings): ViTPatchEmbeddings(
        (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): ViTEncoder(
      (layer): ModuleList(
        (0-11): 12 x ViTLayer(
          (attention): ViTSdpaAttention(
            (attention): ViTSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): ViTSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): ViTIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ViTOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  )
  (classifier): Linear(in_features=768, out_features=7, bias=True)
)
```

> 수정해야할 점
>> (240707)
>> 1. 카메라가 행복과 놀람을 감지했을 때만 출력되는 게 아니라 계속 출력되도록 수정
>> 2. 게임 속도 개선
